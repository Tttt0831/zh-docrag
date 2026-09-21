# -*- coding: utf-8 -*-
"""
用 DeepSeek API 把 SCID 标注补齐到本项目 6 字段 schema，成本最小化。

分层，让免费的部分承担绝大多数工作：
  L1  date / total_amount / invoice_code / invoice_no
      → 直接取 SCID gt.json（CSIG 2022 竞赛标注），零成本
  L2  merchant_name / tax_id → ocr.json 正则抽，零成本
  L3  正则抽不出、但 OCR 文本里**确实存在候选**的，才调 API

L3 的门控是成本的关键。朴素做法是「凡是缺字段就问模型」，那是 90% 的样本；
但其中绝大多数票面上压根没有开票方——定额发票的商户只在印章里（OCR 只能
转出「发票专用章」四个字）、火车票/汽车票没有公司抬头、过路费票只有收费站
名。对这些票问模型，答案必然是 null，纯属花钱买空气。

所以门控要求「有真候选」：
  * merchant：OCR 里匹配到公司名模式，且**排除印刷厂后仍有剩余**。
    票面底部普遍印着发票印制单位（浙江中瑞印业、贵州金黔印务、苏印总厂…），
    实测这是候选榜前列，不排掉的话门控形同虚设。
  * tax_id：**单个文本框内**有严格 18 位统一社会信用代码。必须按框匹配：
    把所有框拼成一条串再正则，相邻框的数字会拼出假的 18 位码（实测前 60 条
    触发税号门控的 18 条，全部是跨框拼出来的假候选，一条真的都没有）。
    也不放宽到 17/19 位去让模型«猜»被 OCR 吞掉的字符——那是编造。

实测：门控前 90.2% 需调 API，门控后 9.7%（40617 → 3956 条）。

L3 还有两处省钱设计：
  * 不发图片，只发 ocr.json 的文本框。「哪个文本框是开票方」是纯文本推理，
    不需要视觉。实测输入 token 736 → 230。
  * 并发而非打包。试过把多张票塞进一次请求摊薄 reasoning 开销，结果是反效果：
    max_tokens 是**每次请求**的总预算，推理量随票数线性上涨，打包后大面积截断。
    实测 bs=1 解析 20/20，bs=5 只有 5/20，bs=10 只有 10/20。所以保持单条请求
    （100% 解析率），靠多线程把 8 小时压到 1 小时。

用法:
  export DEEPSEEK_API_KEY=...          # 只走环境变量，不写进文件
  python scripts/ds_annotate.py --split testB --out data/scid/testB_ds.jsonl
  python scripts/ds_annotate.py --split testB --dry-run    # 只算账不花钱
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

API = "https://api.deepseek.com/chat/completions"
PRICE_IN, PRICE_OUT = 2.0, 8.0          # CNY / 百万 token

# 门控用的候选模式。比 scid_enrich.COMPANY 宽一点（这里只用来判断「值不值得问」，
# 宽一点只是多花一点钱，窄了会漏掉真商户）
CAND_CO = re.compile(
    r'[一-龥()（）]{2,20}?'
    r'(公司|厂|店|中心|超市|酒店|宾馆|饭店|医院|学校|集团|管理处|客运站|汽车站'
    r'|收费站|加油站|停车场|服务区|高速公路|运输队|事务所|管理所)')
CAND_TAX = re.compile(r'9[1-5][0-9A-HJ-NPQRTUWXY]{16}')
# 发票印制单位，不是开票方
PRINTER_KW = ('印刷', '印务', '印业', '票证', '票据印', '磁卡', '印制', '印总厂',
              '财税', '税务')

HEAD = ("你是票据信息抽取器。下面有若干张中国票据，每张以 ### <编号> 开头，"
        "后跟该票 OCR 出的全部文字（顺序已打乱，可能有错别字或粘连）。\n"
        "对每张票判断【开票方（收款单位）名称全称】和【纳税人识别号】（18位，9开头）。\n"
        "规则：\n"
        "1. 票面底部的发票印制单位（含 印刷/印务/印业/票证/磁卡 等字样）不是开票方。\n"
        "2. 「国家税务总局」「XX省税务局」是监制单位，不是开票方。\n"
        "3. 信息不存在就填 null，不要从别的字段推断，不要编造。\n"
        "4. 税号必须是票面上真实出现的 18 位字符，OCR 残缺就填 null。\n"
        "只输出 {n} 行 JSON，每行一个，不要解释、不要代码块：\n"
        '{{"id":1,"merchant_name":null,"tax_id":null}}\n\n')


def load_split(split: str):
    """返回 (rows, ocr)。SCID 训练集目录名是 Train，其余同名。"""
    d = {"train": "Train"}.get(split, split)
    ocr = json.loads((REPO / f"data/SCID_new/{d}/ocr.json").read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in
            (REPO / f"data/scid/{split}.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    return rows, ocr


def texts_of(row, ocr):
    k = Path(row["image_path"]).name
    return [t for t in ocr.get(k, {}).get("content_ann", {}).get("texts", [])
            if isinstance(t, str)]


def gate(row, texts):
    """返回 (need_merchant, need_tax)。只有 OCR 里真有候选才值得问。

    候选一律**按单个文本框**匹配，不能把所有框拼成一条串再搜——相邻框会拼出
    不存在的公司名和税号，那是纯粹的浪费。
    """
    tgt = row["target_json"]
    boxes = [t.replace(" ", "") for t in texts]
    need_m = need_t = False
    if not tgt.get("merchant_name"):
        need_m = any(not any(p in c for p in PRINTER_KW)
                     for b in boxes for c in (m.group(0) for m in CAND_CO.finditer(b)))
    if not tgt.get("tax_id"):
        need_t = any(CAND_TAX.search(b) for b in boxes)
    return need_m, need_t


def call(prompt, key, model, max_tokens, retries=4):
    body = json.dumps({"model": model, "max_tokens": max_tokens, "temperature": 0.0,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    for i in range(retries):
        try:
            req = urllib.request.Request(
                API, data=body,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as r:
                d = json.load(r)
            txt = (d["choices"][0]["message"].get("content") or "").strip()
            u = d.get("usage", {})
            # content 为空但 HTTP 200：推理 token 吃满了 max_tokens。这是
            # deepseek-flash 最容易被误判成成功的失败模式，必须显式识别。
            return txt, u, ("" if txt else "empty")
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and i < retries - 1:
                time.sleep(2 ** i * 3)
                continue
            return "", {}, f"http{e.code}"
        except Exception as e:                       # 超时/连接错误
            if i < retries - 1:
                time.sleep(2 ** i * 3)
                continue
            return "", {}, type(e).__name__
    return "", {}, "retry_exhausted"


def parse_lines(txt, n):
    """解析每行一个 JSON，返回 {id: (merchant, tax)}。宽容缺行/多余文字。"""
    out = {}
    for m in re.finditer(r'\{[^{}]*\}', txt):
        try:
            d = json.loads(m.group(0))
        except Exception:
            continue
        try:
            i = int(d.get("id"))
        except (TypeError, ValueError):
            continue
        if not 1 <= i <= n:
            continue

        def norm(x):
            if x is None:
                return None
            x = str(x).strip()
            return None if x.upper() in ("NONE", "NULL", "", "N/A") else x
        out[i] = (norm(d.get("merchant_name")), norm(d.get("tax_id")))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="testB")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="最多送多少条进 API（0=全部）")
    ap.add_argument("--batch", type=int, default=1,
                    help="每次请求打包几张票。别调大：max_tokens 是每请求总预算，"
                         "打包会导致推理截断（实测 bs=5 解析率从 100%% 掉到 25%%）")
    ap.add_argument("--workers", type=int, default=8, help="并发请求数")
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--max-tokens", type=int, default=6000,
                    help="推理模型，这是 reasoning+content 的总预算，给不够 content 就是空串")
    ap.add_argument("--budget", type=float, default=0.0,
                    help="花费上限（元），超了就停并保存已完成的部分。0=不限")
    ap.add_argument("--dry-run", action="store_true", help="只统计需调 API 的条数，不花钱")
    args = ap.parse_args()

    rows, ocr = load_split(args.split)
    todo, texts_map = [], {}
    for idx, r in enumerate(rows):
        tx = texts_of(r, ocr)
        nm, nt = gate(r, tx)
        if nm or nt:
            todo.append(idx)
            texts_map[idx] = tx

    print(f"{args.split}: 共 {len(rows)} 条，门控后需调 API {len(todo)} 条 "
          f"({len(todo)/len(rows)*100:.1f}%)")
    if args.dry_run:
        return 0
    if args.limit:
        todo = todo[:args.limit]

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("请先 export DEEPSEEK_API_KEY=...（不要写进文件）")

    st = {"api": 0, "hit_m": 0, "hit_t": 0, "in": 0, "out": 0, "miss": 0}
    fails = {}
    lock = Lock()
    t0 = time.time()
    stopped = {"v": False}

    def work(s0):
        if stopped["v"]:
            return
        chunk = todo[s0:s0 + args.batch]
        parts = [f"### {i}\n" + "\n".join(texts_map[idx])
                 for i, idx in enumerate(chunk, 1)]
        txt, u, err = call(HEAD.format(n=len(chunk)) + "\n".join(parts),
                           key, args.model, args.max_tokens)
        got = parse_lines(txt, len(chunk))
        with lock:
            st["api"] += 1
            st["in"] += u.get("prompt_tokens", 0)
            st["out"] += u.get("completion_tokens", 0)
            if err:
                fails[err] = fails.get(err, 0) + 1
            st["miss"] += len(chunk) - len(got)
            for i, idx in enumerate(chunk, 1):
                m, t = got.get(i, (None, None))
                r = rows[idx]
                src = r.setdefault("anno_source", {})
                if m and not r["target_json"].get("merchant_name"):
                    r["target_json"]["merchant_name"] = m
                    src["merchant_name"] = "deepseek"
                    st["hit_m"] += 1
                # 模型可能把 OCR 残缺的税号"补全"成 18 位——那是编造。只接受
                # 确实逐字出现在 OCR 原文里的税号。
                if t and not r["target_json"].get("tax_id"):
                    if CAND_TAX.fullmatch(t) and t in "".join(texts_map[idx]).replace(" ", ""):
                        r["target_json"]["tax_id"] = t
                        src["tax_id"] = "deepseek"
                        st["hit_t"] += 1
            cost = st["in"] / 1e6 * PRICE_IN + st["out"] / 1e6 * PRICE_OUT
            n_done = st["api"] * args.batch
            if st["api"] % 50 == 0:
                eta = (time.time() - t0) / max(n_done, 1) * (len(todo) - n_done)
                print(f"  {min(n_done,len(todo))}/{len(todo)}  ¥{cost:.2f}  "
                      f"补出 m={st['hit_m']} t={st['hit_t']}  未解析 {st['miss']}  "
                      f"ETA {eta/60:.0f}min", flush=True)
            if args.budget and cost >= args.budget and not stopped["v"]:
                stopped["v"] = True
                print(f"  已达预算上限 ¥{args.budget}，停止提交新请求", flush=True)

    starts = list(range(0, len(todo), args.batch))
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, starts))
    stopped = stopped["v"]

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    cost = st["in"] / 1e6 * PRICE_IN + st["out"] / 1e6 * PRICE_OUT
    per = cost / max(st["hit_m"] + st["hit_t"], 1)
    print(f"\n写出 {len(rows)} 条 -> {out}{'（预算中断，部分完成）' if stopped else ''}")
    print(f"  API 请求 {st['api']} 次（batch={args.batch}）")
    print(f"  补出 merchant={st['hit_m']}  tax_id={st['hit_t']}  未解析 {st['miss']}")
    print(f"  失败分布: {fails or '无'}")
    print(f"  token in={st['in']:,} out={st['out']:,}   花费 ¥{cost:.2f}  (¥{per:.4f}/补出字段)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
