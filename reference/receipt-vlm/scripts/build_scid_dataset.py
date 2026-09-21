# -*- coding: utf-8 -*-
"""
把 SCID 转成本项目的训练/评测数据集（清洗 + 字段补齐 + schema 对齐）。

schema 变更（2026-09）：tax_amount -> invoice_code
  tax_amount 在 SCID 的票种（交通票/定额发票）里覆盖率是 0%，留着等于让模型
  学一个真实数据里永远见不到的字段；发票代码覆盖率 77.6%，仅次于金额和发票
  号码，是中文票据查验真伪的核心字段，理应入选。

清洗规则都是从 SCID 实测出来的（gt.json 不是人工精校，带 OCR 噪声）：
  * 形近字误识：井→#、·→.、O→0、l/I→1、全角→半角
  * 两位年份：18-03-29 → 2018-03-29
  * 火车票日期带发车时间：2018年01月05日15:16开 → 2018-01-05
  * 占位符 XXX / # / ## → null
  实测日期可解析率 65.8% → 94.4%（救回 1874 条），金额 97.8% → 98.9%。

merchant_name / tax_id 从 ocr.json 抽取（gt.json 没有这两个字段）。
注意票面上排最前的公司名往往是**发票印刷厂**而非开票方，必须按关键词排除。

用法:
  python scripts/build_scid_dataset.py --split Train --out data/scid/train.jsonl
"""
import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from scripts.scid_enrich import TAXID, pick_merchant   # noqa
from scripts.scid_to_ours import parse_amount as _pa, PLACEHOLDER   # noqa

OCRFIX = str.maketrans({'井': '#', '·': '.', '．': '.', '：': ':', '，': ',',
                        'O': '0', 'o': '0', 'l': '1', 'I': '1',
                        '￥': '¥', '—': '-', '－': '-'})

PROMPT = ("<image>请从这张票据中抽取以下字段并以 JSON 输出:\n"
          "merchant_name, date, total_amount, invoice_code, tax_id, invoice_no。\n"
          "缺失字段填 null,不要编造。")


def receipt_type(raw: dict) -> str:
    """
    从 gt.json 的字段组合推断票种。

    SCID 没有给票种标签，但字段组合本身就是指纹：定额发票没有日期栏、
    火车票没有发票代码——这不是标注缺失，是票面上真的没有。按票种分组
    报指标比一个笼统的 F1 有诊断价值得多。
    """
    k = set(raw)
    if {'始发站', '到达站'} <= k:
        return 'train_bus'      # 火车票/汽车票
    if {'入口', '出口'} & k:
        return 'toll'           # 高速通行费
    if '日期' in k:
        return 'machine'        # 机打发票
    return 'quota'              # 定额发票


def _clean(s):
    if s is None:
        return None
    s = str(s).translate(OCRFIX).strip()
    return re.sub(r'[\s　]+', '', s)


def clean_date(s):
    c = _clean(s)
    if not c or c in PLACEHOLDER or '#' in c:
        return None
    c = re.sub(r'\d{1,2}:\d{2}.*$', '', c)          # 火车票「15:16开」
    c = re.sub(r'[年月]', '-', c).rstrip('日-')
    m = re.fullmatch(r'(\d{2}|\d{4})[-/.](\d{1,2})[-/.](\d{1,2})', c)
    if not m:
        # 8 位纯数字 YYYYMMDD——scid_to_ours 里处理了，这里漏掉了，
        # 实测占「清洗后仍失败」的一大半。手写规则的长尾就是这样漏的。
        m8 = re.fullmatch(r'(\d{4})(\d{2})(\d{2})', c)
        if not m8:
            return None
        m = m8
    y, mo, d = m.groups()
    y = int(y)
    y = y + 2000 if y < 100 else y
    if 2000 <= y <= 2100 and 1 <= int(mo) <= 12 and 1 <= int(d) <= 31:
        return f'{y:04d}-{int(mo):02d}-{int(d):02d}'
    return None


def clean_amount(s):
    c = _clean(s)
    if not c or c in PLACEHOLDER or '#' in c:
        return None
    return _pa(c.replace('¥', '').replace(':', ''))


def clean_code(s):
    c = _clean(s)
    if not c or c in PLACEHOLDER or '#' in c:
        return None
    return c if re.fullmatch(r'[0-9A-Za-z]{4,20}', c) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scid", default="data/SCID_new")
    ap.add_argument("--split", default="Train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=0)
    args = ap.parse_args()

    base = REPO / args.scid / args.split
    gt = json.loads((base / "gt.json").read_text(encoding="utf-8"))
    ocr = json.loads((base / "ocr.json").read_text(encoding="utf-8"))
    img_root = base / "images"
    index = {p.name: p for p in img_root.rglob("*.jpg")}
    index.update({p.name: p for p in img_root.rglob("*.png")})

    out = Path(args.out)
    out = out if out.is_absolute() else REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)

    st = {k: 0 for k in ("n", "no_img", "merchant", "tax_id", "date",
                         "amount", "code", "invno", "empty")}
    with open(out, "w", encoding="utf-8") as f:
        for k, v in gt.items():
            if args.max and st["n"] >= args.max:
                break
            p = index.get(k)
            if p is None:
                st["no_img"] += 1
                continue
            ca = ocr.get(k, {}).get("content_ann", {})
            texts, bbs = ca.get("texts", []), ca.get("bboxes", [])
            merchant, _ = pick_merchant(texts, bbs)
            blob = " ".join(t for t in texts if isinstance(t, str))
            tm = TAXID.search(blob)
            tgt = {
                "merchant_name": merchant,
                "date": clean_date(v.get("日期")),
                "total_amount": clean_amount(v.get("金额")),
                "invoice_code": clean_code(v.get("发票代码")),
                "tax_id": tm.group(1) if tm else None,
                "invoice_no": clean_code(v.get("发票号码")),
            }
            if all(x is None for x in tgt.values()):
                st["empty"] += 1
                continue
            for key, sk in (("merchant_name", "merchant"), ("tax_id", "tax_id"),
                            ("date", "date"), ("total_amount", "amount"),
                            ("invoice_code", "code"), ("invoice_no", "invno")):
                if tgt[key] is not None:
                    st[sk] += 1
            f.write(json.dumps({"image_path": str(p), "prompt": PROMPT,
                                "target_json": tgt, "scid_raw": v,
                                "receipt_type": receipt_type(v),
                                # 四个核心字段来自 SCID gt.json（竞赛标注，权威）；
                                # merchant_name / tax_id 是本项目从 ocr.json 正则抽取，
                                # 非官方标注，报结果时必须声明来源。
                                "field_source": {"date": "gt", "total_amount": "gt",
                                                 "invoice_code": "gt", "invoice_no": "gt",
                                                 "merchant_name": "derived",
                                                 "tax_id": "derived"},
                                "template": "scid_" + args.split},
                               ensure_ascii=False) + "\n")
            st["n"] += 1

    n = max(st["n"], 1)
    print(f"[{args.split}] 写出 {st['n']} 条 -> {out}")
    print(f"  图片缺失 {st['no_img']}  全字段为空已丢弃 {st['empty']}")
    print(f"  {'字段':<16}{'有值':>8}{'覆盖率':>9}")
    for label, sk in (("merchant_name", "merchant"), ("date", "date"),
                      ("total_amount", "amount"), ("invoice_code", "code"),
                      ("tax_id", "tax_id"), ("invoice_no", "invno")):
        print(f"  {label:<16}{st[sk]:>8}{st[sk]*100/n:>8.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
