#!/usr/bin/env python
"""查询集清洗：剔除退化样本，并做唯一性检查补全真值。

生成阶段抽查发现两类必须处理的问题：

1. **退化样本**：模型把原文陈述句直接当成"查询"，query 与 answer 几乎相同
   （例：问「长期股权投资包括对被投资单位实施控制…」答同一句）。
   这不是真实检索意图，而且天然利好词法匹配，会把 BM25 的分数抬虚。

2. **真值不唯一**：查询不含可区分实体（例：「银行承兑票据期末终止确认金额」），
   40 家年报每家都有这一行。若只把出题那页记为真值，模型召回别家的正确页
   会被判错——这会系统性压低所有系统的分数，且对词法/语义系统的影响不对称。

3. **答案是条文而非值**（2026-09-27 smoke 抽查发现）：会计政策页上模型会把
   「怎么做账」的准则原文抄下来当答案，再把它倒装成一句「……金额是多少」。
   这类样本骗过了全部三道关——接地校验满分（确实逐字出现在页上），
   退化检测也不触发（改写成问句后字面差很远）。而 40 家年报的会计政策章节
   是同一套准则的复述，真值根本不唯一，且对词法系统的误导最大。

处理：
* 答案剥掉开头列表编号后，长度 < MIN_ANS 或 > MAX_ANS → 丢弃
  （长度是这类条文样本唯一靠得住的信号）
* 同一 query 文本只保留第一条
* 答案以「详见/参见」开头（指引不是值）→ 丢弃
* query 剥掉公司名前缀后与 answer 相似度 > 0.8，或答案原样出现在 query 里 → 丢弃
* 答案里的数字千分位格式不合法（「255,3」）→ 丢弃，是窄列表格折行留下的残片
* 答案串在多少页出现，就把这些页全记为真值（用 pdfplumber 参照文本判定，
  独立于两条路线）
* 若命中页数 > MAX_GOLD，说明该查询完全不具区分性 → 丢弃
"""
import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MAX_GOLD = 8
MIN_ANS = 4
MAX_ANS = 30          # 超过即是散文而非可定位的值，见下方第 3 条


def norm(s):
    return re.sub(r"[\s,，]", "", s)


def strip_prefix(query: str, name: str) -> str:
    """剥掉查询开头的公司名再算退化相似度。

    「强制带公司名」是为了让查询有区分度，但它同时把退化样本的相似度压了下来：
    实测「中粮科技期末公司已质押的应收款项融资」对答案「(4) 期末公司已质押的应收款项融资」
    只算出 0.80，刚好躲过 >0.8 的阈值——而这条 query 就是照抄页面小标题。
    """
    q = norm(query)
    n = norm(name)
    return q[len(n):] if n and q.startswith(n) else q


def strip_marker(answer: str) -> str:
    """剥掉答案开头的列表编号，如 "(4) "、"（三）"、"1. "。

    参照文本里这些编号与正文之间的空格情况不一致，不剥就会匹配不到任何页，
    于是退回单页标注，唯一性检查形同虚设。
    """
    # (?!\d)：编号后不能紧跟数字，否则会把小数当编号剥掉（实测「710.26万元」被剥成「26万元」）
    return re.sub(r"^[(（]?[0-9一二三四五六七八九十]{1,3}[)）.、](?!\d)\s*", "", answer.strip())


# 千分位逗号后必须恰好 3 位数字。竖排/窄列表格里长数字会被折成两行，
# PDF 内嵌文本本身就是「255,3 479,7 9,922 42,51」这样的残片（pdfplumber 原样抽出），
# 模型照抄后接地校验满分，但这不是一个值。前后边界检查拦不住它——残片后面确实跟着空格。
MALFORMED_NUM = re.compile(r"\d,(?!\d{3}(?!\d))")


POINTER = re.compile(r"^(详见|参见|见本|请见|具体见)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", default=["image", "text"])
    args = ap.parse_args()

    ref = {}
    for l in open(REPO / "data/corpus/ref_text.jsonl"):
        r = json.loads(l)
        if not r.get("garbled"):
            ref[r["page_id"]] = norm(r["text"])
    print(f"  参照页 {len(ref)}", flush=True)

    for src in args.inputs:
        fin = REPO / f"data/corpus/queries_{src}.jsonl"
        if not fin.exists():
            print(f"  跳过 {src}（文件不存在）", flush=True)
            continue
        rows = [json.loads(l) for l in open(fin)]
        kept, drop_deg, drop_short, drop_broad = [], 0, 0, 0
        drop_long = drop_dup = drop_inq = drop_ptr = drop_frag = 0
        seen_q = set()
        for q in rows:
            a, qq = strip_marker(q["answer"]), q["query"].strip()
            if len(norm(a)) < MIN_ANS:
                drop_short += 1
                continue
            # 「详见本节五、……」是指路不是值，答对了也没有检索意义
            if POINTER.match(a):
                drop_ptr += 1
                continue
            # 答案过长 = 模型抄了一段条文而不是答某个值。实测「……金额是多少」
            # 配一段 89 字的准则原文，接地校验反而满分（确实逐字出现在页上），
            # 唯一靠得住的信号就是长度。
            if len(norm(a)) > MAX_ANS:
                drop_long += 1
                continue
            # 同一问题的两种措辞会被算两次分，且至少一条的答案是错的
            if norm(qq) in seen_q:
                drop_dup += 1
                continue
            seen_q.add(norm(qq))
            if SequenceMatcher(None, strip_prefix(qq, q.get("name", "")), norm(a)).ratio() > 0.8:
                drop_deg += 1
                continue
            # 答案写在问题里（「优利德在全球有近400家经销商」→「近400家」）：
            # 整串相似度测不出来，因为答案只是问题的一小段
            if norm(a) in norm(qq):
                drop_inq += 1
                continue
            q["answer"] = a
            na = norm(a)
            if MALFORMED_NUM.search(re.sub(r"\s", "", a)):
                drop_frag += 1
                continue
            gold = [pid for pid, t in ref.items() if na in t]
            if not gold:
                gold = q["gold_pages"]           # 参照缺失（乱码页）时退回原始标注
            if len(gold) > MAX_GOLD:
                drop_broad += 1
                continue
            q["gold_pages"] = gold
            q["n_gold"] = len(gold)
            kept.append(q)
        fout = REPO / f"data/corpus/queries_{src}_clean.jsonl"
        with fout.open("w", encoding="utf-8") as f:
            for q in kept:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
        multi = sum(1 for q in kept if q["n_gold"] > 1)
        print(f"  {src}: {len(rows)} → 保留 {len(kept)}"
              f"（退化 {drop_deg}，答案过短 {drop_short}，答案过长 {drop_long}，"
              f"重复 {drop_dup}，答案在问题里 {drop_inq}，指引 {drop_ptr}，"
              f"残片 {drop_frag}，无区分度 {drop_broad}）"
              f"；其中 {multi} 条真值为多页", flush=True)


if __name__ == "__main__":
    main()
