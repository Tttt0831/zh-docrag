#!/usr/bin/env python
"""查询集清洗：剔除退化样本，并做唯一性检查补全真值。

生成阶段抽查发现两类必须处理的问题：

1. **退化样本**：模型把原文陈述句直接当成"查询"，query 与 answer 几乎相同
   （例：问「长期股权投资包括对被投资单位实施控制…」答同一句）。
   这不是真实检索意图，而且天然利好词法匹配，会把 BM25 的分数抬虚。

2. **真值不唯一**：查询不含可区分实体（例：「银行承兑票据期末终止确认金额」），
   40 家年报每家都有这一行。若只把出题那页记为真值，模型召回别家的正确页
   会被判错——这会系统性压低所有系统的分数，且对词法/语义系统的影响不对称。

处理：
* query 与 answer 相似度 > 0.8 → 丢弃
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


def norm(s):
    return re.sub(r"[\s,，]", "", s)


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
        for q in rows:
            a, qq = q["answer"].strip(), q["query"].strip()
            if len(norm(a)) < MIN_ANS:
                drop_short += 1
                continue
            if SequenceMatcher(None, norm(qq), norm(a)).ratio() > 0.8:
                drop_deg += 1
                continue
            na = norm(a)
            gold = [pid for pid, t in ref.items() if na in t]
            if not gold:
                gold = q["gold_pages"]           # 参照缺失时退回原始标注
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
              f"（退化 {drop_deg}，答案过短 {drop_short}，无区分度 {drop_broad}）"
              f"；其中 {multi} 条真值为多页", flush=True)


if __name__ == "__main__":
    main()
