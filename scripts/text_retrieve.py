#!/usr/bin/env python
"""解析式一侧：BGE-m3 编码解析出的文本块，余弦检索，页面级聚合。

与视觉式共用 scripts/metrics.py，保证两边算的是同一个数。
页面得分 = 该页所有块得分的最大值（max-pooling），这是文本检索里
把块级得分聚到文档级最常见的做法，也和视觉式的 MaxSim 精神一致。
"""
import json, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import ndcg_at_k, recall_at_k, mrr  # noqa: E402

MODEL = "BAAI/bge-m3"


def embed(texts, tok, model, bs=8, max_len=512):
    outs = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(model.device)
        with torch.no_grad():
            h = model(**b).last_hidden_state[:, 0]      # BGE-m3 稠密向量取 CLS
        outs.append(F.normalize(h, dim=-1).cpu())
    return torch.cat(outs)


def main():
    chunks = [json.loads(l) for l in open(REPO / "data/parsed/pages_50_80.jsonl")]
    qs = [json.loads(l) for l in open(REPO / "data/queries_mvp.jsonl")]
    print(f"  文本块 {len(chunks)}  查询 {len(qs)}", flush=True)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda:0").eval()
    print(f"  BGE-m3 加载 {time.time()-t0:.1f}s", flush=True)

    t1 = time.time()
    ce = embed([c["text"] for c in chunks], tok, model)
    qe = embed([q["query"] for q in qs], tok, model)
    print(f"  编码耗时 {time.time()-t1:.1f}s", flush=True)

    sims = qe @ ce.T                                     # 余弦（已归一化）
    pages = sorted({c["page"] for c in chunks})
    by_page = {p: [i for i, c in enumerate(chunks) if c["page"] == p] for p in pages}

    rows, agg = [], {}
    for i, q in enumerate(qs):
        page_score = {p: max(sims[i, idx].item() for idx in ix) for p, ix in by_page.items() if ix}
        ranked = [p for p, _ in sorted(page_score.items(), key=lambda kv: -kv[1])]
        gold = q["gold_pages"]
        nd, r1, mr = ndcg_at_k(ranked, gold, 5), recall_at_k(ranked, gold, 1), mrr(ranked, gold)
        r5 = recall_at_k(ranked, gold, 5)
        agg.setdefault(q["stratum"], []).append((nd, r1, mr, r5))
        rows.append((q["qid"], q["stratum"], gold, nd, r1, mr, ranked[:3]))

    print("  qid  分层            真值      nDCG@5  R@1    MRR    top3")
    for qid, st, gold, nd, r1, mr, top in rows:
        print(f"  {qid}  {st:14s} {str(gold)[:10]:10s} {nd:.3f}  {r1:.3f}  {mr:.3f}  {top}")
    print("  --- 按分层 ---")
    for k, v in agg.items():
        n = len(v)
        print(f"   {k:14s} n={n}  nDCG@5={sum(x[0] for x in v)/n:.3f}  R@1={sum(x[1] for x in v)/n:.3f}  MRR={sum(x[2] for x in v)/n:.3f}")
    al = [x for v in agg.values() for x in v]
    print(f"   {'总体':14s} n={len(al)}  nDCG@5={sum(x[0] for x in al)/len(al):.3f}  R@1={sum(x[1] for x in al)/len(al):.3f}  MRR={sum(x[2] for x in al)/len(al):.3f}")
    json.dump({"rows": [[r[0], r[1], r[3], r[4], r[5]] for r in rows]}, open(REPO / "data/text_results.json", "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
