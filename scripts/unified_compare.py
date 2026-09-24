#!/usr/bin/env python
"""受控对比：同一个模型，两种页面表示。

为什么要有这个脚本
------------------
第一轮 MVP 用 BGE-m3 编码解析文本、用 ColQwen2.5 编码页面图，得到
解析式 0.892 / 视觉式 1.000。但那个数字**不能用来回答项目的问题**：
两条链路一次变了两个东西——表示方式不同，编码模型也不同，
而且视觉侧的底座大五倍、新一年多（BGE-m3 5.7亿/XLM-R/2024 vs
ColQwen2.5 30亿/Qwen2.5-VL/2025）。差距来自「图保住了版面」还是
「模型更大」，无法区分。

这里换成 Qwen3-VL-Embedding-2B —— 同一套权重既能编码文字也能编码图片。
于是：

    同一模型 ← 解析出的文字  → 一组分数
    同一模型 ← 同一页的图片  → 另一组分数

唯一的变量就是表示方式。

仍然保留的不对称（如实记录，不掩盖）
------------------------------------
* 文本侧是「每页多个块，取最大值聚合到页」，图片侧是「每页一个向量」。
  这是解析式的固有产物（它就是会把页面切成块），不是我引入的偏差，
  但聚合方式不同本身会影响结果，必须写明。
* 文本块长度未调优，用的是 DeepDoc 默认切法。
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import mrr, ndcg_at_k, recall_at_k  # noqa: E402


def local_snapshot(repo_id: str) -> str:
    """用本地快照路径而不是 repo id。

    教训：transformers 4.57.6 按 repo id 解析 additional_chat_templates/ 下的
    模板文件时会返回 None，随后 open(None) 崩在 processing_utils.py 里，
    报错信息完全看不出真正原因。给本地路径可绕过那套解析。
    """
    pat = str(Path.home() / f".cache/huggingface/hub/models--{repo_id.replace('/', '--')}/snapshots/*/")
    snaps = sorted(glob.glob(pat))
    if not snaps:
        raise SystemExit(f"未找到 {repo_id} 的本地快照，先 hf download")
    return snaps[0]


def evaluate(name, score_fn, qs, pages, out_rows):
    agg = {}
    for i, q in enumerate(qs):
        page_score = score_fn(i)
        ranked = [p for p, _ in sorted(page_score.items(), key=lambda kv: -kv[1])]
        gold = q["gold_pages"]
        nd = ndcg_at_k(ranked, gold, 5)
        r1 = recall_at_k(ranked, gold, 1)
        r5 = recall_at_k(ranked, gold, 5)
        mr = mrr(ranked, gold)
        agg.setdefault(q["stratum"], []).append((nd, r1, mr, r5))
        out_rows.append((name, q["qid"], q["stratum"], gold, nd, r1, mr, ranked[:3]))
    return agg


def summarize(name, agg):
    print(f"  --- {name} ---")
    order = ["跨页三级表头", "多级表头", "普通表格", "纯文字"]
    for k in [x for x in order if x in agg] + [x for x in agg if x not in order]:
        v = agg[k]
        n = len(v)
        print(f"   {k:14s} n={n}  nDCG@5={sum(x[0] for x in v)/n:.3f}  R@1={sum(x[1] for x in v)/n:.3f}  MRR={sum(x[2] for x in v)/n:.3f}")
    al = [x for v in agg.values() for x in v]
    n = len(al)
    res = (sum(x[0] for x in al)/n, sum(x[1] for x in al)/n, sum(x[2] for x in al)/n)
    print(f"   {'总体':14s} n={n}  nDCG@5={res[0]:.3f}  R@1={res[1]:.3f}  MRR={res[2]:.3f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-Embedding-2B")
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    chunks = [json.loads(l) for l in open(REPO / "data/parsed/pages_50_80.jsonl")]
    qs = [json.loads(l) for l in open(REPO / "data/queries_mvp.jsonl")]
    imgs = sorted(glob.glob(str(REPO / "data/pages/*.png")))
    pages = sorted({c["page"] for c in chunks})
    img_page = [int(Path(p).stem.split("-")[1]) for p in imgs]
    print(f"  文本块 {len(chunks)}  页面图 {len(imgs)}  查询 {len(qs)}  页码 {min(pages)}~{max(pages)}", flush=True)

    t0 = time.time()
    path = local_snapshot(args.model)
    model = SentenceTransformer(path, model_kwargs={"dtype": torch.bfloat16}, device="cuda:0")
    print(f"  模型加载 {time.time()-t0:.1f}s  （{args.model}）", flush=True)

    t1 = time.time()
    qe = model.encode([q["query"] for q in qs], batch_size=args.batch,
                      prompt="Retrieve the document page that answers the query.",
                      convert_to_tensor=True, normalize_embeddings=True)
    print(f"  查询编码 {time.time()-t1:.1f}s  维度 {tuple(qe.shape)}", flush=True)

    t2 = time.time()
    te = model.encode([c["text"] for c in chunks], batch_size=args.batch,
                      convert_to_tensor=True, normalize_embeddings=True)
    dt_text = time.time() - t2
    print(f"  解析文本编码 {dt_text:.1f}s  {tuple(te.shape)}", flush=True)

    t3 = time.time()
    ie = model.encode([{"image": p} for p in imgs], batch_size=args.batch,
                      convert_to_tensor=True, normalize_embeddings=True)
    dt_img = time.time() - t3
    print(f"  页面图编码 {dt_img:.1f}s  {tuple(ie.shape)}", flush=True)

    sim_t = (qe @ te.T).float().cpu()
    sim_i = (qe @ ie.T).float().cpu()
    by_page = {p: [i for i, c in enumerate(chunks) if c["page"] == p] for p in pages}

    rows = []
    agg_t = evaluate("解析式", lambda i: {p: max(sim_t[i, j].item() for j in ix)
                                          for p, ix in by_page.items() if ix}, qs, pages, rows)
    agg_i = evaluate("视觉式", lambda i: {pg: sim_i[i, j].item()
                                          for j, pg in enumerate(img_page)}, qs, pages, rows)

    print("  模型  qid  分层            真值        nDCG@5  R@1    top3")
    for name, qid, st, gold, nd, r1, mr, top in rows:
        print(f"  {name}  {qid}  {st:14s} {str(gold)[:10]:10s} {nd:.3f}  {r1:.3f}  {top}")
    rt = summarize("解析式（同模型，输入=解析文本）", agg_t)
    ri = summarize("视觉式（同模型，输入=页面图）", agg_i)
    print(f"  ### 表示方式带来的差距 nDCG@5: {ri[0]-rt[0]:+.3f}   R@1: {ri[1]-rt[1]:+.3f}")
    print(f"  ### 编码耗时: 文本 {dt_text:.1f}s / 图片 {dt_img:.1f}s")

    json.dump({"rows": [[r[0], r[1], r[2], r[4], r[5], r[6]] for r in rows],
               "text_overall": rt, "image_overall": ri,
               "enc_sec": {"text": dt_text, "image": dt_img}},
              open(REPO / "data/unified_results.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
