#!/usr/bin/env python
"""五系统对比评测 + 配对自助法统计检验。

系统
----
A   解析式·稠密      DeepDoc 默认切块 → Qwen3-VL-Emb
A′  解析式·按行切块   表格按 <tr> 拆行 → Qwen3-VL-Emb   （消融）
B   解析式·词法      BM25（字符 bigram）
C   解析式·混合      A 与 B 的 RRF 融合
D   视觉式·单向量    页面图 → Qwen3-VL-Emb          （与 A 构成受控对）
E   视觉式·多向量    页面图 → ColQwen2.5 MaxSim      （视觉式上限）

两个关键对比
------------
A vs D：同一套权重、同样内容、唯一变量是表示形式 → 表示方式本身的影响
C vs E：两条路线各自最强配置 → 实际选型该选哪条

统计
----
per-query nDCG@5 的**配对自助法**：对查询重采样 10000 次，给出差值的 95% 置信区间
和 p 值。MVP 那个 +0.065 到底是不是噪声，只有这一步能回答。

融合用 RRF（倒数排名融合）而不是分数加权：BM25 分和余弦相似度量纲不同，
直接加权需要调一个归一化超参，那会把「融合调得好不好」混进结论。
RRF 只用排名，无超参（k=60 是文献标准值）。
"""
import argparse
import json
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
IDX = REPO / "data/index"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import mrr, ndcg_at_k, recall_at_k  # noqa: E402


def local_snapshot(repo_id):
    pat = f"models--{repo_id.replace('/', '--')}/snapshots/*/"
    return str(sorted((Path.home() / ".cache/huggingface/hub").glob(pat))[0])


def page_scores_from_chunks(sim_row, metas):
    """块级得分聚合到页：取最大值。"""
    best = {}
    for j, m in enumerate(metas):
        v = sim_row[j]
        p = m["page_id"]
        if p not in best or v > best[p]:
            best[p] = v
    return best


def rrf(rank_lists, k=60):
    """倒数排名融合。rank_lists: [[page_id 按名次降序], ...]"""
    s = defaultdict(float)
    for lst in rank_lists:
        for i, p in enumerate(lst):
            s[p] += 1.0 / (k + i + 1)
    return s


def ranked_from(score_map):
    return [p for p, _ in sorted(score_map.items(), key=lambda kv: -kv[1])]


def paired_bootstrap(a, b, n=10000, seed=7):
    """配对自助法：返回 (均值差, 95%CI下界, 上界, 双尾 p)。"""
    rng = random.Random(seed)
    d = [x - y for x, y in zip(a, b)]
    obs = sum(d) / len(d)
    boots = []
    N = len(d)
    for _ in range(n):
        s = sum(d[rng.randrange(N)] for _ in range(N)) / N
        boots.append(s)
    boots.sort()
    lo, hi = boots[int(0.025 * n)], boots[int(0.975 * n)]
    # 双尾 p：自助分布中落在 0 另一侧的比例
    cnt = sum(1 for x in boots if (x <= 0 if obs > 0 else x >= 0))
    p = 2.0 * cnt / n
    return obs, lo, hi, min(p, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", nargs="+", default=["image", "text"])
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    qs = []
    for src in args.queries:
        f = REPO / f"data/corpus/queries_{src}.jsonl"
        if f.exists():
            qs += [json.loads(l) for l in open(f)]
    if not qs:
        raise SystemExit("没有查询文件")
    print(f"  查询 {len(qs)} 条（" + "，".join(f"{s}源 {sum(1 for q in qs if q['source']==s)}" for s in args.queries) + "）", flush=True)

    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer(local_snapshot("Qwen/Qwen3-VL-Embedding-2B"),
                              model_kwargs={"dtype": torch.bfloat16}, device="cuda:0")
    qe = emb.encode([q["query"] for q in qs], batch_size=args.batch,
                    prompt="Retrieve the document page that answers the query.",
                    convert_to_tensor=True, normalize_embeddings=True).cpu()

    systems = {}

    for key, fn in [("A", "A_text.pt"), ("A'", "Aprime_text.pt"), ("D", "D_image.pt")]:
        p = IDX / fn
        if not p.exists():
            continue
        d = torch.load(p, weights_only=False)
        sims = (qe.float() @ d["emb"].float().T)
        if key == "D":
            metas = [{"page_id": m["page_id"]} for m in d["meta"]]
        else:
            metas = d["meta"]
        systems[key] = [page_scores_from_chunks(sims[i].tolist(), metas) for i in range(len(qs))]
        print(f"  {key} 就绪", flush=True)

    if (IDX / "B_bm25.pkl").exists():
        with open(IDX / "B_bm25.pkl", "rb") as f:
            d = pickle.load(f)
        out = []
        for q in qs:
            sc = d["bm25"].score_all(q["query"])
            out.append(page_scores_from_chunks(sc, d["meta"]))
        systems["B"] = out
        print("  B 就绪", flush=True)

    if "A" in systems and "B" in systems:
        systems["C"] = [rrf([ranked_from(systems["A"][i]), ranked_from(systems["B"][i])])
                        for i in range(len(qs))]
        print("  C 就绪（A+B RRF 融合）", flush=True)

    if (IDX / "E_colqwen.pt").exists():
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
        d = torch.load(IDX / "E_colqwen.pt", weights_only=False)
        path = local_snapshot("vidore/colqwen2.5-v0.2")
        m = ColQwen2_5.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
        pr = ColQwen2_5_Processor.from_pretrained(path)
        qb = pr.process_queries([q["query"] for q in qs]).to(m.device)
        with torch.no_grad():
            qv = m(**qb).to(torch.float16).cpu()
        sc = pr.score_multi_vector(qv, d["emb"])
        pids = [x["page_id"] for x in d["meta"]]
        systems["E"] = [{pids[j]: sc[i][j].item() for j in range(len(pids))} for i in range(len(qs))]
        print("  E 就绪", flush=True)

    # ---------- 评测 ----------
    per_q = {k: [] for k in systems}
    strat = {k: defaultdict(list) for k in systems}
    for k, maps in systems.items():
        for i, q in enumerate(qs):
            ranked = ranked_from(maps[i])
            gold = q["gold_pages"]
            nd = ndcg_at_k(ranked, gold, 5)
            per_q[k].append(nd)
            strat[k][q["source"]].append((nd, recall_at_k(ranked, gold, 1),
                                          recall_at_k(ranked, gold, 5), mrr(ranked, gold)))

    order = ["A", "A'", "B", "C", "D", "E"]
    names = {"A": "解析式·稠密", "A'": "解析式·按行切块", "B": "解析式·BM25",
             "C": "解析式·混合", "D": "视觉式·单向量", "E": "视觉式·多向量"}
    print("\n  === 总体（全部查询）===")
    print(f"  {'系统':18s} {'nDCG@5':>8s} {'R@1':>7s} {'R@5':>7s} {'MRR':>7s}")
    for k in order:
        if k not in systems:
            continue
        v = [x for lst in strat[k].values() for x in lst]
        n = len(v)
        print(f"  {k+' '+names[k]:18s} {sum(x[0] for x in v)/n:8.3f} {sum(x[1] for x in v)/n:7.3f} "
              f"{sum(x[2] for x in v)/n:7.3f} {sum(x[3] for x in v)/n:7.3f}")

    print("\n  === 按查询来源分组（检验出题偏差）===")
    for src in sorted({q["source"] for q in qs}):
        print(f"  --- {src} 源 ---")
        for k in order:
            if k not in systems:
                continue
            v = strat[k][src]
            if not v:
                continue
            n = len(v)
            print(f"   {k+' '+names[k]:18s} n={n:3d}  nDCG@5={sum(x[0] for x in v)/n:.3f}  R@1={sum(x[1] for x in v)/n:.3f}")

    print("\n  === 配对自助法（10000 次重采样）===")
    pairs = [("D", "A", "表示方式的独立影响（同模型）"),
             ("E", "C", "两条路线各自最强"),
             ("A'", "A", "按行切块 vs 默认切块"),
             ("C", "A", "混合检索相对稠密的增益")]
    for x, y, why in pairs:
        if x not in systems or y not in systems:
            continue
        obs, lo, hi, p = paired_bootstrap(per_q[x], per_q[y])
        sig = "显著" if (lo > 0 or hi < 0) else "不显著（置信区间跨 0）"
        print(f"  {x} − {y}  {why}")
        print(f"    Δ nDCG@5 = {obs:+.4f}   95%CI [{lo:+.4f}, {hi:+.4f}]   p={p:.4f}   → {sig}")

    json.dump({"per_query": per_q, "n": len(qs)},
              open(REPO / "data/eval_results.json", "w"), ensure_ascii=False)
    print("\n  结果已存 data/eval_results.json", flush=True)


if __name__ == "__main__":
    main()
