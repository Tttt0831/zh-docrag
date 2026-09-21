# -*- coding: utf-8 -*-
"""
从 train.jsonl 切出嵌套子集，用于量「合成数据的边际收益曲线」。

为什么要分层截断而不是直接取前 N 条：
生成时模板是按权重随机抽的，前 N 条的模板比例只是近似一致。若子集之间
模板分布有漂移，规模曲线上的差异就分不清是「数据变多了」还是「模板配比
变了」——那正是我们要避免的测量假象。所以按模板分层、等比例抽取。

嵌套性：小子集是大子集的真子集，这样曲线上相邻两点的差异只来自「新增的
那部分数据」，不来自样本重新洗牌。

用法: python scripts/make_subsets.py --sizes 1000 2000 4000
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(REPO / "data/synthetic/train/train.jsonl"))
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 2000, 4000])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src = Path(args.train)
    recs = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    total = len(recs)
    print(f"源: {src}  共 {total} 条")

    by_t = defaultdict(list)
    for r in recs:
        by_t[r["template"]].append(r)
    rng = random.Random(args.seed)
    for t in by_t:
        rng.shuffle(by_t[t])
    print("模板分布: " + "  ".join(f"{t}={len(v)}" for t, v in sorted(by_t.items())))

    for n in sorted(args.sizes):
        if n >= total:
            print(f"\n跳过 {n}（不小于全量 {total}）")
            continue
        # 按模板比例分配名额，余数给数量最多的模板
        picked = []
        alloc = {t: int(round(len(v) / total * n)) for t, v in by_t.items()}
        drift = n - sum(alloc.values())
        biggest = max(by_t, key=lambda t: len(by_t[t]))
        alloc[biggest] += drift
        for t, k in alloc.items():
            picked.extend(by_t[t][:max(0, k)])      # 前 k 条 → 保证嵌套
        rng_out = random.Random(args.seed)
        rng_out.shuffle(picked)

        out = src.parent / f"train_{n}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for r in picked:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        dist = defaultdict(int)
        for r in picked:
            dist[r["template"]] += 1
        print(f"\n{out.name}: {len(picked)} 条")
        print("  " + "  ".join(f"{t}={dist[t]}({dist[t]/len(picked)*100:.1f}%)"
                               for t in sorted(dist)))

    # 嵌套性自检
    print("\n嵌套性自检:")
    sizes = sorted(s for s in args.sizes if s < total)
    prev_ids = None
    for n in sizes:
        f = src.parent / f"train_{n}.jsonl"
        ids = {json.loads(l)["image_path"] for l in f.read_text(encoding="utf-8").splitlines() if l.strip()}
        if prev_ids is not None:
            ok = prev_ids <= ids
            print(f"  train_{prev_n} ⊆ train_{n}: {'✓' if ok else '✗ 不是子集'}")
            if not ok:
                return 1
        prev_ids, prev_n = ids, n
    full_ids = {r["image_path"] for r in recs}
    if prev_ids is not None:
        print(f"  train_{prev_n} ⊆ train(全量): {'✓' if prev_ids <= full_ids else '✗'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
