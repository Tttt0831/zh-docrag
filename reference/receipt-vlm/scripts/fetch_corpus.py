# -*- coding: utf-8 -*-
"""
下载中文预训练语料。

目标：412M 模型的 Chinchilla 最优训练量约 82 亿 token。

来源（都在 HF，这台机器可达；国内源连不上）：
  1. jingyaogong/minimind_dataset  pretrain_t2t.jsonl  —— 原路线 C 用的就是这个
     系列（当时用的是 1.24GB 的 mini 版），这里取 8.28GB 的完整版
  2. opencsg/Fineweb-Edu-Chinese-V2.1  4_5 桶 —— 教育质量分最高的一档，9.1GB

用法: python scripts/fetch_corpus.py [--with-3_4 N]
"""
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

DEST = Path(__file__).resolve().parent.parent / "data" / "corpus"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-3_4", type=int, default=0,
                    help="额外从 Fineweb 3_4 桶取前 N 个分片（每个约 0.12GB）")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)

    print("═══ 1/2  MiniMind pretrain_t2t.jsonl (8.28 GB) ═══")
    p = hf_hub_download(repo_id="jingyaogong/minimind_dataset",
                        filename="pretrain_t2t.jsonl", repo_type="dataset",
                        local_dir=str(DEST / "minimind"))
    print(f"  ✓ {p}")

    print("\n═══ 2/2  Fineweb-Edu-Chinese-V2.1  4_5 桶 (9.1 GB / 1000 分片) ═══")
    d = snapshot_download(repo_id="opencsg/Fineweb-Edu-Chinese-V2.1", repo_type="dataset",
                          allow_patterns="4_5/*.parquet",
                          local_dir=str(DEST / "fineweb"), max_workers=args.workers)
    print(f"  ✓ {d}")

    if args.with_3_4 > 0:
        pats = [f"3_4/{i:06d}.parquet" for i in range(args.with_3_4)]
        print(f"\n═══ 补充  Fineweb 3_4 桶前 {args.with_3_4} 个分片 ═══")
        snapshot_download(repo_id="opencsg/Fineweb-Edu-Chinese-V2.1", repo_type="dataset",
                          allow_patterns=pats, local_dir=str(DEST / "fineweb"),
                          max_workers=args.workers)
        print("  ✓ 完成")

    print(f"\n全部完成 -> {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
