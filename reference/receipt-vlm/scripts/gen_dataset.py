# -*- coding: utf-8 -*-
"""
生成 train/val/test 三个 split，布局对齐 configs/*.yaml 里的路径：

    data/synthetic/train/train.jsonl  +  data/synthetic/train/images/
    data/synthetic/val/val.jsonl      +  data/synthetic/val/images/
    data/synthetic/test/test.jsonl    +  data/synthetic/test/images/

三个 split 用不同随机种子，互不重叠。

用法: python scripts/gen_dataset.py [--train 3000] [--val 300] [--test 500]
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.synth import DIFFICULTY_PRESETS, generate_synthetic_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=3000)
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--test", type=int, default=500)
    ap.add_argument("--augmentation", default="auto",
                    choices=["none", "light", "medium", "heavy", "auto"],
                    help="auto = 跟随 difficulty 档位")
    ap.add_argument("--difficulty", choices=list(DIFFICULTY_PRESETS), default="hard",
                    help="捷径多少：easy 复刻旧行为作对照，hard 关闭位置/闭集先验")
    ap.add_argument("--output-dir", default=str(REPO / "data" / "synthetic"))
    args = ap.parse_args()

    root = Path(args.output_dir)
    splits = [
        ("train", args.train, 42),
        ("val", args.val, 4242),
        ("test", args.test, 424242),
    ]

    for name, num, seed in splits:
        if num <= 0:
            continue
        out = root / name
        print(f"\n══ split={name}  n={num}  seed={seed} ══")
        dataset = generate_synthetic_dataset(
            output_dir=out,
            num_samples=num,
            augmentation=args.augmentation,
            seed=seed,
            difficulty=args.difficulty,
        )
        jsonl = out / f"{name}.jsonl"
        with open(jsonl, "w", encoding="utf-8") as f:
            for rec in dataset:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  ✓ {jsonl}  ({len(dataset)} 条)")

    print(f"\n完成，根目录: {root}")


if __name__ == "__main__":
    main()
