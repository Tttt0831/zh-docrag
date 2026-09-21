# -*- coding: utf-8 -*-
"""
从已保存的 predictions.jsonl 重算指标，不重跑推理。

存在的理由：F1 的计算口径出过一次错（抽错值只记 FN 不记 FP，precision 被
钉死在 1.0），当时因为没保存预测，四个规模的评测全部要重跑。有了这个脚本，
以后改指标口径只是重算。

用法: python scripts/recompute_metrics.py evaluation_results/scale_2000
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.eval import compute_all_metrics, format_metrics_as_markdown


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for d in sys.argv[1:]:
        d = Path(d)
        f = d / "predictions.jsonl"
        if not f.exists():
            print(f"[跳过] {d} 没有 predictions.jsonl（该次评测早于落盘功能）")
            continue
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        m = compute_all_metrics([r["pred"] for r in rows], [r["target"] for r in rows], d.name)
        (d / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
        (d / "metrics.md").write_text(format_metrics_as_markdown(m), encoding="utf-8")
        print(f"{d.name}: n={m['num_samples']} P={m['precision']*100:.2f}% "
              f"R={m['recall']*100:.2f}% F1={m['f1']*100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
