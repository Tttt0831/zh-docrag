# -*- coding: utf-8 -*-
"""
分组评测报告：按字段来源、按票种拆开报数。

为什么需要分组：
1. 六个字段的标注权威性不同——date/total_amount/invoice_code/invoice_no 来自
   SCID gt.json（CSIG 2022 竞赛就是拿它排名的），merchant_name/tax_id 是本项目
   从 ocr.json 正则抽取的。混在一起报会拿自造标注冒充权威基准。
2. 票种决定字段：定额发票没有日期栏、火车票没有发票代码，字段缺失是客观事实
   不是标注问题。全六字段齐全的样本只占 0.54%，按整体报 F1 会把「票种差异」
   和「模型能力」混在一起。

用法:
  python scripts/report_grouped.py evaluation_results/xxx --data data/scid/testB.jsonl
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CORE = ("date", "total_amount", "invoice_code", "invoice_no")
DERIVED = ("merchant_name", "tax_id")
ALL = CORE + DERIVED
TYPE_CN = {"quota": "定额发票", "toll": "高速通行费",
           "machine": "机打发票", "train_bus": "火车票/汽车票"}


def prf(pairs, fields):
    """pairs: [(pred, target)]  -> precision / recall / f1 / 计数"""
    tp = fp = fn = 0
    for p, t in pairs:
        for f in fields:
            pv, tv = p.get(f), t.get(f)
            if tv is not None:
                if pv == tv:
                    tp += 1
                elif pv is None:
                    fn += 1
                else:                 # 抽了个错的：既多抽也漏抽
                    fp += 1
                    fn += 1
            elif pv is not None:
                fp += 1               # 真值为 null 却填了值 = 幻觉
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return P, R, F, tp, fp, fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir")
    ap.add_argument("--data", required=True, help="对应的数据集 jsonl（取 receipt_type）")
    args = ap.parse_args()

    rd = Path(args.result_dir)
    rd = rd if rd.is_absolute() else REPO / rd
    preds = [json.loads(l) for l in (rd / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    data = [json.loads(l) for l in (REPO / args.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(preds) > len(data):
        raise SystemExit(f"预测 {len(preds)} 条多于数据集 {len(data)} 条")
    types = [d.get("receipt_type", "unknown") for d in data[:len(preds)]]
    pairs = [(r["pred"], r["target"]) for r in preds]

    print(f"样本 {len(pairs)} 条   来源 {rd.name}\n")

    print("═══ 按字段来源分组 ═══")
    print(f"{'分组':<26}{'Precision':>11}{'Recall':>9}{'F1':>9}")
    for label, fields in (("核心四字段（gt 权威）", CORE),
                          ("派生两字段（本项目抽取）", DERIVED),
                          ("全六字段", ALL)):
        P, R, F, *_ = prf(pairs, fields)
        print(f"  {label:<24}{P*100:>10.2f}%{R*100:>8.2f}%{F*100:>8.2f}%")

    print("\n═══ 核心四字段，按票种分组 ═══")
    by = defaultdict(list)
    for (p, t), ty in zip(pairs, types):
        by[ty].append((p, t))
    print(f"{'票种':<16}{'样本':>7}{'Precision':>11}{'Recall':>9}{'F1':>9}")
    for ty in sorted(by, key=lambda x: -len(by[x])):
        P, R, F, *_ = prf(by[ty], CORE)
        print(f"  {TYPE_CN.get(ty, ty):<14}{len(by[ty]):>7}{P*100:>10.2f}%{R*100:>8.2f}%{F*100:>8.2f}%")

    print("\n═══ 逐字段（只统计真值非 null 的样本）═══")
    print(f"{'字段':<16}{'来源':>8}{'可比样本':>9}{'抽对':>7}{'准确率':>9}{'幻觉':>7}")
    for f in ALL:
        src = "gt" if f in CORE else "derived"
        n = sum(1 for p, t in pairs if t.get(f) is not None)
        c = sum(1 for p, t in pairs if t.get(f) is not None and p.get(f) == t.get(f))
        h = sum(1 for p, t in pairs if t.get(f) is None and p.get(f) is not None)
        acc = c * 100 / n if n else 0.0
        print(f"  {f:<14}{src:>8}{n:>9}{c:>7}{acc:>8.2f}%{h:>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
