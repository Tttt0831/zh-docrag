# -*- coding: utf-8 -*-
"""
校验落盘后的数据集产物（scripts/check_synth_labels.py 校验的是生成逻辑，
这个脚本校验真正写到磁盘上的 jsonl + 图片）。

检查项：
  1. 图片存在、可打开、尺寸合理
  2. target_json 字段齐全、类型正确
  3. tax_id 唯一率 —— 若税号取自固定公司池，模型可以靠商户名背出税号，
     该字段就不再需要 OCR，指标会虚高
  4. split 之间标注无重复（数据泄漏）
  5. 金额字段数值合理

用法: python scripts/check_dataset.py [--root data/synthetic]
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

FIELDS = ["merchant_name", "date", "total_amount", "tax_amount",
          "tax_id", "invoice_no"]


def load(split_dir: Path, name: str):
    f = split_dir / f"{name}.jsonl"
    if not f.exists():
        return None
    return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/synthetic")
    ap.add_argument("--check-images", type=int, default=200,
                    help="逐张打开校验的图片数上限（0=全部）")
    args = ap.parse_args()

    root = Path(args.root)
    problems = []
    fingerprints = defaultdict(set)

    for name in ("train", "val", "test"):
        recs = load(root / name, name)
        if recs is None:
            print(f"[{name}] 缺失，跳过")
            continue

        print(f"\n═══ split={name}  n={len(recs)} ═══")
        print("模板分布: " + "  ".join(
            f"{k}={v}" for k, v in Counter(r["template"] for r in recs).most_common()))

        # ── 字段完整性与空值率 ──
        null_cnt = Counter()
        for r in recs:
            tj = r["target_json"]
            missing = [f for f in FIELDS if f not in tj]
            if missing:
                problems.append(f"[{name}] 样本缺字段 {missing}: {r['image_path']}")
            for f in FIELDS:
                if tj.get(f) is None:
                    null_cnt[f] += 1
        print("null 率: " + "  ".join(
            f"{f}={null_cnt[f]/len(recs)*100:.0f}%" for f in FIELDS))

        # ── 金额合理性 ──
        for r in recs:
            tj = r["target_json"]
            t, x = tj.get("total_amount"), tj.get("tax_amount")
            if not isinstance(t, (int, float)) or t <= 0:
                problems.append(f"[{name}] total_amount 异常 {t!r}: {r['image_path']}")
            if x is not None and (not isinstance(x, (int, float)) or x < 0 or x >= t):
                problems.append(f"[{name}] tax_amount 异常 {x!r} (total={t}): {r['image_path']}")

        # ── tax_id 是否可被商户名记忆 ──
        pairs = [(r["target_json"]["merchant_name"], r["target_json"]["tax_id"])
                 for r in recs if r["target_json"].get("tax_id")]
        if pairs:
            ids = [t for _, t in pairs]
            uniq = len(set(ids)) / len(ids) * 100
            by_merchant = defaultdict(set)
            for mn, tid in pairs:
                by_merchant[mn].add(tid)
            # 只统计出现过多次的商户：商户名是开集时几乎每个只出现一次，
            # 「该商户只有一个税号」是必然的，拿它当记忆风险指标会误导。
            repeated = {m: v for m, v in by_merchant.items()
                        if sum(1 for x, _ in pairs if x == m) >= 2}
            memorizable = sum(1 for v in repeated.values() if len(v) == 1)
            print(f"tax_id 唯一率: {uniq:.1f}%  （{len(set(ids))}/{len(ids)}）")
            if repeated:
                print(f"重复出现的商户: {len(repeated)}，其中税号唯一（可被背下来）: "
                      f"{memorizable}")
            else:
                print("商户名为开集（无重复），不存在商户名→税号的记忆捷径")
            if uniq < 95:
                problems.append(
                    f"[{name}] tax_id 唯一率仅 {uniq:.1f}%，模型可能靠商户名背税号")

        # ── 图片可读性 ──
        n_check = len(recs) if args.check_images == 0 else min(args.check_images, len(recs))
        step = max(1, len(recs) // n_check)
        bad_img = 0
        for r in recs[::step]:
            p = Path(r["image_path"])
            if not p.exists():
                bad_img += 1
                problems.append(f"[{name}] 图片缺失: {p}")
                continue
            try:
                with Image.open(p) as im:
                    im.load()
                    if min(im.size) < 100:
                        problems.append(f"[{name}] 图片过小 {im.size}: {p}")
            except Exception as e:
                bad_img += 1
                problems.append(f"[{name}] 图片打不开 {p}: {e}")
        print(f"图片抽检 {len(recs[::step])} 张，异常 {bad_img} 张")

        for r in recs:
            fingerprints[name].add(json.dumps(r["target_json"], sort_keys=True,
                                              ensure_ascii=False))

    # ── split 间泄漏 ──
    print("\n═══ split 间标注重叠 ═══")
    names = [n for n in ("train", "val", "test") if n in fingerprints]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = fingerprints[a] & fingerprints[b]
            print(f"  {a} ∩ {b}: {len(overlap)}")
            if overlap:
                problems.append(f"{a} 与 {b} 有 {len(overlap)} 条标注完全相同（数据泄漏）")

    print()
    if problems:
        print(f"❌ 发现 {len(problems)} 个问题:")
        for p in problems[:20]:
            print("  " + p)
        if len(problems) > 20:
            print(f"  ... 另有 {len(problems) - 20} 个")
        return 1
    print("✅ 数据集校验通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
