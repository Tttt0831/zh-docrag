# -*- coding: utf-8 -*-
"""
核查合成数据的标注是否真的画到了图上。

做法：monkeypatch PIL 的 ImageDraw.text，记录每张图实际绘制的全部文字，
再逐字段检查标注值能否在这些文字里找到。任何字段的可学比例 < 100%，
说明该字段存在「标注存在但图上没有」的样本 —— 在这种数据上训练，模型
只能靠先验猜，指标会在数据上界处触顶，与算法无关。

用法: python scripts/check_synth_labels.py [--num 400]
"""
import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import ImageDraw

from src.data.synth import (
    CFG, DIFFICULTY_PRESETS, TEMPLATE_WEIGHTS, _ENGLISH_COMPANIES,
    amount_surface_forms, date_surface_forms, draw_receipt,
    generate_invoice_data, set_difficulty,
)

FIELDS = ["merchant_name", "date", "total_amount", "tax_amount",
          "tax_id", "invoice_no"]
AMOUNT_FIELDS = {"total_amount", "tax_amount"}

_drawn: list = []
_orig_text = ImageDraw.ImageDraw.text


def _spy_text(self, xy, text, *a, **kw):
    _drawn.append(str(text))
    return _orig_text(self, xy, text, *a, **kw)


def field_is_drawn(field: str, value, blob: str) -> bool:
    """
    标注值是否出现在实际绘制的文字里。

    注意标注是「归一化形式」而图上是「表面形式」：date 标注恒为 ISO，图上
    可能写成 2024年1月5日；金额标注是 float，图上可能带千分位或货币符号。
    所以要按字段各自的表面形式集合去匹配，不能直接 str(value) in blob。
    """
    if field in AMOUNT_FIELDS:
        return any(f in blob for f in amount_surface_forms(value))
    if field == "date":
        return any(f in blob for f in date_surface_forms(value))
    return str(value) in blob


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--difficulty", choices=list(DIFFICULTY_PRESETS), default="medium")
    args = ap.parse_args()

    set_difficulty(args.difficulty)
    random.seed(args.seed)
    ImageDraw.ImageDraw.text = _spy_text

    templates = list(TEMPLATE_WEIGHTS)
    weights = list(TEMPLATE_WEIGHTS.values())

    # present[f] = 标注非空的样本数; drawn[f] = 其中真的画上去了的数量
    present = defaultdict(int)
    drawn = defaultdict(int)
    per_tmpl = defaultdict(lambda: (defaultdict(int), defaultdict(int)))
    tmpl_count = defaultdict(int)
    misses = []

    for i in range(args.num):
        tmpl = random.choices(templates, weights=weights, k=1)[0]
        data = generate_invoice_data(tmpl)
        if tmpl == "receipt_english" and not CFG["open_merchant"]:
            data["merchant_name"] = random.choice(_ENGLISH_COMPANIES)[0]

        _drawn.clear()
        draw_receipt(data, template=tmpl)   # 绘制函数会把实际值写回 data
        blob = "\n".join(_drawn)

        tmpl_count[tmpl] += 1
        p_t, d_t = per_tmpl[tmpl]
        for f in FIELDS:
            v = data.get(f)
            if v is None or v == "":
                continue
            present[f] += 1
            p_t[f] += 1
            if field_is_drawn(f, v, blob):
                drawn[f] += 1
                d_t[f] += 1
            elif len(misses) < 10:
                misses.append((i, tmpl, f, repr(v)))

    ImageDraw.ImageDraw.text = _orig_text

    print(f"难度 {args.difficulty}")
    print(f"样本数 {args.num}  模板分布 " +
          "  ".join(f"{t}={tmpl_count[t]}" for t in templates))
    print()
    print(f"{'字段':<16}{'标注非空':>9}{'画上图':>9}{'可学比例':>11}")
    print("-" * 46)
    bad = []
    for f in FIELDS:
        p, d = present[f], drawn[f]
        ratio = d / p * 100 if p else 100.0
        flag = "" if ratio >= 99.999 else "   ← 有缺口"
        print(f"{f:<16}{p:>9}{d:>9}{ratio:>10.1f}%{flag}")
        if ratio < 99.999:
            bad.append(f)

    print()
    print("按模板拆分（仅显示不足 100% 的）:")
    any_tmpl_bad = False
    for t in templates:
        p_t, d_t = per_tmpl[t]
        for f in FIELDS:
            if p_t[f] and d_t[f] < p_t[f]:
                any_tmpl_bad = True
                print(f"  {t:<18}{f:<16}{d_t[f]}/{p_t[f]}"
                      f"  ({d_t[f]/p_t[f]*100:.1f}%)")
    if not any_tmpl_bad:
        print("  （无）")

    if misses:
        print("\n未落图样例:")
        for i, t, f, v in misses:
            print(f"  #{i} {t} {f} = {v}")

    print()
    if bad:
        print(f"❌ 以下字段标注不可学: {', '.join(bad)}")
        return 1
    print("✅ 全部字段 100% 可学")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
