# -*- coding: utf-8 -*-
"""
量化合成数据里的「捷径」—— 模型不读图能拿到多少分。

check_synth_labels.py 回答的是「标注可不可学」，这个脚本回答的是
「标注好不好猜」。两者都通过，数据才算真的在考 OCR。

三类捷径：
  1. 闭集先验：字段取值来自小集合时，常数预测器就能命中一部分
  2. 位置先验：字段永远画在同一个位置时，模型学坐标即可，不必读标签
  3. 字形先验：整个数据集只用一种字体时，模型只需适应这一种字形

用法: python scripts/check_shortcuts.py --num 300 --difficulty hard
      python scripts/check_shortcuts.py --compare        # easy vs hard 对比
"""
import argparse
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import ImageDraw

from src.data.synth import (
    CFG, DIFFICULTY_PRESETS, TEMPLATE_WEIGHTS, _CUR_FONT, _ENGLISH_COMPANIES,
    amount_surface_forms, date_surface_forms, draw_receipt,
    generate_invoice_data, set_difficulty,
)

FIELDS = ["merchant_name", "date", "total_amount", "tax_amount", "tax_id", "invoice_no"]
CATEGORICAL = ["merchant_name", "date", "tax_id", "invoice_no"]

_calls = []          # [(x, y, text)]
_orig = ImageDraw.ImageDraw.text


def _spy(self, xy, text, *a, **kw):
    try:
        x, y = xy[0], xy[1]
    except Exception:
        x, y = -1, -1
    _calls.append((x, y, str(text)))
    return _orig(self, xy, text, *a, **kw)


def surfaces(field, value):
    if field in ("total_amount", "tax_amount"):
        return amount_surface_forms(value)
    if field == "date":
        return date_surface_forms(value)
    return [str(value)]


def collect(num, difficulty, seed=0):
    set_difficulty(difficulty)
    random.seed(seed)
    ImageDraw.ImageDraw.text = _spy

    templates = list(TEMPLATE_WEIGHTS)
    weights = list(TEMPLATE_WEIGHTS.values())
    values = defaultdict(list)                  # field -> [value]
    ypos = defaultdict(lambda: defaultdict(list))  # tmpl -> field -> [y/H]
    fonts, labels = Counter(), Counter()

    for _ in range(num):
        tmpl = random.choices(templates, weights=weights, k=1)[0]
        data = generate_invoice_data(tmpl)
        if tmpl == "receipt_english" and not CFG["open_merchant"]:
            data["merchant_name"] = random.choice(_ENGLISH_COMPANIES)[0]

        _calls.clear()
        img = draw_receipt(data, template=tmpl)
        H = img.size[1]
        fonts[_CUR_FONT.get("cn") or "(固定)"] += 1

        for x, y, t in _calls:
            if ":" in t:
                labels[t.split(":", 1)[0].strip()[:16]] += 1

        for f in FIELDS:
            v = data.get(f)
            if v is None or v == "":
                continue
            values[f].append(v)
            forms = surfaces(f, v)
            for x, y, t in _calls:
                if any(s in t for s in forms):
                    if 0 <= y <= H:
                        ypos[tmpl][f].append(y / H)
                    break

    ImageDraw.ImageDraw.text = _orig
    return values, ypos, fonts, labels


def report(num, difficulty, seed=0, quiet=False):
    values, ypos, fonts, labels = collect(num, difficulty, seed)
    out = {}

    if not quiet:
        print(f"\n{'='*62}\n难度 = {difficulty}   样本 {num}\n{'='*62}")
        print(f"\n[1] 闭集先验 —— 常数预测器（永远猜最高频值）能拿多少分")
        print(f"{'字段':<16}{'唯一值':>8}{'样本':>7}{'唯一率':>9}{'瞎猜命中':>10}")
        print("-" * 52)

    for f in CATEGORICAL:
        vs = values[f]
        if not vs:
            continue
        c = Counter(vs)
        uniq = len(c) / len(vs) * 100
        guess = c.most_common(1)[0][1] / len(vs) * 100
        out[f"guess_{f}"] = guess
        if not quiet:
            flag = "   ← 可猜" if guess > 2.0 else ""
            print(f"{f:<16}{len(c):>8}{len(vs):>7}{uniq:>8.1f}%{guess:>9.2f}%{flag}")

    if not quiet:
        print(f"\n[2] 位置先验 —— 字段归一化 y 坐标的标准差（越小越可被坐标定位）")
        print(f"{'模板':<18}{'字段':<16}{'std':>8}{'样本':>7}")
        print("-" * 50)
    stds = []
    for tmpl in sorted(ypos):
        for f in FIELDS:
            ys = ypos[tmpl][f]
            if len(ys) < 5:
                continue
            sd = statistics.pstdev(ys)
            stds.append(sd)
            if not quiet:
                flag = "   ← 位置固定" if sd < 0.02 else ""
                print(f"{tmpl:<18}{f:<16}{sd:>8.4f}{len(ys):>7}{flag}")
    out["y_std_mean"] = sum(stds) / len(stds) if stds else 0.0

    out["n_fonts"] = len(fonts)
    out["n_labels"] = len(labels)
    if not quiet:
        print(f"\n[3] 字形 / 版式多样性")
        print(f"   中文字体种类: {len(fonts)}")
        print(f"   不同标签文案: {len(labels)}")
        print(f"   位置标准差均值: {out['y_std_mean']:.4f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=300)
    ap.add_argument("--difficulty", choices=list(DIFFICULTY_PRESETS), default="hard")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare", action="store_true", help="对比所有难度档位")
    args = ap.parse_args()

    if not args.compare:
        report(args.num, args.difficulty, args.seed)
        return 0

    rows = {d: report(args.num, d, args.seed, quiet=True) for d in DIFFICULTY_PRESETS}
    keys = ["guess_merchant_name", "guess_date", "guess_tax_id", "guess_invoice_no",
            "y_std_mean", "n_fonts", "n_labels"]
    names = {"guess_merchant_name": "商户名瞎猜命中%", "guess_date": "日期瞎猜命中%",
             "guess_tax_id": "税号瞎猜命中%", "guess_invoice_no": "票号瞎猜命中%",
             "y_std_mean": "位置标准差(越大越好)", "n_fonts": "字体种类",
             "n_labels": "标签文案种类"}
    ds = list(DIFFICULTY_PRESETS)
    print(f"\n{'指标':<24}" + "".join(f"{d:>12}" for d in ds))
    print("-" * (24 + 12 * len(ds)))
    for k in keys:
        print(f"{names[k]:<24}" + "".join(
            f"{rows[d].get(k, 0):>12.4f}" if isinstance(rows[d].get(k), float)
            else f"{rows[d].get(k, 0):>12}" for d in ds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
