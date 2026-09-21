# -*- coding: utf-8 -*-
"""
SCID 标注定稿：把免费能拿到的补上，把不可信的去掉。

两件事：

1. 补回过路费发票的开票日期。
   SCID 的 gt.json 对 toll 票种**系统性地不标 date**（10081 条里一条都没有），
   但票面上是印了的——OCR 里 80% 能找到形如 2018-03-30 的独立文本框。
   这不是「票上没有」，是 gt 漏标，null 会把模型训成「过路费票没有日期」。

   只采纳**候选唯一**的（6227/10081，歧义的只有 26 条，直接跳过）。
   抽取器用 gt 已有日期的 12901 条样本验过：精确率 98.13%、召回 93.93%。
   不对 machine/train_bus 用——那两类 gt 覆盖已经 91~98%，而且票面有
   售票日期和乘车日期两个，正是上面 227 条抽错的来源。
   定额发票不补：它找到的「日期」是有效期（此发票限在X年X月X日前开具有效）
   和印刷批次年月，都不是开票日期。

2. 剔除 DeepSeek 补出但在 OCR 里找不到依据的 merchant_name。
   模型会从残缺证据里**推断**全称——票面只印「山东高速」，它补成
   「山东高速集团有限公司」。名字大概率对，但那是世界知识而非票面信息，
   拿它当标注等于教模型编造。判据：与 OCR 原文的最佳子串相似度 < 0.7。

用法:
  python scripts/scid_finalize.py --split testB --in data/scid/testB_ds.jsonl \
                                  --out data/scid/testB_final.jsonl
"""
import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# 这些词出现在框里说明不是开票日期（有效期、印刷批次…）
DATE_NOISE = ('限在', '有效', '印', '承', '号', '批', '本', '@', '企业')
DATE_EXACT = re.compile(r'^(20[0-2]\d)[-/年\.](\d{1,2})[-/月\.](\d{1,2})日?$')
DATE_LOOSE = re.compile(r'(20[0-2]\d)[-/年\.](\d{1,2})[-/月\.](\d{1,2})')
GROUND_MIN = 0.7


def date_candidates(texts):
    out = []
    for t in texts:
        s = t.replace(" ", "")
        if any(k in s for k in DATE_NOISE):
            continue
        m = DATE_EXACT.match(s) or (DATE_LOOSE.search(s) if len(s) <= 22 else None)
        if not m:
            continue
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            out.append(f"{y:04d}-{mo:02d}-{d:02d}")
    return sorted(set(out))


def grounded(name, texts):
    """merchant 是否有 OCR 依据。返回 (是否保留, 相似度)。"""
    blob = "".join(texts).replace(" ", "")
    if name in blob:
        return True, 1.0
    if not blob or len(name) > len(blob):
        return False, 0.0
    best = max((SequenceMatcher(None, name, blob[i:i + len(name)]).ratio()
                for i in range(len(blob) - len(name) + 1)), default=0.0)
    return best >= GROUND_MIN, best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = {"train": "Train"}.get(args.split, args.split)
    ocr = json.loads((REPO / f"data/SCID_new/{d}/ocr.json").read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in Path(args.inp).read_text(encoding="utf-8").splitlines()
            if l.strip()]

    st = {"date_add": 0, "date_ambig": 0, "mer_drop": 0, "mer_keep": 0}
    for r in rows:
        tx = [t for t in ocr.get(Path(r["image_path"]).name, {})
              .get("content_ann", {}).get("texts", []) if isinstance(t, str)]
        tgt = r["target_json"]
        src = r.setdefault("anno_source", {})

        if r.get("receipt_type") == "toll" and not tgt.get("date"):
            c = date_candidates(tx)
            if len(c) == 1:
                tgt["date"] = c[0]
                src["date"] = "ocr_regex"
                st["date_add"] += 1
            elif len(c) > 1:
                st["date_ambig"] += 1

        if src.get("merchant_name") == "deepseek" and tgt.get("merchant_name"):
            ok, sim = grounded(tgt["merchant_name"], tx)
            if ok:
                st["mer_keep"] += 1
            else:
                tgt["merchant_name"] = None
                src["merchant_name"] = f"deepseek_rejected_{sim:.2f}"
                st["mer_drop"] += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    F = ["merchant_name", "date", "total_amount", "invoice_code", "invoice_no", "tax_id"]
    n = len(rows)
    print(f"{args.split}: {n} 条 -> {out}")
    print(f"  补回过路费日期 {st['date_add']}（歧义跳过 {st['date_ambig']}）")
    print(f"  DeepSeek merchant 保留 {st['mer_keep']}  剔除无依据 {st['mer_drop']}")
    print("  字段覆盖: " + "  ".join(
        f"{k[:9]}={sum(1 for r in rows if r['target_json'].get(k))/n*100:.1f}%" for k in F))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
