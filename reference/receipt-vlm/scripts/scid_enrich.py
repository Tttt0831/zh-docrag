# -*- coding: utf-8 -*-
"""
用 SCID 的 ocr.json 把标注补齐到本项目的 6 字段 schema。

SCID 的 gt.json 只标了 金额/发票号码/发票代码/日期/出口/入口/始发站/到达站/
座位类型/保险费，没有 merchant_name 和 tax_id。但票面上这些信息是有的，
ocr.json 里有全部文字的转写和坐标，可以抽出来。

两个关键难点：

1. 票面上往往有多个公司名，排在最前的是**发票印刷厂**而非开票方
   （实测高频榜前几名：浙江中瑞印业 436 次、四川万汇票证印务 145 次、
   云南省国税印刷厂 110 次）。必须按关键词排除印刷/印务/印业/票证一类，
   否则补出来的 merchant_name 全是印刷厂。

2. 一张票可能有多个公司名，需要定序：先排除印刷厂，再按「出现在印章文字里
   或靠近票面上部」优先——开票方通常在抬头或印章上。

税号用 18 位统一社会信用代码的正则（9x + 6 位行政区划 + 10 位）。

用法:
  python scripts/scid_enrich.py --split testA --max 500 --out data/scid_eval/testA_500_rich.jsonl
"""
import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from scripts.scid_to_ours import parse_amount, parse_date, parse_str, PLACEHOLDER  # noqa

TAXID = re.compile(r'(9[1-5]\d{6}[0-9A-HJ-NPQRTUWXY]{10})')
COMPANY = re.compile(
    r'([一-鿿]{2,18}?(?:有限责任公司|股份有限公司|有限公司|集团|'
    r'管理中心|印刷厂|印务有限公司|印业有限公司|发展有限公司))')
# 这些是印刷/制票单位，不是开票方
PRINTER_KW = ('印刷', '印务', '印业', '票证', '票据印', '磁卡', '印制')


def pick_merchant(texts, bboxes=None):
    """从一张票的全部文本里挑出最可能的开票方。"""
    cands = []
    for i, t in enumerate(texts):
        if not isinstance(t, str):
            continue
        for m in COMPANY.finditer(t):
            name = m.group(1)
            is_printer = any(kw in name for kw in PRINTER_KW) or any(kw in t for kw in PRINTER_KW)
            # 印章文字里出现的公司名，几乎一定是开票方
            in_seal = ('发票专用章' in t) or ('财务专用章' in t)
            y = None
            if bboxes and i < len(bboxes) and bboxes[i]:
                ys = bboxes[i][1::2]
                y = min(ys) if ys else None
            cands.append({"name": name, "printer": is_printer, "seal": in_seal, "y": y})
    if not cands:
        return None, []
    non_printer = [c for c in cands if not c["printer"]]
    pool = non_printer or []          # 全是印刷厂时宁可不标，避免污染标注
    if not pool:
        return None, [c["name"] for c in cands]
    pool.sort(key=lambda c: (not c["seal"], c["y"] if c["y"] is not None else 1e9))
    return pool[0]["name"], [c["name"] for c in cands]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scid", default="data/SCID_new")
    ap.add_argument("--split", default="testA")
    ap.add_argument("--max", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = REPO / args.scid / args.split
    gt = json.loads((base / "gt.json").read_text(encoding="utf-8"))
    ocr_path = base / "ocr.json"
    if ocr_path.stat().st_size == 0:
        raise SystemExit(f"{ocr_path} 是空的——该 split 没有 OCR 标注，无法补字段")
    ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
    img_root = base / "images"

    import random
    keys = sorted(gt)
    random.Random(args.seed).shuffle(keys)

    from src.data.synth import DEFAULT_PROMPT

    out = Path(args.out)
    out = out if out.is_absolute() else REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)

    st = {"n": 0, "merchant": 0, "taxid": 0, "printer_only": 0, "no_ocr": 0}
    with open(out, "w", encoding="utf-8") as f:
        for k in keys:
            if st["n"] >= args.max:
                break
            hits = list(img_root.rglob(k))
            if not hits:
                continue
            v = gt[k]
            oc = ocr.get(k)
            if oc is None:
                st["no_ocr"] += 1
                merchant, taxid, allco = None, None, []
            else:
                ca = oc.get("content_ann", {})
                texts, bbs = ca.get("texts", []), ca.get("bboxes", [])
                merchant, allco = pick_merchant(texts, bbs)
                blob = " ".join(t for t in texts if isinstance(t, str))
                tm = TAXID.search(blob)
                taxid = tm.group(1) if tm else None
                if allco and merchant is None:
                    st["printer_only"] += 1
            if merchant:
                st["merchant"] += 1
            if taxid:
                st["taxid"] += 1

            f.write(json.dumps({
                "image_path": str(hits[0]),
                "prompt": DEFAULT_PROMPT,
                "target_json": {
                    "merchant_name": merchant,
                    "date": parse_date(v.get("日期")),
                    "total_amount": parse_amount(v.get("金额")),
                    "tax_amount": None,          # SCID 票种无税额
                    "tax_id": taxid,
                    "invoice_no": parse_str(v.get("发票号码")),
                },
                "scid_raw": v,
                "ocr_companies": allco,
                "template": "scid_" + args.split,
            }, ensure_ascii=False) + "\n")
            st["n"] += 1

    print(f"写出 {st['n']} 条 -> {out}")
    print(f"  补出 merchant_name: {st['merchant']} ({st['merchant']*100/max(st['n'],1):.1f}%)")
    print(f"  补出 tax_id:        {st['taxid']} ({st['taxid']*100/max(st['n'],1):.1f}%)")
    print(f"  只有印刷厂、已跳过:  {st['printer_only']}")
    print(f"  无 OCR 记录:        {st['no_ocr']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
