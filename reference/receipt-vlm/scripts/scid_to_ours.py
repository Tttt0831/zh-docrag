# -*- coding: utf-8 -*-
"""
把 SCID 的标注转成本项目的 schema，用于「合成数据训练 → 真实票据测试」的泛化评测。

两套标注的哲学不同，必须做口径转换才能比较：
    我们:  total_amount = 1234.56 (float)      date = "2024-01-05" (ISO)
    SCID:  金额 = "壹拾元" (票面原文)            日期 = "2021年03月17日" (票面原文)
SCID 的金额有 63.3% 是中文大写、日期有 20.4% 是「年月日」写法，直接比较会
把「标注口径不同」误判成「模型读错了」。

只转换三个重叠字段：金额->total_amount、日期->date、发票号码->invoice_no。
merchant_name / tax_id / tax_amount 在 SCID 里不存在，置为 None 并在评测时排除。

用法:
  python scripts/scid_to_ours.py --split testA --max 500 --out data/scid_eval/testA.jsonl
"""
import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

_CN_DIGIT = {'零':0,'〇':0,'一':1,'壹':1,'二':2,'贰':2,'两':2,'三':3,'叁':3,'四':4,'肆':4,
             '五':5,'伍':5,'六':6,'陆':6,'七':7,'柒':7,'八':8,'捌':8,'九':9,'玖':9}
_CN_UNIT = {'十':10,'拾':10,'百':100,'佰':100,'千':1000,'仟':1000}
_CN_BIG = {'万':10**4,'亿':10**8}
PLACEHOLDER = {'XXX', '***', '#', '', 'xxx', 'X', '#####'}


def cn_int(s: str):
    """解析中文整数部分，如 壹佰贰拾叁 -> 123。"""
    if not s:
        return 0
    total, section, num = 0, 0, 0
    for ch in s:
        if ch in _CN_DIGIT:
            num = _CN_DIGIT[ch]
        elif ch in _CN_UNIT:
            section += (num or 1) * _CN_UNIT[ch]
            num = 0
        elif ch in _CN_BIG:
            section = (section + num) * _CN_BIG[ch]
            total += section
            section = num = 0
        else:
            return None
    return total + section + num


def parse_amount(s):
    """SCID 的金额 -> float。覆盖中文大写、纯数字、带货币符号、占位符。"""
    if s is None:
        return None
    s = str(s).strip()
    if s in PLACEHOLDER:
        return None
    # 纯数字/带符号：去掉货币符号与千分位
    t = re.sub(r'[¥￥$元圆\s,，]', '', s)
    if re.fullmatch(r'\d+(\.\d+)?', t):
        return round(float(t), 2)
    # 中文大写：整数部分 + 角 + 分
    m = re.match(r'^(.*?)[元圆](.*)$', s)
    head, tail = (m.group(1), m.group(2)) if m else (s, '')
    yuan = cn_int(head)
    if yuan is None:
        return None
    jiao = fen = 0
    mj = re.search(r'([零〇一壹二贰两三叁四肆五伍六陆七柒八捌九玖])角', tail)
    mf = re.search(r'([零〇一壹二贰两三叁四肆五伍六陆七柒八捌九玖])分', tail)
    if mj:
        jiao = _CN_DIGIT[mj.group(1)]
    if mf:
        fen = _CN_DIGIT[mf.group(1)]
    return round(yuan + jiao / 10 + fen / 100, 2)


def parse_date(s):
    """SCID 的日期 -> ISO。"""
    if s is None:
        return None
    s = str(s).strip()
    if s in PLACEHOLDER:
        return None
    for pat in (r'^(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})日?$',):
        m = re.fullmatch(pat, s)
        if m:
            y, mo, d = (int(x) for x in m.groups())
            if 1900 <= y <= 2100 and 1 <= mo <= 12 and 1 <= d <= 31:
                return f'{y:04d}-{mo:02d}-{d:02d}'
    if re.fullmatch(r'\d{8}', s):
        y, mo, d = int(s[:4]), int(s[4:6]), int(s[6:])
        if 1900 <= y <= 2100 and 1 <= mo <= 12 and 1 <= d <= 31:
            return f'{y:04d}-{mo:02d}-{d:02d}'
    return None


def parse_str(s):
    if s is None:
        return None
    s = str(s).strip()
    return None if s in PLACEHOLDER else s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scid", default="data/SCID_new")
    ap.add_argument("--split", default="testA")
    ap.add_argument("--max", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = REPO / args.scid / args.split
    gt = json.loads((base / "gt.json").read_text(encoding="utf-8"))
    img_root = base / "images"

    import random
    keys = sorted(gt)
    random.Random(args.seed).shuffle(keys)

    from src.data.synth import DEFAULT_PROMPT  # noqa

    out = Path(args.out or f"data/scid_eval/{args.split}.jsonl")
    out = out if out.is_absolute() else REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "no_image": 0, "amount_fail": 0, "date_fail": 0}
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for k in keys:
            if n >= args.max:
                break
            hits = list(img_root.rglob(k))
            if not hits:
                stats["no_image"] += 1
                continue
            v = gt[k]
            stats["total"] += 1
            amt_raw, date_raw = v.get("金额"), v.get("日期")
            amt, dt = parse_amount(amt_raw), parse_date(date_raw)
            if amt_raw not in (None,) and str(amt_raw).strip() not in PLACEHOLDER and amt is None:
                stats["amount_fail"] += 1
            if date_raw is not None and str(date_raw).strip() not in PLACEHOLDER and dt is None:
                stats["date_fail"] += 1
            rec = {
                "image_path": str(hits[0]),
                "prompt": DEFAULT_PROMPT,
                "target_json": {
                    "merchant_name": None,          # SCID 无此字段
                    "date": dt,
                    "total_amount": amt,
                    "tax_amount": None,             # SCID 无此字段
                    "tax_id": None,                 # SCID 无此字段
                    "invoice_no": parse_str(v.get("发票号码")),
                },
                "scid_raw": v,
                "template": "scid_" + args.split,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    print(f"写出 {n} 条 -> {out}")
    print(f"  图片缺失跳过: {stats['no_image']}")
    print(f"  金额解析失败: {stats['amount_fail']}  日期解析失败: {stats['date_fail']}")
    return 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(REPO))
    raise SystemExit(main())
