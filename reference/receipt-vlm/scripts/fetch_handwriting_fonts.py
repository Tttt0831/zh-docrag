# -*- coding: utf-8 -*-
"""
下载开源手写字体（全部 OFL 许可，来自 google/fonts）。

真实票据上的收款人/复核/开票人几乎都是手签的，备注也常是手写。系统自带
字体里没有手写体（只有 ukai 楷体勉强算半手写），所以单独取一批。

字体文件较大（中文手写体 5~20MB），不入库，用本脚本按需获取。

用法: python scripts/fetch_handwriting_fonts.py
"""
import sys
import urllib.request
from pathlib import Path

DEST = Path(__file__).resolve().parent.parent / "assets" / "fonts" / "handwriting"
BASE = "https://raw.githubusercontent.com/google/fonts/main/"

FONTS = {
    # 中文手写 / 毛笔
    "MaShanZheng-Regular.ttf":   "ofl/mashanzheng/MaShanZheng-Regular.ttf",
    "ZhiMangXing-Regular.ttf":   "ofl/zhimangxing/ZhiMangXing-Regular.ttf",
    "LiuJianMaoCao-Regular.ttf": "ofl/liujianmaocao/LiuJianMaoCao-Regular.ttf",
    "LongCang-Regular.ttf":      "ofl/longcang/LongCang-Regular.ttf",
    # 英文手写
    "Caveat-Regular.ttf":        "ofl/caveat/Caveat%5Bwght%5D.ttf",
    "Kalam-Regular.ttf":         "ofl/kalam/Kalam-Regular.ttf",
    "IndieFlower-Regular.ttf":   "ofl/indieflower/IndieFlower-Regular.ttf",
}


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    for name, path in FONTS.items():
        out = DEST / name
        if out.exists() and out.stat().st_size > 1000:
            print(f"  跳过（已存在） {name}")
            ok += 1
            continue
        url = BASE + path
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                data = r.read()
            if len(data) < 1000:
                raise ValueError(f"文件过小 {len(data)}B")
            out.write_bytes(data)
            print(f"  ✓ {name}  {len(data)/1e6:.1f}MB")
            ok += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            fail += 1

    print(f"\n完成: {ok} 成功 / {fail} 失败 -> {DEST}")
    print("许可: SIL Open Font License 1.1")
    return 1 if ok == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
