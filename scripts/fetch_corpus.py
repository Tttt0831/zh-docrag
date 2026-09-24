#!/usr/bin/env python
"""从巨潮抓 2024 年年报，跨四个板块各 10 家，共 40 份。

设计要点
--------
* 跨板块取样：沪主板(600/601/603)、深主板(000/001)、创业板(300)、科创板(688)。
  巨潮没有行业字段，板块是能拿到的最接近「不同版式传统」的代理变量。
* 去重：同一家公司会有「年度报告」「摘要」「更正后」多条，只保留一条正文年报。
* 限速：巨潮是证监会指定披露平台，年报是公开文件，但仍按 1.5s/请求 限速，
  量也控制在项目所需（40 份）。
* 记录完整元数据，后续页码/真值都要能追溯回原始文件。
"""
import json
import random
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_PDF = REPO / "data/corpus/pdf"
META = REPO / "data/corpus/meta.jsonl"
UA = "Mozilla/5.0 (X11; Linux x86_64)"

BOARDS = {
    "沪主板": (("600", "601", "603"), "sse"),
    "深主板": (("000", "001"), "szse"),
    "创业板": (("300",), "szse"),
    "科创板": (("688",), "sse"),
}
PER_BOARD = 10

# 排除：摘要、英文版、已取消、更正说明本身
BAD = re.compile(r"摘要|英文|English|取消|更正说明|补充公告|问询函")


def query(column, page=1, size=30):
    d = urllib.parse.urlencode({
        "pageNum": page, "pageSize": size, "column": column, "tabName": "fulltext",
        "category": "category_ndbg_szsh", "seDate": "2024-01-01~2024-12-31", "isHLtitle": "true",
    }).encode()
    req = urllib.request.Request(
        "http://www.cninfo.com.cn/new/hisAnnouncement/query", data=d,
        headers={"User-Agent": UA, "Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.load(r)


def collect():
    picked, seen_code = {b: [] for b in BOARDS}, set()
    for board, (prefixes, column) in BOARDS.items():
        page = 1
        while len(picked[board]) < PER_BOARD and page <= 25:
            try:
                d = query(column, page)
            except Exception as e:
                print(f"   查询失败 {board} p{page}: {type(e).__name__}", flush=True)
                page += 1
                time.sleep(2)
                continue
            for a in d.get("announcements") or []:
                code = a.get("secCode", "")
                title = a.get("announcementTitle", "")
                if not code.startswith(prefixes) or code in seen_code:
                    continue
                if BAD.search(title) or "年度报告" not in title:
                    continue
                if (a.get("adjunctSize") or 0) < 1500:      # 太小的多半是摘要/说明
                    continue
                seen_code.add(code)
                picked[board].append({
                    "board": board, "code": code, "name": a.get("secName", ""),
                    "title": title, "url": "http://static.cninfo.com.cn/" + a["adjunctUrl"],
                    "size_kb": a.get("adjunctSize"),
                })
                if len(picked[board]) >= PER_BOARD:
                    break
            page += 1
            time.sleep(1.5)
        print(f"   {board}: {len(picked[board])} 家", flush=True)
    return [x for v in picked.values() for x in v]


def download(items):
    OUT_PDF.mkdir(parents=True, exist_ok=True)
    ok = []
    for i, it in enumerate(items, 1):
        dst = OUT_PDF / f"{it['code']}.pdf"
        if dst.exists() and dst.stat().st_size > 100_000:
            it["path"] = str(dst)
            ok.append(it)
            continue
        try:
            req = urllib.request.Request(it["url"], headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as f:
                f.write(r.read())
            it["path"] = str(dst)
            ok.append(it)
            print(f"   [{i}/{len(items)}] {it['code']} {it['name'][:10]} {dst.stat().st_size//1024}KB", flush=True)
        except Exception as e:
            print(f"   [{i}/{len(items)}] {it['code']} 失败 {type(e).__name__}", flush=True)
        time.sleep(1.5)
    return ok


def main():
    random.seed(20260923)
    print("  === 检索公告列表 ===", flush=True)
    items = collect()
    print(f"  共选中 {len(items)} 家", flush=True)
    print("  === 下载 PDF ===", flush=True)
    ok = download(items)
    META.parent.mkdir(parents=True, exist_ok=True)
    with META.open("w", encoding="utf-8") as f:
        for it in ok:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"  成功 {len(ok)}/{len(items)} 份 → {META}", flush=True)


if __name__ == "__main__":
    main()
