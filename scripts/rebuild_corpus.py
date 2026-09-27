#!/usr/bin/env python
"""在新机器上按已入库清单精确重建语料（PDF + 页面图），不重新选样。

为什么不直接重跑 fetch_corpus.py + build_pages.py
------------------------------------------------
* fetch_corpus.py 按巨潮「当前」公告列表选公司，列表会变，重跑可能选中另一批 40 家，
  与已入库的 parsed.jsonl.gz / ref_text.jsonl.gz 对不上。
* 这里只读 meta.jsonl（固定 URL）和 pages.jsonl（固定页码），产物与原机器一一对应。

做三件事
--------
1. 按 meta.jsonl 的 url 下载 PDF，校验文件头与页数（清单里的页码必须都在范围内）。
2. 按 pages.jsonl 渲染页面图，参数与 build_pages.py 相同（pdftoppm, 110 DPI），
   并核对每页的文件名与原清单一致（pdftoppm 的补零宽度取决于总页数）。
3. 把 meta.jsonl / pages.jsonl 里的绝对路径改写到本机仓库。幂等，可重复跑。

大文件放在 $ZHDOC_DATA（默认 /projects/$USER/zh-docrag_data），
data/corpus/pdf 与 data/corpus/pages 软链过去。
"""
import getpass, json, os, shutil, subprocess, sys, time, urllib.request
from collections import defaultdict
from pathlib import Path
import pypdf

REPO = Path(__file__).resolve().parent.parent
CORPUS = REPO / "data/corpus"
DATA = Path(os.environ.get("ZHDOC_DATA", f"/projects/{getpass.getuser()}/zh-docrag_data"))
UA = "Mozilla/5.0 (X11; Linux x86_64)"
DPI = 110                                   # 与 build_pages.py 一致


def link_dirs():
    for name in ("pdf", "pages"):
        real, link = DATA / name, CORPUS / name
        real.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or not link.exists():
            if link.is_symlink():
                link.unlink()
            link.symlink_to(real)
        elif link.resolve() != real.resolve():
            sys.exit(f"{link} 已是真实目录，先挪走再跑")


def local(p: str) -> str:
    """原机器绝对路径 → 本机仓库路径，只保留 data/corpus/ 之后的部分。"""
    return str(CORPUS / p.split("/data/corpus/", 1)[1])


def download(meta):
    for i, m in enumerate(meta, 1):
        dst = CORPUS / "pdf" / f"{m['code']}.pdf"
        if not (dst.exists() and dst.stat().st_size > 100_000):
            req = urllib.request.Request(m["url"], headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if not data.startswith(b"%PDF"):
                sys.exit(f"{m['code']} 下载内容不是 PDF：{data[:60]!r}")
            dst.write_bytes(data)
            time.sleep(1.5)                 # 与 fetch_corpus.py 相同的限速
        print(f"   [{i}/{len(meta)}] {m['code']} {m['name'][:10]} {dst.stat().st_size // 1024}KB", flush=True)


def ranges(pages):
    """把页码列表合成连续段，一段调一次 pdftoppm。"""
    out, pages = [], sorted(pages)
    for p in pages:
        if out and p == out[-1][1] + 1:
            out[-1][1] = p
        else:
            out.append([p, p])
    return out


def render(rows, pdftoppm):
    by_code = defaultdict(list)
    for r in rows:
        by_code[r["code"]].append(r)
    t0 = time.time()
    for i, (code, rs) in enumerate(by_code.items(), 1):
        pdf = CORPUS / "pdf" / f"{code}.pdf"
        n = len(pypdf.PdfReader(pdf).pages)
        if max(r["page"] for r in rs) > n:
            sys.exit(f"{code} 只有 {n} 页，清单要第 {max(r['page'] for r in rs)} 页——下载的不是同一份文件")
        want = {r["page"]: CORPUS / "pages" / Path(r["img"]).name for r in rs}
        for a, b in ranges(p for p, f in want.items() if not f.exists()):
            subprocess.run([pdftoppm, "-png", "-r", str(DPI), "-f", str(a), "-l", str(b),
                            str(pdf), str(CORPUS / "pages" / code)], check=True)
        missing = [f.name for f in want.values() if not f.exists()]
        if missing:
            sys.exit(f"{code} 渲染后缺文件（补零宽度不一致？）：{missing[:3]}")
        print(f"   [{i}/{len(by_code)}] {code} {n}页 → {len(rs)} 页图", flush=True)
    print(f"  渲染耗时 {time.time() - t0:.0f}s", flush=True)


def rewrite(path: Path, keys):
    rows = [json.loads(l) for l in path.open(encoding="utf-8")]
    for r in rows:
        for k in keys:
            r[k] = local(r[k])
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def main():
    pdftoppm = os.environ.get("PDFTOPPM") or shutil.which("pdftoppm")
    if not pdftoppm:
        sys.exit("找不到 pdftoppm（poppler），装上或用 PDFTOPPM= 指定")
    link_dirs()
    print(f"  数据目录 {DATA}", flush=True)
    meta = rewrite(CORPUS / "meta.jsonl", ["path"])
    rows = rewrite(CORPUS / "pages.jsonl", ["img", "pdf"])
    print("  === 下载 PDF ===", flush=True)
    download(meta)
    print("  === 渲染页面图 ===", flush=True)
    render(rows, pdftoppm)
    print(f"  完成：{len(meta)} 份 PDF，{len(rows)} 页图，清单路径已改写到 {CORPUS}", flush=True)


if __name__ == "__main__":
    main()
