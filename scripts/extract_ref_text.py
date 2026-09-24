#!/usr/bin/env python
"""抽取每页的 PDF 内嵌文本，作为**独立于两条路线**的接地校验参照。

为什么不能用 DeepDoc 的解析文本做参照：
若某页答案在图上清晰可见、但被 DeepDoc 解析坏了，用它做参照会把这条查询
判为「未接地」而丢弃——恰好剔掉解析式最难的样本，等于给解析式放水。
pdfplumber 直接读 PDF 内嵌文字层，和 DeepDoc、和视觉编码都无关。

已知局限：部分年报的内嵌文字层是乱码（扫描件或字体子集化），这类页记为
garbled，不用于出题。数量会被统计出来。
"""
import json, re, sys, time, warnings
from pathlib import Path
import pdfplumber
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data/corpus/ref_text.jsonl"

CJK = re.compile(r"[一-鿿]")

def garbled(t: str) -> bool:
    """中文年报页若几乎没有 CJK 字符，多半是字体子集化导致的乱码。"""
    if len(t.strip()) < 30:
        return True
    return len(CJK.findall(t)) / max(len(t), 1) < 0.05

def main():
    pages = [json.loads(l) for l in open(REPO / "data/corpus/pages.jsonl")]
    by_pdf = {}
    for p in pages:
        by_pdf.setdefault(p["pdf"], []).append(p)
    done = set()
    if OUT.exists():
        for l in OUT.open(encoding="utf-8"):
            try: done.add(json.loads(l)["page_id"])
            except Exception: pass
    n = ok = bad = 0
    t0 = time.time()
    with OUT.open("a", encoding="utf-8") as f:
        for pdf, plist in by_pdf.items():
            todo = [p for p in plist if p["page_id"] not in done]
            if not todo: continue
            with pdfplumber.open(pdf) as doc:
                for p in todo:
                    try:
                        t = doc.pages[p["page"] - 1].extract_text() or ""
                    except Exception:
                        t = ""
                    g = garbled(t)
                    f.write(json.dumps({"page_id": p["page_id"], "text": t, "garbled": g}, ensure_ascii=False) + "\n")
                    n += 1; ok += (not g); bad += g
            f.flush()
            print(f"   {Path(pdf).stem}  累计 {n} 页（可用 {ok} / 乱码 {bad}）  {time.time()-t0:.0f}s", flush=True)
    print(f"  完成 {n} 页，可用 {ok}，乱码 {bad}，耗时 {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
