#!/usr/bin/env python
"""解析全语料 2400 页，每块带页面归属。

* 逐页解析：页码归属必须确定，否则页面级检索的真值无从对齐（MVP 时吃过这个亏）。
* 断点续跑：已解析的页跳过，中断后重跑不会重来。
* 剥掉 DeepDoc 内联的位置标记 @@页\tx0\tx1\ttop\tbottom##，否则坐标数字会进向量。
* 同时保存表格 HTML 原文，供后续「按行切块」消融组 A′ 使用。
"""
import json, re, sys, time, warnings
from pathlib import Path
import pypdf
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import deepdoc_standalone  # noqa: E402,F401
from deepdoc.parser.pdf_parser import RAGFlowPdfParser  # noqa: E402

OUT = REPO / "data/corpus/parsed.jsonl"
POS = re.compile(r"@@\d+\t[\d.]+\t[\d.]+\t[\d.]+\t[\d.]*#*")


def clean(t):
    return POS.sub(" ", t).replace("##", " ").strip()


def main():
    pages = [json.loads(l) for l in open(REPO / "data/corpus/pages.jsonl")]
    done = set()
    if OUT.exists():
        for l in OUT.open(encoding="utf-8"):
            try: done.add(json.loads(l)["page_id"])
            except Exception: pass
    todo = [p for p in pages if p["page_id"] not in done]
    print(f"  总 {len(pages)} 页，已完成 {len(done)}，待解析 {len(todo)}", flush=True)

    parser = RAGFlowPdfParser()
    tmp = REPO / "data/corpus/_one.pdf"
    readers = {}
    t0 = time.time()
    with OUT.open("a", encoding="utf-8") as f:
        for i, pg in enumerate(todo, 1):
            if pg["pdf"] not in readers:
                readers[pg["pdf"]] = pypdf.PdfReader(pg["pdf"])
            w = pypdf.PdfWriter(); w.add_page(readers[pg["pdf"]].pages[pg["page"] - 1])
            w.write(open(tmp, "wb"))
            try:
                text, tables = parser(str(tmp), need_image=False, zoomin=3, return_html=True)
            except Exception as e:
                f.write(json.dumps({"page_id": pg["page_id"], "error": f"{type(e).__name__}: {e}"}, ensure_ascii=False) + "\n")
                f.flush(); continue
            chunks = []
            for para in [p for p in text.split("\n\n") if p.strip()]:
                c = clean(para)
                if c: chunks.append({"type": "text", "text": c})
            for item in tables:
                body = item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else item
                if isinstance(body, (list, tuple)): body = "\n".join(str(x) for x in body)
                html = str(body)
                c = clean(re.sub(r"<[^>]+>", " ", html))
                if c: chunks.append({"type": "table", "text": c, "html": html})
            f.write(json.dumps({"page_id": pg["page_id"], "code": pg["code"], "name": pg["name"],
                                "board": pg["board"], "page": pg["page"], "chunks": chunks},
                               ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0:
                el = time.time() - t0
                print(f"   {i}/{len(todo)}  {el/i:.1f}s/页  剩余约 {(len(todo)-i)*el/i/3600:.1f}h", flush=True)
    print(f"  完成，总耗时 {(time.time()-t0)/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
