#!/usr/bin/env python
"""逐页解析，每个块带页码归属。

为什么不一次解析整份再从位置标记里抠页码：DeepDoc 把位置信息以
`@@页\tx0\tx1\ttop\tbottom##` 的形式**内联在文本里**，表格则根本不带位置
（need_position=False 时）。逐页解析虽然慢一点，但页码归属是确定的，
而页面级检索的真值就是页码，这里不能含糊。

模型只加载一次，31 页约 5 分钟。
"""
import json, re, sys, time
from pathlib import Path
import pypdf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import deepdoc_standalone  # noqa: E402,F401
from deepdoc.parser.pdf_parser import RAGFlowPdfParser  # noqa: E402

POS = re.compile(r"@@\d+\t[\d.]+\t[\d.]+\t[\d.]+\t[\d.]*#*")


def clean(t: str) -> str:
    """去掉内联的位置标记，否则这些数字会进向量，污染检索。"""
    return POS.sub(" ", t).replace("##", " ").strip()


def main():
    src = REPO / "data/pdf/test.pdf"
    first, last = 50, 80          # 1-based，与视觉式的页面池一致
    out = REPO / "data/parsed/pages_50_80.jsonl"
    tmp = REPO / "data/pdf/_one.pdf"

    reader = pypdf.PdfReader(str(src))
    parser = RAGFlowPdfParser()
    n = 0
    t0 = time.time()
    with out.open("w", encoding="utf-8") as f:
        for pg in range(first, last + 1):
            w = pypdf.PdfWriter()
            w.add_page(reader.pages[pg - 1])
            w.write(open(tmp, "wb"))
            try:
                text, tables = parser(str(tmp), need_image=False, zoomin=3, return_html=True)
            except Exception as e:
                print(f"   第 {pg} 页解析失败: {type(e).__name__} {e}", flush=True)
                continue
            for para in [p for p in text.split("\n\n") if p.strip()]:
                c = clean(para)
                if c:
                    f.write(json.dumps({"page": pg, "type": "text", "text": c}, ensure_ascii=False) + "\n")
                    n += 1
            for item in tables:
                body = item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else item
                if isinstance(body, (list, tuple)):
                    body = "\n".join(str(x) for x in body)
                c = clean(re.sub(r"<[^>]+>", " ", str(body)))   # 表格转纯文本供检索
                if c:
                    f.write(json.dumps({"page": pg, "type": "table", "text": c}, ensure_ascii=False) + "\n")
                    n += 1
            print(f"   第 {pg} 页完成，累计 {n} 块，用时 {time.time()-t0:.0f}s", flush=True)
    print(f"  共 {n} 块，总耗时 {time.time()-t0:.1f}s → {out}", flush=True)


if __name__ == "__main__":
    main()
