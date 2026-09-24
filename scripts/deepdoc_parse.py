#!/usr/bin/env python
"""DeepDoc 独立解析：不起 docker，不连 ES/MySQL/MinIO。

验证 RAGFlow 的解析核心能否脱离整套服务单独跑，并把结果落成 jsonl，
供后续解析式 vs 视觉式的对测使用。

用法：
    envs/deepdoc/bin/python scripts/deepdoc_parse.py data/pdf/test.pdf --from-page 59 --to-page 62
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RAGFLOW = REPO / "ragflow"

# 必须在 import deepdoc 之前：它会 stub 掉 common.settings（否则 import 链会拖进
# RAGFlow 整个存储层，要装十几个数据库客户端），并 ctypes 预加载 cuDNN/cuBLAS
# 让 onnxruntime 真正用上 GPU。细节见该模块的 docstring。
sys.path.insert(0, str(Path(__file__).resolve().parent))
import deepdoc_standalone  # noqa: E402,F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--from-page", type=int, default=0)
    ap.add_argument("--to-page", type=int, default=4)
    ap.add_argument("--out", default=None, help="输出 jsonl，默认 data/parsed/<名>.jsonl")
    args = ap.parse_args()

    from deepdoc.parser.pdf_parser import RAGFlowPdfParser

    out = Path(args.out) if args.out else REPO / "data/parsed" / (Path(args.pdf).stem + ".jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    parser = RAGFlowPdfParser()
    # 返回值结构（读源码确认，别想当然）：
    #   第 1 个是 __filterout_scraps() 的结果 —— 一整个**字符串**，段落间以 "\n\n" 分隔，
    #     不是块列表。当成列表遍历会逐字符拆开，得到一堆「1 字符文本块」的假象。
    #   第 2 个是 list[(PIL.Image, table)]，表格正文在 **[1]**；[0] 是裁出来的图片。
    #     return_html=True 时 [1] 是 HTML 字符串，否则是行列表。
    text, tables = parser(args.pdf, need_image=False, zoomin=3, return_html=True)
    dt = time.time() - t0

    paras = [p for p in text.split("\n\n") if p.strip()]
    n_tbl = 0
    with out.open("w", encoding="utf-8") as f:
        for p in paras:
            f.write(json.dumps({"type": "text", "text": p}, ensure_ascii=False) + "\n")
        for item in tables:
            body = item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else item
            if isinstance(body, (list, tuple)):
                body = "\n".join(str(x) for x in body)
            f.write(json.dumps({"type": "table", "html": str(body)}, ensure_ascii=False) + "\n")
            n_tbl += 1

    print(f"  解析耗时 {dt:.1f}s")
    print(f"  段落 {len(paras)}  表格 {n_tbl}  正文 {len(text)} 字符")
    print(f"  输出 {out}")


if __name__ == "__main__":
    main()
