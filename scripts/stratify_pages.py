#!/usr/bin/env python
"""给 2400 页打页面类型标签，两套口径并存，用于分层报告与交叉核对。

为什么要两套
------------
DESIGN 规定用 DeepDoc 的版面输出分层，但 DeepDoc 本身是解析式路线的一部分：
一张表若被 DeepDoc 漏认成正文，这页就被标成「纯文字」而不是「复杂表格」——
解析式最难的页恰好被分到别的层。这与接地校验不用 DeepDoc 文本是同一个道理。

所以再用 pdfplumber 读 PDF 矢量层独立标一遍。两者对「合并单元格」的理解不同：
DeepDoc 从图像识别表格结构（能看出没画线的跨行格），pdfplumber 靠表格线
划分单元格（实测 600165_p25：DeepDoc 见 rowspan=18，pdfplumber 只见 3 行无合并，
因为那张表内部没有横线）。两套口径不会逐页吻合，交叉核对要回答的是：
**换一个完全独立的标注方式，结论还在不在**。

口径
----
deepdoc：无表格块 → 纯文字；任一表格 html 含 rowspan/colspan≥2 → 复杂表格；否则普通表格
plumber：find_tables() 为空 → 纯文字；任一表存在被合并覆盖的格（row.cells 为 None）
         → 复杂表格；否则普通表格。另记 multiline：是否有单元格文字折行（窄列的信号）
跨页续表（两套各一）：deepdoc=本页首块与上页末块都是表格；
         plumber=本页有表顶端在页高 15% 内，且上页有表底端在页高 85% 外
"""
import json, re, sys, time, warnings
from pathlib import Path
import pdfplumber
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data/corpus/page_strata.jsonl"
MERGE = re.compile(r"(rowspan|colspan)\s*=\s*\"?([2-9]|\d{2,})")


def deepdoc_labels():
    lab, fl = {}, {}
    for l in open(REPO / "data/corpus/parsed.jsonl"):
        r = json.loads(l)
        ch = r.get("chunks", [])
        tabs = [c for c in ch if c["type"] == "table"]
        if not tabs:
            lab[r["page_id"]] = "纯文字"
        elif any(MERGE.search(c.get("html", "")) for c in tabs):
            lab[r["page_id"]] = "复杂表格"
        else:
            lab[r["page_id"]] = "普通表格"
        fl[r["page_id"]] = (ch[0]["type"] if ch else None, ch[-1]["type"] if ch else None)
    return lab, fl


def prev_id(pid):
    code, p = pid.rsplit("_p", 1)
    return f"{code}_p{int(p) - 1}"


def main():
    pages = [json.loads(l) for l in open(REPO / "data/corpus/pages.jsonl")]
    dd, fl = deepdoc_labels()
    pl, edges = {}, {}
    t0 = time.time()
    by_pdf = {}
    for p in pages:
        by_pdf.setdefault(p["pdf"], []).append(p)
    done = 0
    for path, ps in by_pdf.items():
        with pdfplumber.open(path) as pdf:
            for p in ps:
                pg = pdf.pages[p["page"] - 1]
                h = float(pg.height)
                try:
                    ts = pg.find_tables()
                except Exception:
                    ts = []
                merged = any(c is None for t in ts for row in t.rows for c in row.cells)
                multiline = False
                for t in ts:
                    for row in t.extract():
                        if any(c and "\n" in c for c in row):
                            multiline = True
                            break
                    if multiline:
                        break
                pl[p["page_id"]] = {"label": "纯文字" if not ts else ("复杂表格" if merged else "普通表格"),
                                    "n_tables": len(ts), "multiline": multiline}
                edges[p["page_id"]] = (any(t.bbox[1] < 0.15 * h for t in ts),
                                       any(t.bbox[3] > 0.85 * h for t in ts))
                done += 1
                if done % 200 == 0:
                    print(f"   {done}/{len(pages)}  {time.time() - t0:.0f}s", flush=True)
    with OUT.open("w", encoding="utf-8") as f:
        for p in pages:
            pid = p["page_id"]
            prv = prev_id(pid)
            f.write(json.dumps({
                "page_id": pid,
                "deepdoc": dd.get(pid, "无解析"),
                "deepdoc_cross": fl.get(pid, (0, 0))[0] == "table" and fl.get(prv, (0, 0))[1] == "table",
                "plumber": pl[pid]["label"],
                "plumber_multiline": pl[pid]["multiline"],
                "plumber_n_tables": pl[pid]["n_tables"],
                "plumber_cross": edges[pid][0] and edges.get(prv, (False, False))[1],
            }, ensure_ascii=False) + "\n")
    print(f"  完成 {len(pages)} 页，{time.time() - t0:.0f}s → {OUT}", flush=True)


if __name__ == "__main__":
    main()
