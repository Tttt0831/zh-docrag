#!/usr/bin/env python
"""切页段 + 渲染页面图，产出全局页面清单。

页段按文档长度的比例取，不写死页码——年报长度从 169 到 326 页不等。
15% 处多为经营讨论与管理层分析（文字+图表），50% 处多为财务报表区（密集表格），
两段版式差异大，覆盖面比连续取一段好。

页面 ID 用 `{code}_p{页码}`，页码是**原始 PDF 的 1-based 页码**，
这样任何结果都能追溯回原文件的具体一页。
"""
import json, subprocess, time, warnings
from pathlib import Path
import pypdf
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
IMG = REPO / "data/corpus/pages"
SPAN = 30
FRACS = (0.15, 0.50)
DPI = 110


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(REPO / "data/corpus/meta.jsonl")]
    manifest = []
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        n = len(pypdf.PdfReader(r["path"]).pages)
        ranges = []
        for f in FRACS:
            s = max(1, min(int(n * f), n - SPAN))
            ranges.append((s, s + SPAN - 1))
        # 两段若重叠则后移第二段
        if ranges[1][0] <= ranges[0][1]:
            s = min(ranges[0][1] + 1, n - SPAN)
            ranges[1] = (s, s + SPAN - 1)
        for (a, b) in ranges:
            subprocess.run(["pdftoppm", "-png", "-r", str(DPI), "-f", str(a), "-l", str(b),
                            r["path"], str(IMG / f"{r['code']}")], check=True)
        for (a, b) in ranges:
            for p in range(a, b + 1):
                for cand in (IMG / f"{r['code']}-{p:03d}.png", IMG / f"{r['code']}-{p:02d}.png",
                             IMG / f"{r['code']}-{p}.png"):
                    if cand.exists():
                        manifest.append({"page_id": f"{r['code']}_p{p}", "code": r["code"],
                                         "name": r["name"], "board": r["board"],
                                         "page": p, "img": str(cand), "pdf": r["path"]})
                        break
        print(f"   [{i}/{len(rows)}] {r['code']} {r['name'][:10]:10s} {n}页 → 段 {ranges}", flush=True)
    with open(REPO / "data/corpus/pages.jsonl", "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"  共 {len(manifest)} 页，耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
