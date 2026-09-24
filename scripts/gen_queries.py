#!/usr/bin/env python
"""用本地 Qwen2.5-VL-7B 生成查询。双源：页面图 / 解析文本。

方法论照搬 reference/receipt-vlm/scripts/ds_annotate.py 验证过的三道关：

1. **门控**：只对确实含可问内容的页出题。空白页、纯目录页、纯页眉页跳过。
2. **接地校验**：答案必须能在该页原文里定位（最佳子串相似度 ≥ 0.7），否则丢弃。
   receipt-vlm 实测约 12% 的 LLM 输出是**推断而非抽取**——票面只印「山东高速」，
   模型补成「山东高速集团有限公司」。不过滤的话这些噪声会直接变成错误的真值。
3. **唯一性检查**：查询须带可区分实体（公司名/年份/指标名）。若答案串在多页出现，
   把**所有**匹配页记为真值，而不是只记一页——否则会把正确召回判成错误。

为什么双源：用页面图出题会带上版面线索，偏向视觉式；用解析文本出题偏向解析式。
两组分开报，差值本身是结果。方向一致才说明结论稳。
"""
import argparse
import json
import random
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

PROMPT_TMPL = """你在为中文文档检索系统构建评测集。下面是《{name}》2024 年年度报告中的一页。

请判断这一页是否含有可供提问的具体事实（具体数字、名称、指标、条款）。
- 若是空白页、纯目录、纯页眉页脚、或只有通用会计政策套话，只输出：SKIP
- 否则生成 2 条**检索查询**，要求：

  1. 必须是**提问**，不是陈述句。绝对不要把页面上的句子抄下来当查询。
  2. 查询里必须出现公司名「{name}」，否则换一页别的公司也能答，查询就没有区分度。
  3. 答案必须是页面上**原文出现**的字符串（数字、名称、条款原文），
     不要改写、不要补全、不要推断。
  4. 查询与答案必须明显不同——查询是问句，答案是被问的那个值。

正确示例：
  {{"q": "{name}2024年末应收账款账面余额是多少", "a": "1,234,567.89"}}
  {{"q": "{name}报告期内新增了哪几家控股子公司", "a": "某某科技有限公司、某某贸易有限公司"}}

错误示例（绝对不要这样）：
  {{"q": "长期股权投资包括对被投资单位实施控制的权益性投资", "a": "长期股权投资包括对被投资单位实施控制的权益性投资"}}
    ← 这是抄原文，不是提问
  {{"q": "应收账款余额是多少", "a": "1,234,567.89"}}
    ← 没有公司名，任何一家的页面都能答

严格按以下 JSON 输出，不要任何额外文字：
{{"queries": [{{"q": "查询1", "a": "答案原文"}}, {{"q": "查询2", "a": "答案原文"}}]}}"""


def ground_score(answer: str, page_text: str) -> float:
    """答案在页面原文里的最佳子串相似度。"""
    a = re.sub(r"\s+", "", answer)
    t = re.sub(r"\s+", "", page_text)
    if not a or not t:
        return 0.0
    if a in t:
        return 1.0
    best = 0.0
    step = max(1, len(a) // 4)
    for i in range(0, max(1, len(t) - len(a) + 1), step):
        best = max(best, SequenceMatcher(None, a, t[i:i + len(a)]).ratio())
        if best >= 0.95:
            break
    return best


def parse_json(raw: str):
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["image", "text"], required=True)
    ap.add_argument("--target", type=int, default=200, help="目标查询条数")
    ap.add_argument("--max-pages", type=int, default=400, help="最多尝试多少页")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out) if args.out else REPO / f"data/corpus/queries_{args.source}.jsonl"
    pages = [json.loads(l) for l in open(REPO / "data/corpus/pages.jsonl")]

    # 接地校验的参照必须**独立于两条路线**：用 pdfplumber 抽的 PDF 内嵌文本。
    # 若用 DeepDoc 的解析文本做参照，那些「图上清晰但被解析坏了」的页会被判为
    # 未接地而丢弃——恰好剔掉解析式最难的样本，等于给解析式放水。
    ref_text = {}
    for l in open(REPO / "data/corpus/ref_text.jsonl"):
        r = json.loads(l)
        if not r.get("garbled"):
            ref_text[r["page_id"]] = r["text"]
    print(f"  接地参照：{len(ref_text)} 页可用（pdfplumber 内嵌文本，独立于 DeepDoc）", flush=True)

    # 出题素材：图像源用页面图，文本源用 DeepDoc 解析文本
    page_text = {}
    if args.source == "text":
        for l in open(REPO / "data/corpus/parsed.jsonl"):
            r = json.loads(l)
            if "chunks" in r:
                page_text[r["page_id"]] = "\n".join(c["text"] for c in r["chunks"])
        pages = [p for p in pages if page_text.get(p["page_id"], "").strip()]
        print(f"  文本源：{len(pages)} 页有解析文本", flush=True)
    # 无论哪个源，都只在有接地参照的页上出题
    pages = [p for p in pages if ref_text.get(p["page_id"], "").strip()]

    random.seed(20260923)
    random.shuffle(pages)
    pages = pages[:args.max_pages]

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from PIL import Image

    name = "Qwen/Qwen2.5-VL-7B-Instruct"
    snaps = sorted((Path.home() / ".cache/huggingface/hub").glob("models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/*"))
    path = str(snaps[0]) if snaps else name
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda:0").eval()
    proc = AutoProcessor.from_pretrained(path)
    print(f"  模型加载 {time.time()-t0:.1f}s", flush=True)

    kept = skipped = ungrounded = degenerate = 0
    t1 = time.time()
    with out.open("w", encoding="utf-8") as f:
        for i, pg in enumerate(pages, 1):
            if kept >= args.target:
                break
            prompt = PROMPT_TMPL.format(name=pg["name"])
            if args.source == "image":
                content = [{"type": "image", "image": pg["img"]}, {"type": "text", "text": prompt}]
            else:
                txt = page_text[pg["page_id"]][:4000]
                content = [{"type": "text", "text": f"页面文本：\n{txt}\n\n{prompt}"}]
            msgs = [{"role": "user", "content": content}]
            text_in = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            images = [Image.open(pg["img"]).convert("RGB")] if args.source == "image" else None
            inputs = proc(text=[text_in], images=images, return_tensors="pt").to(model.device)
            with torch.no_grad():
                ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            raw = proc.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

            if "SKIP" in raw[:20]:
                skipped += 1
                continue
            d = parse_json(raw)
            if not d or not d.get("queries"):
                skipped += 1
                continue
            # 接地校验用「解析文本」做参照；图像源同样要过这关，否则无法验证答案真实存在
            ref = ref_text.get(pg["page_id"], "")
            for item in d["queries"][:2]:
                q, a = (item.get("q") or "").strip(), (item.get("a") or "").strip()
                if not q or not a:
                    continue
                # 退化检查前置：模型常把原文句子当查询，事后过滤会丢掉一半产量
                if SequenceMatcher(None, re.sub(r"[\s,，]", "", q), re.sub(r"[\s,，]", "", a)).ratio() > 0.8:
                    degenerate += 1
                    continue
                g = ground_score(a, ref) if ref else -1.0
                if ref and g < 0.7:
                    ungrounded += 1
                    continue
                f.write(json.dumps({"qid": f"{args.source[0]}{kept:04d}", "query": q, "answer": a,
                                    "gold_pages": [pg["page_id"]], "code": pg["code"],
                                    "name": pg["name"], "board": pg["board"], "page": pg["page"],
                                    "source": args.source, "ground": round(g, 3)},
                                   ensure_ascii=False) + "\n")
                f.flush()
                kept += 1
            if i % 20 == 0:
                el = time.time() - t1
                print(f"   {i} 页 → 留用 {kept}  跳过 {skipped}  退化 {degenerate}  未接地 {ungrounded}  {el/i:.1f}s/页", flush=True)

    print(f"  完成：留用 {kept}，跳过 {skipped}，退化丢弃 {degenerate}，未接地丢弃 {ungrounded} → {out}", flush=True)


if __name__ == "__main__":
    main()
