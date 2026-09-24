#!/usr/bin/env python
"""为五个对比系统建索引。

A  解析式·稠密    DeepDoc 文本块 → Qwen3-VL-Emb-2B
A′ 解析式·按行切块 表格按 <tr> 拆成行 → Qwen3-VL-Emb-2B   （消融：测「压平结构」的代价）
B  解析式·词法    DeepDoc 文本块 → BM25（字符 bigram）
D  视觉式·单向量  页面图 → Qwen3-VL-Emb-2B
E  视觉式·多向量  页面图 → ColQwen2.5 → MaxSim

C（混合）在评测时由 A+B 的分数融合得到，不需要单独索引。

BM25 为什么用字符 bigram：中文分词器会引入额外依赖和分词质量这个变量，
而字符 bigram 是中文检索里久经验证的无依赖做法，也避免「分词器选得好不好」
混进结论里。
"""
import argparse
import json
import math
import pickle
import re
import time
from collections import Counter
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
IDX = REPO / "data/index"
IDX.mkdir(parents=True, exist_ok=True)


def local_snapshot(repo_id: str) -> str:
    pat = f"models--{repo_id.replace('/', '--')}/snapshots/*/"
    snaps = sorted((Path.home() / ".cache/huggingface/hub").glob(pat))
    if not snaps:
        raise SystemExit(f"未找到 {repo_id} 本地快照")
    return str(snaps[0])


def load_chunks():
    """A/B 用：DeepDoc 默认切块。"""
    out = []
    for l in open(REPO / "data/corpus/parsed.jsonl"):
        r = json.loads(l)
        if "chunks" not in r:
            continue
        for j, c in enumerate(r["chunks"]):
            if c["text"].strip():
                out.append({"page_id": r["page_id"], "cid": f"{r['page_id']}#{j}",
                            "type": c["type"], "text": c["text"]})
    return out


ROW = re.compile(r"<tr>(.*?)</tr>", re.S)
CELL = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S)
TAG = re.compile(r"<[^>]+>")
CAP = re.compile(r"<caption>(.*?)</caption>", re.S)


def load_chunks_rowlevel():
    """A′ 用：表格按 <tr> 拆行，每行带上表格标题作为上下文。

    动机：MVP 里第 55 页整张三级表头表被压成一个 2139 字符的块，单向量被稀释，
    具体是哪家公司哪一行反而淹没了。按行切块后每行自成检索单元，
    这能测出「解析式的劣势有多少其实来自切块粒度，而非解析本身」。
    """
    out = []
    for l in open(REPO / "data/corpus/parsed.jsonl"):
        r = json.loads(l)
        if "chunks" not in r:
            continue
        for j, c in enumerate(r["chunks"]):
            if c["type"] == "text":
                if c["text"].strip():
                    out.append({"page_id": r["page_id"], "cid": f"{r['page_id']}#t{j}",
                                "type": "text", "text": c["text"]})
                continue
            html = c.get("html", "")
            cap = CAP.search(html)
            cap = TAG.sub(" ", cap.group(1)).strip() if cap else ""
            rows = ROW.findall(html)
            if not rows:
                if c["text"].strip():
                    out.append({"page_id": r["page_id"], "cid": f"{r['page_id']}#r{j}",
                                "type": "table", "text": c["text"]})
                continue
            for k, tr in enumerate(rows):
                cells = [TAG.sub(" ", x).strip() for x in CELL.findall(tr)]
                line = " ".join(x for x in cells if x)
                if not line.strip():
                    continue
                out.append({"page_id": r["page_id"], "cid": f"{r['page_id']}#r{j}_{k}",
                            "type": "table_row", "text": (cap + " " + line).strip()})
    return out


def bigrams(s: str):
    s = re.sub(r"\s+", "", s)
    return [s[i:i + 2] for i in range(len(s) - 1)] + list(s)


class BM25:
    """字符 bigram BM25。k1=1.5, b=0.75 是标准取值。"""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(bigrams(d)) for d in docs]
        self.len = [sum(c.values()) for c in self.docs]
        self.avg = sum(self.len) / max(len(self.len), 1)
        df = Counter()
        for c in self.docs:
            df.update(c.keys())
        N = len(self.docs)
        self.idf = {t: math.log(1 + (N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def score_all(self, query):
        q = bigrams(query)
        scores = [0.0] * len(self.docs)
        for t in set(q):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, c in enumerate(self.docs):
                f = c.get(t)
                if not f:
                    continue
                dl = self.len[i]
                scores[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
        return scores


def embed_texts(model, texts, bs, tag):
    t0 = time.time()
    e = model.encode(texts, batch_size=bs, convert_to_tensor=True,
                     normalize_embeddings=True, show_progress_bar=False)
    print(f"  {tag}: {len(texts)} 条 → {tuple(e.shape)}  {time.time()-t0:.1f}s", flush=True)
    return e.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["A", "Aprime", "B", "D", "E"])
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    pages = [json.loads(l) for l in open(REPO / "data/corpus/pages.jsonl")]
    print(f"  页面 {len(pages)}", flush=True)

    if {"A", "Aprime", "D"} & set(args.only):
        from sentence_transformers import SentenceTransformer
        emb = SentenceTransformer(local_snapshot("Qwen/Qwen3-VL-Embedding-2B"),
                                  model_kwargs={"dtype": torch.bfloat16}, device="cuda:0")

    if "A" in args.only:
        ch = load_chunks()
        torch.save({"meta": ch, "emb": embed_texts(emb, [c["text"] for c in ch], args.batch, "A 默认切块")},
                   IDX / "A_text.pt")

    if "Aprime" in args.only:
        ch = load_chunks_rowlevel()
        print(f"  A′ 按行切块后 {len(ch)} 块（默认切块是 {len(load_chunks())} 块）", flush=True)
        torch.save({"meta": ch, "emb": embed_texts(emb, [c["text"] for c in ch], args.batch, "A′ 按行切块")},
                   IDX / "Aprime_text.pt")

    if "B" in args.only:
        ch = load_chunks()
        t0 = time.time()
        bm = BM25([c["text"] for c in ch])
        with open(IDX / "B_bm25.pkl", "wb") as f:
            pickle.dump({"meta": ch, "bm25": bm}, f)
        print(f"  B BM25 建索引 {len(ch)} 块  {time.time()-t0:.1f}s", flush=True)

    if "D" in args.only:
        e = embed_texts(emb, [{"image": p["img"]} for p in pages], args.batch, "D 页面图单向量")
        torch.save({"meta": pages, "emb": e}, IDX / "D_image.pt")

    if "E" in args.only:
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
        from PIL import Image
        path = local_snapshot("vidore/colqwen2.5-v0.2")
        m = ColQwen2_5.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
        pr = ColQwen2_5_Processor.from_pretrained(path)
        t0 = time.time()
        embs = []
        for i in range(0, len(pages), 4):
            imgs = [Image.open(p["img"]).convert("RGB") for p in pages[i:i + 4]]
            batch = pr.process_images(imgs).to(m.device)
            with torch.no_grad():
                out = m(**batch)
            embs.extend([x.to(torch.float16).cpu() for x in torch.unbind(out)])
            if (i // 4) % 50 == 0:
                print(f"   E {i+len(imgs)}/{len(pages)}  {time.time()-t0:.0f}s", flush=True)
        torch.save({"meta": pages, "emb": embs}, IDX / "E_colqwen.pt")
        print(f"  E 多向量: {len(embs)} 页，单页 {tuple(embs[0].shape)}  {time.time()-t0:.1f}s", flush=True)

    print("  建索引完成", flush=True)


if __name__ == "__main__":
    main()
