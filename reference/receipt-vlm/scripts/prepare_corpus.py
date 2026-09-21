# -*- coding: utf-8 -*-
"""
把原始语料流式 tokenize 成磁盘上的 uint16 token 流。

为什么要单独做这一步：
原来的 PackedLMDataset 在训练进程里把**整个语料**读成一个 Python int list
再转 int64 张量。300k 行时勉强能用，但 82 亿 token 的 Python list 约 300GB，
必然爆内存；即使装得下，int64 存储也比需要的多 4 倍（12k 词表用 uint16 足够）。

这里改成：一次性流式 tokenize → 追加写入 .bin（uint16）→ 训练时 memmap 读取。
内存占用与语料大小无关，且多次训练不必重复 tokenize。

用法:
  python scripts/prepare_corpus.py --out data/corpus/tokens_main \
      --jsonl data/corpus/minimind/pretrain_t2t.jsonl \
      --parquet-dir data/corpus/fineweb/4_5
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

BATCH = 2000            # 每批送进 tokenizer 的文档数


def iter_jsonl(path, text_key="text"):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = obj.get(text_key) or obj.get("content") or ""
            if t:
                yield t


def iter_parquet_dir(d, text_key="text"):
    import pyarrow.parquet as pq
    files = sorted(Path(d).rglob("*.parquet"))
    for fp in files:
        try:
            tb = pq.read_table(fp)
        except Exception as e:
            print(f"  [跳过] {fp.name}: {e}")
            continue
        col = text_key if text_key in tb.column_names else (
            "content" if "content" in tb.column_names else None)
        if col is None:
            print(f"  [跳过] {fp.name}: 没有 text/content 列 ({tb.column_names[:6]})")
            continue
        for t in tb.column(col).to_pylist():
            if t:
                yield t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="输出前缀，会生成 .bin 和 .meta.json")
    ap.add_argument("--tokenizer", default="tokenizers/receipt-bpe-32k")
    ap.add_argument("--jsonl", nargs="*", default=[])
    ap.add_argument("--parquet-dir", nargs="*", default=[])
    ap.add_argument("--max-tokens", type=int, default=0, help="达到该 token 数即停止（0=不限）")
    ap.add_argument("--text-key", default="text")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    eos = tok.eos_token_id
    if eos is None:
        raise SystemExit("tokenizer 没有 eos_token_id，无法做文档分隔")
    vocab = len(tok)
    if vocab > 65535:
        raise SystemExit(f"词表 {vocab} 超出 uint16 范围，需改用 uint32")
    print(f"tokenizer={args.tokenizer} vocab={vocab} eos={eos} fast={tok.is_fast}")

    out_bin = Path(args.out).with_suffix(".bin")
    out_bin.parent.mkdir(parents=True, exist_ok=True)

    sources = []
    for p in args.jsonl:
        sources.append(("jsonl", p, iter_jsonl(p, args.text_key)))
    for d in args.parquet_dir:
        sources.append(("parquet", d, iter_parquet_dir(d, args.text_key)))
    if not sources:
        raise SystemExit("至少要给一个 --jsonl 或 --parquet-dir")

    total_tok = n_doc = 0
    t0 = time.time()
    with open(out_bin, "wb") as fout:
        for kind, name, it in sources:
            print(f"\n── {kind}: {name} ──")
            buf = []
            src_tok = 0
            for text in it:
                buf.append(text)
                if len(buf) >= BATCH:
                    ids_list = tok(buf, add_special_tokens=False)["input_ids"]
                    flat = []
                    for ids in ids_list:
                        flat.extend(ids); flat.append(eos)
                    arr = np.asarray(flat, dtype=np.uint16)
                    fout.write(arr.tobytes())
                    total_tok += arr.size; src_tok += arr.size; n_doc += len(buf)
                    buf.clear()
                    if n_doc % 200000 < BATCH:
                        el = time.time() - t0
                        print(f"  {n_doc:,} 文档  {total_tok/1e9:.3f}B token  "
                              f"{total_tok/max(el,1)/1e6:.2f}M tok/s  {el/60:.1f} 分钟")
                    if args.max_tokens and total_tok >= args.max_tokens:
                        break
            if buf and not (args.max_tokens and total_tok >= args.max_tokens):
                ids_list = tok(buf, add_special_tokens=False)["input_ids"]
                flat = []
                for ids in ids_list:
                    flat.extend(ids); flat.append(eos)
                arr = np.asarray(flat, dtype=np.uint16)
                fout.write(arr.tobytes())
                total_tok += arr.size; src_tok += arr.size; n_doc += len(buf)
            print(f"  本源贡献 {src_tok/1e9:.3f}B token")
            if args.max_tokens and total_tok >= args.max_tokens:
                print("  已达 --max-tokens，停止")
                break

    meta = {"tokens": int(total_tok), "docs": int(n_doc), "dtype": "uint16",
            "vocab_size": int(vocab), "eos_id": int(eos),
            "tokenizer": args.tokenizer,
            "sources": [f"{k}:{n}" for k, n, _ in sources]}
    Path(args.out).with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    el = time.time() - t0
    print(f"\n完成: {n_doc:,} 文档 -> {total_tok:,} token ({total_tok/1e9:.2f}B)")
    print(f"      {out_bin}  {out_bin.stat().st_size/1e9:.1f} GB  用时 {el/60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
