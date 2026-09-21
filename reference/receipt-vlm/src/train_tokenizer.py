"""
训练本项目自有的 BPE tokenizer，用于自制 MiniLLM 路线（路线 C）。

相对旧版（12k）的三处改动，每一处都有实测依据：

1. 词表 12k -> 32k
   12k 下常用 3500 字只有 69.1% 是单 token，其余要拆成 2~3 个字节 token；
   32k 提到 86.3%。压缩率 1.562 -> 1.730 字符/token，等价于同样的 token
   预算多覆盖 11% 的文字。再往上到 48k 只多涨 4.2pp，却要多 20M embedding
   参数，不划算。

2. 数字逐位切分
   旧版把 18 位税号切成 '9'/'15'/'200'/'00'/... 这种不规则块。票据任务要
   精确读出每一位，这种切法是明确的伤害（Llama、Qwen2 都改成逐位）。
   实测代价只有 1.5% 的压缩率——中文语料里数字占比低。

3. 训词表的语料
   旧版只用了 MiniMind mini 的前 30 万行 + 当时那版合成数据（商户名只有
   20 个固定公司），导致「茅鼎睿泓臻沪浙粤赣皖」这些票据高频字全部缺失。
   新版用真实预训练语料 + 当前开集合成器的完整字符池。

用法：
  python -m src.train_tokenizer --vocab-size 32000 --out tokenizers/receipt-bpe-32k
"""
import argparse
import glob
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

SPECIAL = ["<unk>", "<pad>", "</s>", "<image>", "<boa>", "<eoa>"]


def corpus_iter(jsonl_paths, parquet_dirs, max_docs, receipt_docs):
    """通用语料 + 票据域语料。"""
    n = 0
    for p in jsonl_paths:
        if not Path(p).exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if n >= max_docs:
                    break
                try:
                    t = json.loads(line).get("text", "")
                except json.JSONDecodeError:
                    continue
                if t:
                    yield t
                    n += 1
    import pyarrow.parquet as pq
    for d in parquet_dirs:
        for fp in sorted(Path(d).rglob("*.parquet")):
            if n >= max_docs:
                break
            try:
                tb = pq.read_table(fp)
            except Exception:
                continue
            col = "text" if "text" in tb.column_names else tb.column_names[0]
            for t in tb.column(col).to_pylist():
                if n >= max_docs:
                    break
                if t:
                    yield t
                    n += 1

    # 票据域：用当前合成器生成，确保开集商户名的全部用字被看到
    import src.data.synth as S
    S.set_difficulty("hard")
    random.seed(0)
    for _ in range(receipt_docs):
        tmpl = random.choice(list(S.TEMPLATE_MAP))
        d = S.generate_invoice_data(tmpl)
        yield S.DEFAULT_PROMPT + "\n" + json.dumps(d, ensure_ascii=False,
                                                   separators=(",", ":"))
    # 字符池直接铺若干遍，保证低频字也进得了合并表
    pool = "".join(S._CN_CITY + S._CN_ZONE + S._CN_BRAND + S._CN_INDUSTRY + S._CN_ORG)
    pool += "".join(S._HW_NAMES_CN) + "".join(S._HW_NOTES_CN)
    pool += "壹贰叁肆伍陆柒捌玖拾佰仟万亿元角分整" + "沪浙粤赣皖苏鲁豫鄂湘闽渝蜀黔滇陕甘宁青新蒙藏桂琼"
    for _ in range(300):
        yield pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--jsonl", nargs="*", default=["data/corpus/minimind/pretrain_t2t.jsonl"])
    ap.add_argument("--parquet-dir", nargs="*", default=["data/corpus/fineweb/4_5"])
    ap.add_argument("--max-docs", type=int, default=400000)
    ap.add_argument("--receipt-docs", type=int, default=6000)
    ap.add_argument("--split-digits", action="store_true", default=True)
    ap.add_argument("--out", default="tokenizers/receipt-bpe-32k")
    args = ap.parse_args()

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    pre = []
    if args.split_digits:
        # 数字逐位隔离，BPE 不会把它们合并成 '15'/'200' 这类块
        pre.append(pre_tokenizers.Split(Regex(r"\d"), behavior="isolated"))
    pre.append(pre_tokenizers.ByteLevel(add_prefix_space=False))
    tok.pre_tokenizer = pre_tokenizers.Sequence(pre)
    tok.decoder = decoders.ByteLevel()

    print(f"训练 BPE: vocab={args.vocab_size} 数字逐位={args.split_digits}")
    print(f"  通用语料: {args.jsonl} + {args.parquet_dir} (最多 {args.max_docs:,} 篇)")
    tok.train_from_iterator(
        corpus_iter(args.jsonl, args.parquet_dir, args.max_docs, args.receipt_docs),
        trainer=trainers.BpeTrainer(
            vocab_size=args.vocab_size, special_tokens=SPECIAL,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True))

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>",
        eos_token="</s>", bos_token="</s>",
        additional_special_tokens=["<image>", "<boa>", "<eoa>"])
    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(str(out))
    print(f"✓ 已保存到 {out} | 实际 vocab={len(fast)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
