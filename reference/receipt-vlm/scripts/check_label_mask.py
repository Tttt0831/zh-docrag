# -*- coding: utf-8 -*-
"""
校验 SFT 的 label 掩码是否正确。

Qwen2-VL 的 tokenizer 默认 padding_side='left'。若掩码代码按右填充假设编写
（labels[i, :prompt_len] = -100），左填充时掩掉的是 padding，system prompt
和上千个 image token 会留在监督区，loss 会飙到 ~ln(vocab)。

本脚本构造一个长度差异明显的 batch（强制产生 padding），断言：
  1. 监督区 token 数 == 该样本 answer 的真实 token 数
  2. 解码监督区得到的正是 answer 文本，不含 system prompt / image token
  3. 左填充与右填充下结论一致

用法: python scripts/check_label_mask.py
"""
import sys
from pathlib import Path

import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

MODEL = "Qwen/Qwen2-VL-2B-Instruct"
IMAGE_TOKEN_IDS = set()


def build_batch():
    """两条样本，answer 长度差距大，保证 batch 内必然出现 padding。"""
    return [
        {"image": Image.new("RGB", (224, 224), "white"),
         "image_path": "x/synth_0.png",
         "target_json": {"merchant_name": "A公司", "date": "2024-01-01",
                         "total_amount": 1.0, "tax_amount": None,
                         "tax_id": None, "invoice_no": None}},
        {"image": Image.new("RGB", (224, 224), "white"),
         "image_path": "x/synth_1.png",
         "target_json": {"merchant_name": "北京神州数码科技有限公司",
                         "date": "2025-12-31", "total_amount": 98765.43,
                         "tax_amount": 1234.56,
                         "tax_id": "91110000MA00123456",
                         "invoice_no": "12345678901234"}},
    ]


def main() -> int:
    import json
    from transformers import AutoProcessor
    from src.train_qwen2vl_lora import collate_qwen2vl

    processor = AutoProcessor.from_pretrained(MODEL)
    tok = processor.tokenizer
    batch = build_batch()

    ok = True
    for side in ("left", "right"):
        tok.padding_side = side
        out = collate_qwen2vl(batch, processor)
        labels, ids = out["labels"], out["input_ids"]

        print(f"\n═══ padding_side = {side} ═══")
        print(f"batch 形状 {tuple(ids.shape)}  "
              f"padding token 数 {(out['attention_mask'] == 0).sum().item()}")

        for i, item in enumerate(batch):
            answer = json.dumps(item["target_json"], ensure_ascii=False)
            expect_n = len(tok(answer, add_special_tokens=False)["input_ids"])

            sup = labels[i] != -100
            got_n = int(sup.sum())
            decoded = tok.decode(ids[i][sup])

            # 监督区 = answer + 结尾的 <|im_end|>\n 之类，允许多 1~3 个 token
            n_ok = expect_n <= got_n <= expect_n + 3
            # 正文必须出现在监督区里，且不能混入 system prompt
            text_ok = answer in decoded and "system" not in decoded.lower()

            print(f"  样本{i}: 监督 {got_n} tok (answer {expect_n} tok) "
                  f"{'✓' if n_ok else '✗ 数量异常'}  "
                  f"{'✓' if text_ok else '✗ 内容异常'}")
            if not (n_ok and text_ok):
                ok = False
                print(f"    解码: {decoded[:300]!r}")

    print()
    if ok:
        print("✅ label 掩码在左/右填充下均正确：只监督 answer")
        return 0
    print("❌ label 掩码有误")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
