"""
路线 B: Qwen2-VL-2B-Instruct + LoRA 微调

直接使用 Qwen2-VL-2B 现成 VLM，LoRA 微调。
与路线 A (SigLIP2+Projection+Qwen2-1.5B LoRA) 做同口径对比。

显存需求: ~20-22GB (RTX 4090 24GB)

用法:
    # 完整训练
    python src/train_qwen2vl_lora.py --data data/processed/synthetic/train.jsonl

    # Smoke test
    python src/train_qwen2vl_lora.py --max-samples 50 --epochs 1 --batch-size 1
"""

import argparse
import json
import os
import sys
import math
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ═══════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════

class QwenVLDataset(Dataset):
    """Qwen2-VL 格式数据集。"""

    def __init__(self, data_path: str, max_samples: Optional[int] = None):
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        if max_samples and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]
        print(f"Loaded {len(self.samples)} samples from {data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_path = s["image_path"]
        if not Path(img_path).exists():
            img_path = REPO_ROOT / img_path
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (384, 384), color="white")
        return {
            "image": image,
            "target_json": s["target_json"],
            "prompt": s.get("prompt", ""),
            "image_path": str(img_path),
        }


# ═══════════════════════════════════════════════════════════════════════
# Collate — Qwen2-VL chat 格式
# ═══════════════════════════════════════════════════════════════════════

QWEN_PROMPT = (
    "请从这张票据中抽取以下字段并以 JSON 格式输出:\n"
    "merchant_name (商户名称), date (开票日期 YYYY-MM-DD), "
    "total_amount (总金额), tax_amount (税额), "
    "tax_id (纳税人识别号), invoice_no (发票号码)。\n"
    "如果某个字段无法从票据中找到，请填 null。"
    "只输出 JSON，不要其他文字。"
)

ENGLISH_PROMPT = (
    "Extract the following fields from this receipt as JSON:\n"
    "merchant_name, date (YYYY-MM-DD), total_amount, tax_amount, "
    "tax_id, invoice_no.\n"
    "Use null for missing fields. Output only JSON, no extra text."
)


def collate_qwen2vl(batch, processor, max_length=2048):
    """构造 Qwen2-VL chat 格式 batch。"""
    user_messages_list = []
    full_messages_list = []
    targets = []

    for item in batch:
        is_english = "receipt_english" in item.get("image_path", "").lower()
        prompt = ENGLISH_PROMPT if is_english else QWEN_PROMPT
        target_text = json.dumps(item["target_json"], ensure_ascii=False)

        user_messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": item["image"]},
                {"type": "text", "text": prompt},
            ],
        }]
        full_messages = user_messages + [{
            "role": "assistant",
            "content": [{"type": "text", "text": target_text}],
        }]

        user_messages_list.append(user_messages)
        full_messages_list.append(full_messages)
        targets.append(item["target_json"])

    # Qwen2-VL 的批量处理
    from qwen_vl_utils import process_vision_info

    prompt_texts = []
    full_texts = []
    image_inputs_list = []
    for user_msgs, full_msgs in zip(user_messages_list, full_messages_list):
        prompt_texts.append(processor.apply_chat_template(
            user_msgs, tokenize=False, add_generation_prompt=True
        ))
        full_texts.append(processor.apply_chat_template(
            full_msgs, tokenize=False, add_generation_prompt=False
        ))
        image_inputs, _ = process_vision_info(user_msgs)
        image_inputs_list.append(image_inputs)

    prompt_inputs = processor(
        text=prompt_texts,
        images=image_inputs_list,
        padding=True,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
    )

    inputs = processor(
        text=full_texts,
        images=image_inputs_list,
        padding=True,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
    )

    labels = inputs["input_ids"].clone()
    prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1)
    seq_lengths = inputs["attention_mask"].sum(dim=1)
    padded_len = inputs["input_ids"].shape[1]

    # Qwen2-VL 的 tokenizer 默认 padding_side='left'。左填充时真实内容从
    # padded_len - seq_len 开始，直接 labels[i, :prompt_len] = -100 掩掉的是
    # padding，system prompt 和上千个 image token 会留在监督区，loss 会飙到
    # ~ln(vocab)。这里按实际 padding_side 定位 prompt 的起点。
    padding_side = getattr(processor.tokenizer, "padding_side", "right")
    for i in range(labels.shape[0]):
        start = padded_len - int(seq_lengths[i]) if padding_side == "left" else 0
        labels[i, : start + int(prompt_lengths[i])] = -100
    labels[inputs["attention_mask"] == 0] = -100

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "pixel_values": inputs.get("pixel_values"),
        "image_grid_thw": inputs.get("image_grid_thw"),
        "mm_token_type_ids": inputs.get("mm_token_type_ids"),
        "labels": labels,
        "targets": targets,
    }


# ═══════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════

def lr_at(step, total_steps, warmup_steps, base_lr):
    """线性 warmup + 余弦衰减。"""
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * step / warmup_steps
    if total_steps <= warmup_steps:
        return base_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * base_lr * (1 + math.cos(math.pi * min(progress, 1.0)))


def train_one_epoch(model, loader, optimizer, device, epoch, tcfg,
                    total_steps, step_offset, use_amp):
    model.train()
    total_loss = 0.0
    n = 0
    accum = tcfg.get("gradient_accumulation_steps", 1)
    base_lr = float(tcfg.get("learning_rate", 2e-4))
    warmup = tcfg.get("warmup_steps", 0)
    max_norm = tcfg.get("max_grad_norm", 1.0)

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]")
    for i, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Qwen2-VL 额外输入
        extra = {}
        for k in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
            if k in batch and batch[k] is not None:
                v = batch[k]
                extra[k] = v.to(device) if torch.is_tensor(v) else v

        with torch.set_grad_enabled(True):
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(input_ids=input_ids, attention_mask=attention_mask,
                                labels=labels, **extra)
                    loss = out.loss
            else:
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                            labels=labels, **extra)
                loss = out.loss

        (loss / accum).backward()

        if (i + 1) % accum == 0:
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step_offset + i, total_steps, warmup, base_lr)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        n += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(n, 1)


@torch.no_grad()
def validate(model, loader, device, use_amp):
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in tqdm(loader, desc="Validating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        extra = {}
        for k in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
            if k in batch and batch[k] is not None:
                v = batch[k]
                extra[k] = v.to(device) if torch.is_tensor(v) else v

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                            labels=labels, **extra)
        else:
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        labels=labels, **extra)

        loss = out.loss
        if torch.isfinite(loss):
            total_loss += loss.item()
            n += 1

    return total_loss / max(n, 1)


@torch.no_grad()
def validate_field_f1(model, processor, items, device, max_new_tokens=128):
    """
    在验证集上做真实推理并算字段级 F1。

    为什么不能只用 val loss 选 checkpoint：
    val loss 是 token 级的，一个日期值只占 JSON 里 ~7 个 token（总共 50~90 个）。
    模型对 date 输出 null 在 token loss 上几乎不受惩罚，但字段准确率直接掉 1/6。
    实测 N=8000 的 val loss 是四个规模里最低的，date 准确率却最差（82.67% vs
    94.33%）——按 val loss 选，恰好选中了字段级最差的模型。
    """
    import json as _json
    from qwen_vl_utils import process_vision_info
    sys.path.insert(0, str(REPO_ROOT))
    from src.eval import field_level_f1
    from src.utils.normalize import normalize_invoice_record

    model.eval()
    preds, tgts = [], []
    for item in tqdm(items, desc="Val F1"):
        is_en = "receipt_english" in item.get("image_path", "").lower()
        prompt = ENGLISH_PROMPT if is_en else QWEN_PROMPT
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": item["image"]},
            {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, _ = process_vision_info(msgs)
        inp = processor(text=[text], images=[imgs], padding=True, return_tensors="pt")
        inp = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inp.items()}
        gen = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        out = processor.tokenizer.decode(gen[0][inp["input_ids"].shape[1]:],
                                         skip_special_tokens=True)
        try:
            d = _json.loads(out)
        except Exception:
            a, b = out.find("{"), out.rfind("}")
            try:
                d = _json.loads(out[a:b + 1]) if a >= 0 and b > a else {}
            except Exception:
                d = {}
        preds.append(normalize_invoice_record(d if isinstance(d, dict) else {}))
        tgts.append(normalize_invoice_record(item["target_json"]))
    model.train()
    return field_level_f1(preds, tgts)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Qwen2-VL-2B LoRA 微调 (路线 B)")
    parser.add_argument("--data", default="data/synthetic/train/train.jsonl",
                        help="训练数据 jsonl 路径")
    parser.add_argument("--val-data", default="data/synthetic/val/val.jsonl",
                        help="验证数据 jsonl 路径")
    parser.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", default="checkpoints/route_b")
    parser.add_argument("--device", default=None)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--select-by", choices=["val_loss", "field_f1"],
                        default="val_loss",
                        help="按什么选 best checkpoint。field_f1 需要真实推理，"
                             "每 epoch 多花几分钟，但 val_loss 与字段级指标反相关")
    parser.add_argument("--f1-samples", type=int, default=200,
                        help="算 val F1 时用多少条验证样本")
    parser.add_argument("--config", default=None,
                        help="可选 yaml 配置（如 configs/route_b.yaml）；命令行参数优先级最高")
    args = parser.parse_args()

    # 可选 yaml 配置：仅覆盖用户未在命令行显式指定的项（保持与路线 A/C 一致的 config 驱动）
    if args.config:
        import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        m, d, t, c = (cfg.get(k, {}) for k in ("model", "data", "training", "checkpointing"))
        cli = {a.option_strings[0].lstrip("-").replace("-", "_") for a in parser._actions
               if any(o in sys.argv for o in a.option_strings)}
        defaults = {
            "model": m.get("hf_model_name"), "data": d.get("train_path"),
            "val_data": d.get("val_path"), "epochs": t.get("epochs"),
            "batch_size": t.get("batch_size"), "grad_accum": t.get("gradient_accumulation_steps"),
            "lr": t.get("learning_rate"), "warmup": t.get("warmup_steps"),
            "output_dir": c.get("output_dir"), "lora_r": m.get("lora_r"),
            "lora_alpha": m.get("lora_alpha"),
        }
        for k, v in defaults.items():
            if v is not None and k not in cli:
                setattr(args, k, v)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = (device.type == "cuda")
    print(f"Device: {device} | AMP: {use_amp}")

    # ── 加载模型 + LoRA ──
    print(f"\nLoading {args.model}...")
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from peft import get_peft_model, LoraConfig, TaskType

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if use_amp else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Processor ──
    processor = AutoProcessor.from_pretrained(
        args.model,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        trust_remote_code=True,
    )

    # ── 数据 ──
    train_ds = QwenVLDataset(args.data, max_samples=args.max_samples)
    val_path = args.val_data
    if val_path and Path(val_path).exists():
        val_ds = QwenVLDataset(val_path, max_samples=args.max_samples)
    else:
        n_val = max(1, int(0.1 * len(train_ds)))
        n_train = len(train_ds) - n_val
        from torch.utils.data import random_split
        train_ds, val_ds = random_split(train_ds, [n_train, n_val])
        print(f"Split: train={n_train}, val={n_val}")

    from functools import partial
    collate_fn = partial(collate_qwen2vl, processor=processor, max_length=2048)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    # ── 优化器 ──
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    # ── 训练循环 ──
    tcfg = {
        "learning_rate": args.lr,
        "gradient_accumulation_steps": args.grad_accum,
        "warmup_steps": args.warmup,
        "max_grad_norm": 1.0,
    }
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs

    out_dir = (REPO_ROOT / args.output_dir) if not Path(args.output_dir).is_absolute() \
        else Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best_model.pt"
    best_val = float("inf")

    print(f"\n{'='*50}")
    print(f"Training: {args.epochs} epochs, {len(train_loader)} batches/epoch")
    print(f"Checkpoint dir: {out_dir}")
    print(f"{'='*50}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, tcfg,
            total_steps, (epoch - 1) * steps_per_epoch, use_amp,
        )
        val_loss = validate(model, val_loader, device, use_amp)
        line = f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}"
        score = val_loss                       # 越小越好
        if args.select_by == "field_f1":
            m = validate_field_f1(model, processor,
                                  [val_ds[i] for i in range(min(args.f1_samples,
                                                                len(val_ds)))],
                                  device)
            line += (f", val_F1={m['f1']:.4f}"
                     f" (P={m['precision']:.4f} R={m['recall']:.4f})")
            score = -m["f1"]                   # 取负，统一成「越小越好」
        print(line)

        if score < best_val:
            best_val = score
            model.save_pretrained(out_dir / "best_lora")
            tag = (f"val_F1={-score:.4f}" if args.select_by == "field_f1"
                   else f"val={score:.4f}")
            print(f"  ✓ Saved best LoRA adapter ({tag})")

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")
    print(f"LoRA adapter saved at: {out_dir / 'best_lora'}")


if __name__ == "__main__":
    main()
