"""
训练脚本（config 驱动，真实两阶段训练）

- Stage A (configs/stage1.yaml): 英文/CORD/SROIE 预训练，学会模态对齐 + JSON 格式。
- Stage B (configs/stage2.yaml): 加载 Stage A 权重，在中文票据上精调。

与旧版的区别：使用真实 tokenizer、真实图片、对齐的 causal-LM 标签
（损失只作用在 JSON 答案上），而非随机张量。

用法：
    python -m src.train --config configs/stage1.yaml
    # CPU 小样本 smoke：
    python -m src.train --config configs/stage1.yaml \
        --data data/synthetic/train.jsonl --vision fallback \
        --max-samples 50 --epochs 1 --batch-size 4
"""
import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model.vlm import ReceiptVLM, VLMConfig
from src.model.tokenizer import get_tokenizer
from src.data.dataset import ReceiptDataset, build_training_collate


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg, tokenizer, vision_model_name, device):
    """根据配置 + tokenizer 构建模型。vocab_size 由 tokenizer 决定。"""
    seq = cfg.get("sequence", {})
    freezing = cfg.get("freezing", {})
    model_cfg = cfg.get("model", {})
    strategy = freezing.get("freeze_strategy",
        "projection_llm_ends" if freezing.get("trainable_llm_layers") else "projection_only")

    vlm_config = VLMConfig(
        vision_model_name=vision_model_name,
        freeze_vision=freezing.get("freeze_vision", True),
        vision_trainable_layers=freezing.get("vision_trainable_layers", 0),
        max_num_patches=seq.get("max_num_patches", 256),
        llm_type=model_cfg.get("llm_type", "mini"),
        llm_vocab_size=tokenizer.vocab_size,
        llm_hidden_size=model_cfg.get("llm_hidden_size", 512),
        llm_num_layers=model_cfg.get("llm_num_layers", 6),
        llm_num_heads=model_cfg.get("llm_num_heads", 8),
        llm_intermediate_size=model_cfg.get("llm_intermediate_size", 2048),
        hf_model_name=model_cfg.get("hf_model_name", "Qwen/Qwen2-1.5B"),
        hf_lora_r=model_cfg.get("hf_lora_r", 16),
        hf_lora_alpha=model_cfg.get("hf_lora_alpha", 32),
        hf_lora_dropout=model_cfg.get("hf_lora_dropout", 0.05),
        projection_intermediate_dim=model_cfg.get("projection_intermediate_dim", 2048),
        projection_out_norm=model_cfg.get("projection_out_norm", True),
        projection_target_row_norm=model_cfg.get("projection_target_row_norm", 0.65),
        projection_pool=model_cfg.get("projection_pool", 1),
        max_sequence_length=seq.get("max_sequence_length", 1024),
        gradient_checkpointing=cfg.get("training", {}).get("gradient_checkpointing", False),
        freeze_strategy=strategy,
        pad_token_id=tokenizer.pad_token_id,
        image_token_id=tokenizer.image_token_id,
        boa_token_id=tokenizer.boa_token_id,
        eoa_token_id=tokenizer.eoa_token_id,
    )
    model = ReceiptVLM(vlm_config).to(device)
    return model


def make_loaders(train_path, val_path, max_samples, batch_size, collate,
                 max_val_samples=None):
    train_ds = ReceiptDataset(str(train_path), max_samples=max_samples, split="train")
    if val_path and Path(val_path).exists():
        # 验证集原先写死全量。1000 条验证一轮要 12.5 分钟，10 epoch 光验证
        # 就占 2 小时——和训练本身一个量级，纯属浪费。
        val_ds = ReceiptDataset(str(val_path), max_samples=max_val_samples, split="val")
    else:
        # 没有独立验证集 → 从训练集切 10%
        n_val = max(1, int(0.1 * len(train_ds)))
        n_train = len(train_ds) - n_val
        train_ds, val_ds = random_split(train_ds, [n_train, n_val])
        print(f"未找到验证集，从训练集切分: train={n_train}, val={n_val}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate, num_workers=0)
    return train_loader, val_loader


def lr_at(step, total_steps, warmup_steps, base_lr):
    """线性 warmup + 余弦衰减。"""
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * step / warmup_steps
    if total_steps <= warmup_steps:
        return base_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * base_lr * (1 + math.cos(math.pi * min(progress, 1.0)))


def run_epoch(model, loader, optimizer, device, train, epoch, tcfg, total_steps, step_offset, use_amp):
    model.train() if train else model.eval()
    desc = f"Epoch {epoch} [{'train' if train else 'val'}]"
    total_loss, n = 0.0, 0
    accum = tcfg.get("gradient_accumulation_steps", 1)
    base_lr = float(tcfg.get("learning_rate", 1e-4))
    warmup = tcfg.get("warmup_steps", 0)
    max_norm = tcfg.get("max_grad_norm", 1.0)

    pbar = tqdm(loader, desc=desc)
    for i, batch in enumerate(pbar):
        vision_inputs = batch["vision_inputs"]
        if isinstance(vision_inputs, dict):
            vision_inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in vision_inputs.items()}
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.set_grad_enabled(train):
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(vision_inputs, input_ids, attn, labels)
                    loss = out["loss"]
            else:
                out = model(vision_inputs, input_ids, attn, labels)
                loss = out["loss"]

        if train:
            (loss / accum).backward()
            if (i + 1) % accum == 0:
                sched_lr = lr_at(step_offset + i, total_steps, warmup, base_lr)
                for g in optimizer.param_groups:
                    g["lr"] = sched_lr * g.get("lr_scale", 1.0)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm)
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item()
        n += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description="Receipt-VLM 训练")
    parser.add_argument("--config", required=True, help="stage 配置 yaml")
    parser.add_argument("--data", default=None, help="覆盖训练数据 jsonl 路径")
    parser.add_argument("--val-data", default=None, help="覆盖验证数据 jsonl 路径")
    parser.add_argument("--vision", default=None, help="覆盖视觉编码器 (如 fallback)")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None,
                        help="验证集样本上限（None=全量）")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--init-from", default=None, help="初始化权重（Stage B 加载 Stage A）")
    parser.add_argument("--resume", default=None, help="从 checkpoint 恢复训练（加载权重+optimizer+epoch）")
    parser.add_argument("--init-llm", default=None,
                        help="载入 MiniLLM 语言预训练权重（src/pretrain_lm.py 产物）到 model.llm")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg.get("training", {})
    if args.epochs is not None:
        tcfg["epochs"] = args.epochs
    if args.batch_size is not None:
        tcfg["batch_size"] = args.batch_size

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = (device.type == "cuda") and (tcfg.get("mixed_precision") in ("bf16", "fp16"))
    print(f"Stage: {cfg.get('stage', {}).get('name')} | device: {device} | amp: {use_amp}")

    # 视觉编码器：CLI > config 默认（README 默认真 SigLIP2）
    vision_model_name = args.vision or cfg.get("model", {}).get("vision_model_name") \
        or "google/siglip2-base-patch16-naflex"

    tokenizer = get_tokenizer(cfg.get("model", {}).get("tokenizer_name"))
    model = build_model(cfg, tokenizer, vision_model_name, device)

    # 载入 MiniLLM 语言预训练权重（Stage 0 产物）到 model.llm
    if args.init_llm and Path(args.init_llm).exists():
        pre = torch.load(args.init_llm, map_location=device, weights_only=False)
        sd = pre.get("llm_state_dict", pre)
        missing, unexpected = model.llm.load_state_dict(sd, strict=False)
        print(f"✓ 已从 {args.init_llm} 载入 MiniLLM 语言预训练权重 "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    elif args.init_llm:
        print(f"⚠ --init-llm 指定的文件不存在: {args.init_llm}")

    model.print_trainable_parameters()

    # 初始化权重（Stage B 或 resume）
    start_epoch = 1
    init_from = args.init_from or cfg.get("checkpointing", {}).get("resume_from_checkpoint")
    resume_from = args.resume or cfg.get("checkpointing", {}).get("resume_from_checkpoint")

    def _load_hf_lora_ckpt(ckpt_dir, opt=None):
        """从 hf_lora 格式的 checkpoint 目录加载权重，可选恢复 optimizer state。

        Args:
            ckpt_dir: checkpoint 目录路径
            opt: 如果提供，尝试恢复 optimizer state
        """
        ckpt_dir = Path(ckpt_dir)
        meta = torch.load(ckpt_dir / "meta.pt", map_location=device, weights_only=False)
        # LoRA adapter — 需要先解包 PeftModel 拿到基座模型再重新包装
        from peft import PeftModel
        current_model = model.llm.model
        if hasattr(current_model, 'get_base_model'):
            base_transformer = current_model.get_base_model()
        elif hasattr(current_model, 'model'):
            base_transformer = current_model.model
        else:
            base_transformer = current_model
        model.llm.model = PeftModel.from_pretrained(
            base_transformer, str(ckpt_dir / "lora_adapter"), is_trainable=True)
        # 更新缓存的组件引用
        model.llm._embed_tokens = model.llm.model.get_input_embeddings()
        model.llm._lm_head = model.llm.model.get_output_embeddings()
        # Projection
        proj_sd = torch.load(ckpt_dir / "projection.pt", map_location=device, weights_only=False)
        model.projection.load_state_dict(proj_sd)
        # Vision trainable
        vis_path = ckpt_dir / "vision_trainable.pt"
        if vis_path.exists():
            vis_sd = torch.load(vis_path, map_location=device, weights_only=False)
            model.load_state_dict(vis_sd, strict=False)
            print(f"✓ 已载入 vision_trainable: {len(vis_sd)} 个参数")
        if opt is not None and "optimizer_state_dict" in meta:
            try:
                opt.load_state_dict(meta["optimizer_state_dict"])
                print(f"✓ 已恢复 optimizer state")
            except Exception as e:
                print(f"⚠ 恢复 optimizer state 失败（将从头开始）: {e}")
        return meta.get("epoch", 0), meta.get("val_loss", float("inf"))

    # 数据
    data_cfg = cfg.get("data", {})
    train_path = args.data or data_cfg.get("train_path")
    val_path = args.val_data or data_cfg.get("val_path")
    if not train_path or not Path(train_path).exists():
        raise SystemExit(f"训练数据不存在: {train_path}\n请用 --data 指定 jsonl，或先补齐 {data_cfg.get('train_path')}")

    collate = build_training_collate(
        tokenizer, model.vision_encoder.process_images,
        max_length=cfg.get("sequence", {}).get("max_sequence_length", 1024),
    )
    train_loader, val_loader = make_loaders(
        train_path, val_path, args.max_samples, tcfg.get("batch_size", 4), collate,
        max_val_samples=args.max_val_samples or cfg.get("data", {}).get("max_val_samples"))

    # 优化器（只优化可训练参数）。视觉解冻层用更小的 lr，避免破坏预训练特征。
    base_lr = float(tcfg.get("learning_rate", 1e-4))
    vis_scale = float(tcfg.get("vision_lr_scale", 0.1))
    vis_params = [p for n, p in model.named_parameters()
                  if p.requires_grad and n.startswith("vision_encoder.")]
    other_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and not n.startswith("vision_encoder.")]
    param_groups = [{"params": other_params, "lr": base_lr, "lr_scale": 1.0}]
    if vis_params:
        param_groups.append({"params": vis_params, "lr": base_lr * vis_scale, "lr_scale": vis_scale})
        print(f"✓ 优化器分组: 其他 {sum(p.numel() for p in other_params)/1e6:.1f}M @ lr={base_lr}, "
              f"视觉 {sum(p.numel() for p in vis_params)/1e6:.1f}M @ lr={base_lr*vis_scale}")
    optimizer = torch.optim.AdamW(param_groups, lr=base_lr,
                                  weight_decay=float(tcfg.get("weight_decay", 0.01)))

    # 恢复/初始化权重（必须在 optimizer 创建之后，以便恢复 optimizer state）
    if resume_from and Path(resume_from).exists():
        resume_path = Path(resume_from)
        if resume_path.is_dir():
            saved_epoch, saved_val = _load_hf_lora_ckpt(resume_path, opt=optimizer)
            start_epoch = saved_epoch + 1
            print(f"✓ 已从 {resume_from} 恢复训练 (epoch {saved_epoch}, val_loss={saved_val:.4f})，"
                  f"将从 epoch {start_epoch} 继续")
        else:
            ckpt = torch.load(resume_from, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            print(f"✓ 已从 {resume_from} 恢复训练 (epoch {ckpt.get('epoch')})，将从 epoch {start_epoch} 继续")
    elif init_from and Path(init_from).exists():
        init_path = Path(init_from)
        if init_path.is_dir():
            _load_hf_lora_ckpt(init_path, opt=None)
            print(f"✓ 已从 {init_from} 初始化权重（仅权重，不恢复 optimizer）")
        else:
            ckpt = torch.load(init_from, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"✓ 已从 {init_from} 初始化权重")

    epochs = tcfg.get("epochs", 3)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * epochs

    out_dir = Path(cfg.get("checkpointing", {}).get("output_dir", "checkpoints/stage"))
    out_dir = out_dir if out_dir.is_absolute() else REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best_model.pt"

    # 定期保存配置
    save_every_n = cfg.get("checkpointing", {}).get("save_every_n_epochs", 0)
    save_from_epoch = cfg.get("checkpointing", {}).get("save_from_epoch", 0)

    is_hf_lora = model.config.llm_type == "hf_lora"

    def _save_checkpoint(epoch, val_loss, path):
        """保存 checkpoint：HF LoRA 用轻量格式，MiniLLM 用完整 state_dict。"""
        if is_hf_lora:
            # 轻量保存：只存 LoRA adapter + projection + meta
            # 基座权重始终从 HuggingFace 拉取，避免 ~4GB 的 checkpoint 膨胀
            import shutil
            tmp = Path(str(path) + ".tmp")
            tmp.mkdir(parents=True, exist_ok=True)
            model.llm.model.save_pretrained(str(tmp / "lora_adapter"))
            torch.save(model.projection.state_dict(), tmp / "projection.pt")
            # 也保存视觉编码器的可训练部分（如有）
            # 注意：state_dict() 返回的 tensor 不保留 requires_grad 属性，
            # 必须用 named_parameters() 获取可训练参数名列表
            vis_trainable_names = {n for n, p in model.named_parameters()
                                   if n.startswith("vision_encoder.") and p.requires_grad}
            vis_sd = {k: v for k, v in model.state_dict().items()
                      if k in vis_trainable_names}
            vis_path = tmp / "vision_trainable.pt"
            torch.save(vis_sd, vis_path) if vis_sd else (vis_path.unlink() if vis_path.exists() else None)
            torch.save({
                "model_config": asdict(model.config),
                "vision_model_name": vision_model_name,
                "vocab_size": tokenizer.vocab_size,
                "epoch": epoch,
                "val_loss": val_loss,
                "stage": cfg.get("stage", {}).get("name"),
                "llm_type": "hf_lora",
                "optimizer_state_dict": optimizer.state_dict(),
            }, tmp / "meta.pt")
            # 原子替换
            if path.exists():
                shutil.rmtree(str(path), ignore_errors=True)
            tmp.rename(path)
            size_mb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024*1024)
            print(f"  ✓ 保存 checkpoint: {path} ({size_mb:.1f}MB, epoch {epoch})")
        else:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "vocab_size": tokenizer.vocab_size,
                "model_config": asdict(model.config),
                "vision_model_name": vision_model_name,
                "stage": cfg.get("stage", {}).get("name"),
            }, path)
            size_mb = path.stat().st_size / (1024*1024)
            print(f"  ✓ 保存 checkpoint: {path} ({size_mb:.0f}MB, epoch {epoch})")

    best_val = float("inf")
    for epoch in range(start_epoch, epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, True, epoch,
                               tcfg, total_steps, (epoch - 1) * steps_per_epoch, use_amp)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, optimizer, device, False, epoch,
                                 tcfg, total_steps, 0, use_amp)
        print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

        # 定期保存 checkpoint（仅在达到 save_from_epoch 后）
        if save_every_n > 0 and epoch % save_every_n == 0 and epoch >= save_from_epoch:
            epoch_path = out_dir / f"epoch_{epoch}"
            _save_checkpoint(epoch, val_loss, epoch_path)

        if val_loss < best_val:
            best_val = val_loss
            _save_checkpoint(epoch, val_loss, best_path)

    print(f"\n训练完成。最优 val loss: {best_val:.4f}")
    print(f"checkpoint: {best_path}")


if __name__ == "__main__":
    main()
