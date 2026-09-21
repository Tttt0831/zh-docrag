# -*- coding: utf-8 -*-
"""
MiniLLM 语言预训练（支持多卡 DDP、断点续训、memmap 语料）。

相对旧版的改动（都是为了能真正跑几十小时的训练）：
  * 语料改为 memmap 读取（见 scripts/prepare_corpus.py），内存占用与语料无关。
    旧版把整个语料读成 Python int list 再转 int64 张量，82 亿 token 约需
    300GB 内存，必然 OOM。
  * 支持 torchrun 多卡 DDP。
  * 断点续训：保存 model/optimizer/step/最佳 val，崩了能接着跑，不用从头。
  * weight decay 不再施加到 RMSNorm 权重与 embedding 上。
  * 划出验证集，定期报 val loss / ppl，否则几十小时的训练没有任何收敛信号。

单卡:
  python src/pretrain_lm.py --tokens data/corpus/tokens_main.bin ...
双卡:
  torchrun --nproc_per_node=2 src/pretrain_lm.py --tokens ... --batch-size 16
"""
import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoTokenizer
from src.model.llm import MiniLLM, LLMConfig


class MemmapBlockDataset(Dataset):
    """从磁盘上的 uint16 token 流里按 block_size 切块，内存里不驻留语料。"""

    def __init__(self, bin_path, block_size, start_block=0, end_block=None):
        self.path = str(bin_path)
        self.block_size = block_size
        n_tok = os.path.getsize(self.path) // 2          # uint16
        n_blk = n_tok // block_size
        self.start = start_block
        self.end = n_blk if end_block is None else min(end_block, n_blk)
        self._mm = None

    def __len__(self):
        return max(0, self.end - self.start)

    def __getitem__(self, i):
        if self._mm is None:                             # 每个 worker 各自打开
            self._mm = np.memmap(self.path, dtype=np.uint16, mode="r")
        o = (self.start + i) * self.block_size
        return torch.from_numpy(
            self._mm[o:o + self.block_size].astype(np.int64))


def lr_at(step, total, warmup, base, min_ratio=0.1):
    if warmup > 0 and step < warmup:
        return base * (step + 1) / warmup
    if total <= warmup:
        return base
    prog = (step - warmup) / max(1, total - warmup)
    cos = 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
    return base * (min_ratio + (1 - min_ratio) * cos)


def param_groups(model, wd):
    """只对 2 维以上的权重做 weight decay；RMSNorm 权重和 bias 不衰减。"""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    return [{"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0}]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="data/corpus/tokens_main.bin",
                    help="prepare_corpus.py 产出的 uint16 token 流")
    ap.add_argument("--tokenizer", default="tokenizers/receipt-bpe")
    ap.add_argument("--out-dir", default="checkpoints/route_c")
    ap.add_argument("--hidden-size", type=int, default=1280)
    ap.add_argument("--num-layers", type=int, default=20)
    ap.add_argument("--num-heads", type=int, default=20)
    ap.add_argument("--intermediate-size", type=int, default=3456)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16, help="每卡 micro-batch")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = 按 --epochs 走完语料")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--val-blocks", type=int, default=2000, help="末尾留作验证的 block 数")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--resume", default="", help="从该 checkpoint 续训；auto=用 out-dir/last.pt")
    ap.add_argument("--resume-gstep", type=int, default=-1,
                    help="覆盖 checkpoint 里的 gstep。改变卡数或 batch 时必须换算："
                         "gstep 的含义是「优化器步数」，而每步 token 数 = "
                         "batch*block*grad_accum*world_size。卡数翻倍则每步 token 翻倍，"
                         "沿用旧 gstep 会让 lr 曲线跳变、总训练量算少。"
                         "新 gstep = 旧 gstep * 旧每步token / 新每步token")
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    # ── DDP 初始化 ──
    ddp = int(os.environ.get("RANK", -1)) >= 0
    if ddp:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank(); local_rank = int(os.environ["LOCAL_RANK"])
        world = dist.get_world_size(); torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank, local_rank, world = 0, 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    def log(*a):
        if is_main:
            print(*a, flush=True)

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    vocab_size = len(tok)

    cfg = LLMConfig(vocab_size=vocab_size, hidden_size=args.hidden_size,
                    num_layers=args.num_layers, num_heads=args.num_heads,
                    intermediate_size=args.intermediate_size,
                    max_position_embeddings=max(args.block_size, 2048))
    model = MiniLLM(cfg).to(device)
    n_param = model.num_parameters
    log(f"MiniLLM {n_param/1e6:.1f}M | vocab={vocab_size} | world={world} | device={device}")

    # ── 数据：末尾 val_blocks 个块留作验证 ──
    bin_path = REPO_ROOT / args.tokens if not Path(args.tokens).is_absolute() else Path(args.tokens)
    if not bin_path.exists():
        raise SystemExit(f"token 文件不存在: {bin_path}\n先跑 scripts/prepare_corpus.py")
    n_tok_total = os.path.getsize(bin_path) // 2
    n_blk_total = n_tok_total // args.block_size
    n_val = min(args.val_blocks, max(1, n_blk_total // 100))
    train_ds = MemmapBlockDataset(bin_path, args.block_size, 0, n_blk_total - n_val)
    val_ds = MemmapBlockDataset(bin_path, args.block_size, n_blk_total - n_val, n_blk_total)
    log(f"语料 {n_tok_total/1e9:.2f}B token -> train {len(train_ds):,} block / val {len(val_ds):,} block")

    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True) if ddp else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=args.num_workers, drop_last=True, pin_memory=True,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, drop_last=True, pin_memory=True)

    opt = torch.optim.AdamW(param_groups(model, args.weight_decay),
                            lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    steps_per_epoch = len(train_loader) // args.grad_accum
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    log(f"每 epoch {steps_per_epoch:,} step | 目标 {total_steps:,} step | "
        f"每 step {args.batch_size*args.block_size*args.grad_accum*world:,} token")

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    gstep, start_epoch, best_val = 0, 0, float("inf")

    # ── 断点续训 ──
    resume_path = None
    if args.resume == "auto":
        cand = out_dir / "last.pt"
        resume_path = cand if cand.exists() else None
    elif args.resume:
        resume_path = Path(args.resume)
    if resume_path and resume_path.exists():
        ck = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["llm_state_dict"])
        opt.load_state_dict(ck["optimizer"])
        gstep = ck.get("gstep", 0); start_epoch = ck.get("epoch", 0)
        best_val = ck.get("best_val", float("inf"))
        tok_per_step = args.batch_size * args.block_size * args.grad_accum * world
        log(f"✓ 续训自 {resume_path}  step={gstep} epoch={start_epoch} best_val={best_val:.4f}")
        if args.resume_gstep >= 0:
            old = gstep; gstep = args.resume_gstep
            log(f"  gstep 换算: {old} -> {gstep}（每步 token 变为 {tok_per_step:,}）")
        prev_tok = ck.get("tokens_seen")
        if prev_tok:
            log(f"  checkpoint 记录已见 {prev_tok/1e9:.3f}B token；"
                f"当前口径推算 {gstep*tok_per_step/1e9:.3f}B")

    if ddp:
        model = DDP(model, device_ids=[local_rank])
    raw = model.module if ddp else model

    def save(name, extra=None):
        if not is_main:
            return
        ck = {"llm_state_dict": raw.state_dict(), "optimizer": opt.state_dict(),
              "llm_config": asdict(cfg), "gstep": gstep, "epoch": epoch,
              "best_val": best_val, "vocab_size": vocab_size,
              "tokenizer_dir": args.tokenizer,
              # 记录绝对 token 数，换卡数续训时不必靠 gstep 反推
              "tokens_seen": gstep * args.batch_size * args.block_size
                             * args.grad_accum * world}
        if extra:
            ck.update(extra)
        tmp = out_dir / (name + ".tmp")
        torch.save(ck, tmp); tmp.replace(out_dir / name)   # 原子替换，防写坏

    @torch.no_grad()
    def evaluate(max_batches=100):
        model.eval(); tot, cnt = 0.0, 0
        for i, ids in enumerate(val_loader):
            if i >= max_batches:
                break
            ids = ids.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = raw(ids)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].float().reshape(-1, vocab_size), ids[:, 1:].reshape(-1))
            tot += loss.item(); cnt += 1
        model.train()
        v = torch.tensor([tot, cnt], device=device, dtype=torch.float64)
        if ddp:
            dist.all_reduce(v, op=dist.ReduceOp.SUM)
        return (v[0] / max(v[1], 1)).item()

    model.train()
    t0 = time.time()
    done = False
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not is_main)
        opt.zero_grad(set_to_none=True)
        for i, ids in enumerate(pbar):
            ids = ids.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids)
            # loss 强制 fp32，避免 bf16 数值问题污染梯度
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].float().reshape(-1, vocab_size), ids[:, 1:].reshape(-1))
            (loss / args.grad_accum).backward()

            if (i + 1) % args.grad_accum == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(gstep, total_steps, args.warmup, args.lr, args.min_lr_ratio)
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                opt.step(); opt.zero_grad(set_to_none=True); gstep += 1

                if is_main:
                    tok_seen = gstep * args.batch_size * args.block_size * args.grad_accum * world
                    pbar.set_postfix({"loss": f"{loss.item():.3f}",
                                      "ppl": f"{math.exp(min(loss.item(),20)):.1f}",
                                      "lr": f"{opt.param_groups[0]['lr']:.2e}",
                                      "gn": f"{float(gn):.2f}",
                                      "tok": f"{tok_seen/1e9:.2f}B"})
                if args.eval_every and gstep % args.eval_every == 0:
                    vl = evaluate()
                    log(f"\n[step {gstep}] val_loss={vl:.4f} ppl={math.exp(min(vl,20)):.1f} "
                        f"elapsed={(time.time()-t0)/3600:.2f}h")
                    if vl < best_val:
                        best_val = vl; save("best.pt")
                        log(f"  ✓ 新的最优 val，已保存 best.pt")
                if args.save_every and gstep % args.save_every == 0:
                    save("last.pt")
                if args.max_steps and gstep >= args.max_steps:
                    done = True; break
        save("last.pt")
        if done:
            break

    # 收尾的 evaluate() 内部有 all_reduce，必须**所有 rank 都调用**。
    # 只让 rank0 调会让它在 all_reduce 上死等已经退出的 rank1，NCCL 超时后崩溃
    # —— 而且只在训练全部跑完时触发，即跑完几十小时才在最后一步炸。
    vl = evaluate()
    if is_main:
        log(f"\n预训练结束 step={gstep} val_loss={vl:.4f} 用时 {(time.time()-t0)/3600:.2f}h")
        if vl < best_val:
            best_val = vl; save("best.pt")
        # 兼容路线 C 第二阶段：train.py --init-llm 期望的文件名
        save("llm_pretrained.pt")
        log(f"权重: {out_dir}/best.pt  {out_dir}/llm_pretrained.pt")
    if ddp:
        dist.barrier()      # 等 rank0 写完再一起拆进程组
    if ddp:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
