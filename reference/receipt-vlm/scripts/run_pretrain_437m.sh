#!/bin/bash
# MiniLLM 437M 预训练 —— 双卡 DDP，32k 词表
#   vs 上一次（已废弃）：词表 12k->32k（票据高频字覆盖 77.6%->95.9%，数字逐位），
#   参数 411.9M->437.5M（embedding 15.4M->41M），语料重新 tokenize 为 9.00B token。
set -u
cd /home1/jiajunjie/Projects/receipt-vlm
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${GPUS:-1,2}
export NCCL_P2P_DISABLE=1

/home1/jiajunjie/envs/receiptvlm/bin/torchrun --nproc_per_node=2 --master_port=29545 \
  src/pretrain_lm.py \
  --tokens data/corpus/tokens_32k.bin \
  --tokenizer tokenizers/receipt-bpe-32k \
  --out-dir checkpoints/route_c_437m \
  --hidden-size 1280 --num-layers 20 --num-heads 20 --intermediate-size 3456 \
  --block-size 512 --batch-size 16 --grad-accum 1 \
  --lr 3e-4 --min-lr-ratio 0.1 --warmup 5000 --weight-decay 0.1 \
  --epochs 1 --val-blocks 2000 \
  --eval-every 5000 --save-every 5000 \
  --num-workers 4 --resume auto
