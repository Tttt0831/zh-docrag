#!/bin/bash
# 412M MiniLLM 预训练 —— 单卡 GPU0，9.00B token（412M 的 Chinchilla 最优约 8.24B）
# 断点续训：--resume auto 会自动接上 checkpoints/route_c_412m/last.pt
set -u
cd /home1/jiajunjie/Projects/receipt-vlm
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${GPU:-0}

/home1/jiajunjie/envs/receiptvlm/bin/python src/pretrain_lm.py \
  --tokens data/corpus/tokens_main.bin \
  --out-dir checkpoints/route_c_412m \
  --hidden-size 1280 --num-layers 20 --num-heads 20 --intermediate-size 3456 \
  --block-size 512 --batch-size 16 --grad-accum 1 \
  --lr 3e-4 --min-lr-ratio 0.1 --warmup 5000 --weight-decay 0.1 \
  --epochs 1 --val-blocks 2000 \
  --eval-every 10000 --save-every 10000 \
  --num-workers 4 --resume auto
