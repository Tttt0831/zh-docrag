#!/bin/bash
# 412M MiniLLM 预训练 —— 双卡 DDP（GPU1+GPU3）
# 从单卡的 last.pt 续训。单卡每步 8192 token，双卡每步 16384，
# 所以 gstep 要按 250000*8192/16384 = 125000 换算，否则 lr 曲线跳变、总量算少。
set -u
cd /home1/jiajunjie/Projects/receipt-vlm
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${GPUS:-1,3}
export NCCL_P2P_DISABLE=1        # 这台机器 P2P 是关的，显式声明避免 NCCL 探测卡住

/home1/jiajunjie/envs/receiptvlm/bin/torchrun --nproc_per_node=2 --master_port=29540 \
  src/pretrain_lm.py \
  --tokens data/corpus/tokens_main.bin \
  --out-dir checkpoints/route_c_412m \
  --hidden-size 1280 --num-layers 20 --num-heads 20 --intermediate-size 3456 \
  --block-size 512 --batch-size 16 --grad-accum 1 \
  --lr 3e-4 --min-lr-ratio 0.1 --warmup 5000 --weight-decay 0.1 \
  --epochs 1 --val-blocks 2000 \
  --eval-every 5000 --save-every 5000 \
  --num-workers 4 --resume auto --resume-gstep ${RGSTEP:-125000}
