#!/bin/bash
# N=8000 重跑，checkpoint 按验证集字段 F1 选而非 val loss。
# 验证假设：原先 N=8000 的 date 退步（94.33%→82.67%）是 checkpoint 选择造成的，
# 因为 val loss 最低的那个 epoch 恰好是 date 拒答最多的。
set -u
cd /home1/jiajunjie/Projects/receipt-vlm
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${GPU:-1}
PY=/home1/jiajunjie/envs/receiptvlm/bin/python

echo "════ N=8000 训练（按 field_f1 选）$(date +%H:%M:%S) ════"
$PY src/train_qwen2vl_lora.py \
    --data data/synthetic/train/train.jsonl --val-data data/synthetic/val/val.jsonl \
    --epochs 5 --batch-size 2 --grad-accum 4 \
    --select-by field_f1 --f1-samples 200 \
    --output-dir checkpoints/scale_8000_f1sel > logs/n8000_f1sel_train.log 2>&1 \
  && echo ">>> 训练完成 $(date +%H:%M:%S)" || { echo "!!! 训练失败"; exit 1; }

echo "════ 评测（300条，与规模曲线同口径）════"
$PY -m src.run_eval --checkpoint checkpoints/scale_8000_f1sel/best_lora \
    --data data/synthetic/test/test.jsonl --max-samples 300 \
    --name scale_8000_f1sel --output evaluation_results/scale_8000_f1sel \
    > logs/n8000_f1sel_eval.log 2>&1 \
  && echo ">>> 评测完成 $(date +%H:%M:%S)" || echo "!!! 评测失败"
