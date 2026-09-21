#!/bin/bash
# 只跑 Stage2 + 评测（Stage1 已完成，权重在 checkpoints/route_a_stage1/）
set -u
cd /home1/jiajunjie/Projects/receipt-vlm
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${GPU:-3}
PY=/home1/jiajunjie/envs/receiptvlm/bin/python

CKPT=checkpoints/route_a_stage1/best_model.pt
[ -e "$CKPT" ] || CKPT=checkpoints/route_a_stage1
echo "════ Stage2 (init from $CKPT) $(date +%H:%M:%S) ════"
$PY src/train.py --config configs/route_a_stage2.yaml --init-from "$CKPT" \
    > logs/route_a_stage2.log 2>&1 \
  && echo ">>> Stage2 完成 $(date +%H:%M:%S)" || { echo "!!! Stage2 失败"; exit 1; }

echo "════ 评测 (300条) ════"
$PY -m src.run_eval --checkpoint checkpoints/route_a_stage2/best_model.pt \
    --data data/synthetic/test/test.jsonl --max-samples 300 \
    --name route_a --output evaluation_results/route_a_new \
    > logs/route_a_eval.log 2>&1 \
  && echo ">>> 评测完成 $(date +%H:%M:%S)" || echo "!!! 评测失败"
