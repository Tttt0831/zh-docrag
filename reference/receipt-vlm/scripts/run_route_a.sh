#!/bin/bash
# 路线 A 两阶段训练
#   Stage1  projection_only     —— 把随机初始化的 projection 对齐到 LLM embedding 空间
#   Stage2  projection_llm_ends —— 解冻 LLM 首末层精调，lr 降到 2e-5
set -u
cd /home1/jiajunjie/Projects/receipt-vlm
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${GPU:-3}
PY=/home1/jiajunjie/envs/receiptvlm/bin/python

echo "════ Stage1 (projection_only) $(date +%H:%M:%S) ════"
$PY src/train.py --config configs/route_a.yaml > logs/route_a_stage1.log 2>&1 \
  && echo ">>> Stage1 完成 $(date +%H:%M:%S)" || { echo "!!! Stage1 失败"; exit 1; }

CKPT=checkpoints/route_a_stage1/best_model.pt
[ -e "$CKPT" ] || CKPT=checkpoints/route_a_stage1
echo "════ Stage2 (projection_llm_ends, init from $CKPT) $(date +%H:%M:%S) ════"
$PY src/train.py --config configs/route_a_stage2.yaml --init-from "$CKPT" \
    > logs/route_a_stage2.log 2>&1 \
  && echo ">>> Stage2 完成 $(date +%H:%M:%S)" || { echo "!!! Stage2 失败"; exit 1; }

echo "════ 评测 (300条，与路线B同口径) ════"
$PY -m src.run_eval --checkpoint checkpoints/route_a_stage2 \
    --data data/synthetic/test/test.jsonl --max-samples 300 \
    --name route_a --output evaluation_results/route_a_new \
    > logs/route_a_eval.log 2>&1 \
  && echo ">>> 评测完成 $(date +%H:%M:%S)" || echo "!!! 评测失败"
