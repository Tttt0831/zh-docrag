#!/bin/bash
# 用修正后的 F1 口径重跑全部评测，并保存逐条预测。
# 旧口径把「抽错值」只记 FN 不记 FP，precision 恒为 1.0，F1 退化成 recall 的变换。
set -u
cd /home1/jiajunjie/Projects/receipt-vlm
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${GPU:-3}
PY=/home1/jiajunjie/envs/receiptvlm/bin/python

for N in 1000 2000 4000 8000; do
  echo "════ scale_${N} (300条) $(date +%H:%M:%S) ════"
  if $PY -m src.run_eval --checkpoint checkpoints/scale_${N}/best_lora \
      --data data/synthetic/test/test.jsonl --max-samples 300 \
      --name scale_${N} --output evaluation_results/scale_${N} \
      > logs/reeval_scale_${N}.log 2>&1; then
    echo ">>> scale_${N} 完成"
  else
    echo "!!! scale_${N} 失败，见 logs/reeval_scale_${N}.log"
  fi
done

echo "════ 基线 N=2000 (1000条) $(date +%H:%M:%S) ════"
if $PY -m src.run_eval --checkpoint checkpoints/scale_2000/best_lora \
    --data data/synthetic/test/test.jsonl --max-samples 1000 \
    --name route_b_N2000_full --output evaluation_results/route_b_baseline \
    > logs/reeval_baseline.log 2>&1; then
  echo ">>> 基线完成"
else
  echo "!!! 基线失败"
fi
echo "════ 全部完成 $(date +%H:%M:%S) ════"
