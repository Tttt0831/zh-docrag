#!/bin/bash
# 数据规模曲线：1000/2000/4000/8000 各训一次，测同一个 test 集。
# 子集是嵌套且模板分层的（见 scripts/make_subsets.py），所以曲线上的差异
# 只来自数据量，不来自模板配比漂移或样本重新洗牌。
set -u
cd /home1/jiajunjie/Projects/receipt-vlm
PY=/home1/jiajunjie/envs/receiptvlm/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPU=${GPU:-1}
EPOCHS=${EPOCHS:-5}
EVAL_N=${EVAL_N:-300}

for N in 1000 2000 4000 8000; do
  if [ "$N" = "8000" ]; then DATA=data/synthetic/train/train.jsonl
  else DATA=data/synthetic/train/train_${N}.jsonl; fi
  OUT=checkpoints/scale_${N}

  echo "════════ N=${N} 训练开始 $(date +%H:%M:%S) ════════"
  CUDA_VISIBLE_DEVICES=$GPU $PY src/train_qwen2vl_lora.py \
      --data "$DATA" --val-data data/synthetic/val/val.jsonl \
      --epochs $EPOCHS --batch-size 2 --grad-accum 4 \
      --output-dir "$OUT" > logs/scale_${N}_train.log 2>&1
  if [ $? -ne 0 ]; then echo "!!! N=${N} 训练失败，见 logs/scale_${N}_train.log"; continue; fi

  echo "════════ N=${N} 评测开始 $(date +%H:%M:%S) ════════"
  CUDA_VISIBLE_DEVICES=$GPU $PY -m src.run_eval \
      --checkpoint "${OUT}/best_lora" \
      --data data/synthetic/test/test.jsonl \
      --max-samples $EVAL_N --name "scale_${N}" \
      --output evaluation_results/scale_${N} > logs/scale_${N}_eval.log 2>&1
  if [ $? -ne 0 ]; then echo "!!! N=${N} 评测失败，见 logs/scale_${N}_eval.log"; continue; fi

  F1=$($PY -c "import json;print(json.load(open('evaluation_results/scale_${N}/metrics.json'))['f1'])" 2>/dev/null)
  echo ">>> N=${N} 完成  F1=${F1}  $(date +%H:%M:%S)"
done
echo "════════ 规模曲线全部完成 $(date +%H:%M:%S) ════════"
