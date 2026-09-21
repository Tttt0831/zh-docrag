#!/bin/bash
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
L=/home1/jiajunjie/Projects/zh-docrag/logs
while ! grep -q "^=== 全部结束" $L/fetch.log 2>/dev/null; do sleep 30; done
m=Qwen/Qwen3-VL-Embedding-8B
echo "=== [$(date +%H:%M:%S)] $m"
/home1/jiajunjie/envs/receiptvlm/bin/hf download "$m" >> "$L/Qwen_Qwen3-VL-Embedding-8B.log" 2>&1 \
  && echo "=== OK $m" || echo "=== FAIL $m"
echo "=== 第二批结束 $(date +%H:%M:%S)"
