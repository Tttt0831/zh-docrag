#!/bin/bash
# 模型下载调度。两个已确认的坑：
#  1) HF Xet 传输在本机卡死（45 秒零字节），必须 HF_HUB_DISABLE_XET=1 走普通 CDN
#  2) colqwen2.5-v0.2 的基座是 vidore/colqwen2.5-base，不是 Qwen2.5-VL-3B-Instruct
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
B=/home1/jiajunjie/envs/receiptvlm/bin/hf
L=/home1/jiajunjie/Projects/zh-docrag/logs
for m in vidore/colqwen2.5-v0.2 vidore/colqwen2.5-base OpenSearch-AI/Ops-Colqwen3-4B Qwen/Qwen3-VL-Embedding-2B; do
  echo "=== [$(date +%H:%M:%S)] $m"
  $B download "$m" >> "$L/$(echo $m|tr '/' '_').log" 2>&1 && echo "=== OK $m" || echo "=== FAIL $m"
done
echo "=== 全部结束 $(date +%H:%M:%S)"
