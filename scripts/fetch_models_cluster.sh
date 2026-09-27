#!/bin/bash
# 集群版模型下载（原机器版见 fetch_models.sh）。三个坑照搬 HANDOFF 第六节：
#  1) HF Xet 传输在校园网卡死（零字节且不报错）→ HF_HUB_DISABLE_XET=1
#  2) colqwen2.5-v0.2 的基座是 vidore/colqwen2.5-base，不是 Qwen2.5-VL-3B-Instruct
#  3) 缓存必须落在 /projects：/home 配额仅 50G
# 与原机器的差别：只下本轮 DESIGN 五系统真正用到的三个模型，不下 Ops-Colqwen3-4B（阶段 1 才用）。
set -eu
P=/projects/$USER
export HF_HOME=$P/hf_cache HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
HF=${HF_BIN:-$HOME/.local/bin/hf}
L=$HOME/zh-docrag/logs
mkdir -p "$L"

for m in Qwen/Qwen3-VL-Embedding-2B vidore/colqwen2.5-v0.2 vidore/colqwen2.5-base Qwen/Qwen2.5-VL-7B-Instruct; do
  log="$L/dl_$(echo "$m" | tr '/' '_').log"
  echo "=== [$(date +%H:%M:%S)] $m"
  if $HF download "$m" >> "$log" 2>&1; then echo "=== OK   $m"; else echo "=== FAIL $m（见 $log）"; fi
done
echo "=== 全部结束 $(date +%H:%M:%S)"
du -sh $HF_HOME/hub 2>/dev/null
