#!/bin/bash
#SBATCH --job-name=zhdoc_genq
#SBATCH --output=/home/haiqian001/zh-docrag/logs/genq_%j.log
#SBATCH --error=/home/haiqian001/zh-docrag/logs/genq_%j.log
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
# 双源出题：看页面图出题 / 看 DeepDoc 解析文本出题，各 200 条。
# 两源必须同一模型同一提示词，差值才能归因到出题偏差（DESIGN 第三节）。
# 一个作业跑完两源再过滤，省去排队两次；7B bf16 约 16G 显存，A6000 48G 富余。
set -u
source ~/zh-docrag/scripts/env_cluster.sh
cd "$ZHDOC" || exit 1

for src in image text; do
  echo "=== [$(date +%H:%M:%S)] 出题源=$src"
  $PY scripts/gen_queries.py --source "$src" --target 200 --max-pages 500
  echo "=== 源 $src 退出码=$?"
done

echo "=== [$(date +%H:%M:%S)] 过滤（退化 + 接地 + 唯一性）"
$PY scripts/filter_queries.py
echo "GENQ_EXIT=$?"
wc -l data/corpus/queries_*_clean.jsonl 2>/dev/null
