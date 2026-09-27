#!/bin/bash
#SBATCH --job-name=zhdoc_idx
#SBATCH --output=/home/haiqian001/zh-docrag/logs/idx_%j.log
#SBATCH --error=/home/haiqian001/zh-docrag/logs/idx_%j.log
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
# 建五系统索引（A/A′/B/D/E）后直接跑评测与配对自助法。
# 索引与评测放同一作业：评测要重新加载同样的两个模型，分开排队等于多等一次。
# E（ColQwen2.5 多向量）最重，2400 页每页约 1000 个向量。
set -u
source ~/zh-docrag/scripts/env_cluster.sh
cd "$ZHDOC" || exit 1

echo "=== [$(date +%H:%M:%S)] 建索引"
$PY scripts/build_indexes.py
rc=$?; echo "=== build_indexes 退出码=$rc"
[ $rc -ne 0 ] && exit $rc

echo "=== [$(date +%H:%M:%S)] 评测 + 统计检验"
$PY scripts/evaluate_all.py
echo "EVAL_EXIT=$?"
du -sh data/index 2>/dev/null
