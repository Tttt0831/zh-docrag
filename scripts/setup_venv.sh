#!/bin/bash
# zhdocrag 环境：Ops-Colqwen3-4B 要 transformers>=4.57，现有 receiptvlm 环境是 4.53.2 不能动。
# torch 锁在 2.7.1+cu126（本机已验证可用），wheel 走阿里云——官方源实测 4.8 KB/s。
set -eu
V=/home1/jiajunjie/envs/zhdocrag/bin
export PIP_CONSTRAINT=/home1/jiajunjie/Projects/zh-docrag/constraints.txt
export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
export PIP_FIND_LINKS=https://mirrors.aliyun.com/pytorch-wheels/cu126/
export PIP_TRUSTED_HOST="mirrors.aliyun.com"
# /tmp 在根分区上（已 100% 满），pip 解包必须改到 /home1，否则半路 No space left
export TMPDIR=/home1/jiajunjie/tmp
mkdir -p "$TMPDIR"
echo "=== [1/3] torch 2.7.1+cu126"
$V/pip install -q --upgrade pip
$V/pip install torch==2.7.1+cu126 torchvision==0.22.1+cu126
echo "=== [2/3] transformers 4.57+ / colpali-engine"
$V/pip install "transformers>=4.57,<5" colpali-engine accelerate peft pillow einops
echo "=== [3/3] 自检"
$V/python - <<'PY'
import torch, transformers
print("  torch", torch.__version__, "| cuda", torch.version.cuda, "| 可用", torch.cuda.is_available())
print("  transformers", transformers.__version__)
import importlib.util as u
for m in ["qwen3_vl","colqwen2_5"]:
    print(f"   models.{m:12s}", "有" if u.find_spec(f"transformers.models.{m}") else "无")
try:
    import colpali_engine; print("  colpali_engine", colpali_engine.__version__)
except Exception as e: print("  colpali_engine 失败:", e)
PY
echo "=== venv 完成 $(date +%H:%M:%S)"
