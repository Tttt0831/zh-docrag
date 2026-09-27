#!/bin/bash
# 集群版 zhdocrag 环境（原机器版见 setup_venv.sh）。
# 与原机器唯一差别：torch 同为 2.7.1，但换 cu128 构建——集群有 RTX 5090 / Pro 6000（Blackwell sm_120），
# cu126 构建不含 sm_120 的内核；cu128 同时覆盖 A40/A6000/6000Ada。其余版本照 HANDOFF 第五节。
# /home 配额只有 47G：环境、uv/pip 缓存、临时目录全部放 /projects。
set -eu
P=/projects/$USER
V=$P/envs/zhdocrag
export UV_CACHE_DIR=$P/caches/uv PIP_CACHE_DIR=$P/pip_cache TMPDIR=$P/tmp
mkdir -p "$TMPDIR"

# 走阿里云：torch 官方源解析出的 nvidia-* 依赖指向 pypi.nvidia.com，本机对该域名超时
# （三次重试 128s 全灭）。阿里云同时镜像了 PyPI 与 pytorch-wheels/cu128，两处都已核实有对应 wheel。
export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/

echo "=== [1/3] venv $V"
[ -x $V/bin/python ] || uv venv -q --python 3.11 $V

echo "=== [2/3] 依赖"
uv pip install -q -p $V/bin/python --index-strategy unsafe-best-match \
    --find-links https://mirrors.aliyun.com/pytorch-wheels/cu128/ \
    "torch==2.7.1+cu128" "torchvision==0.22.1+cu128" \
    "transformers==5.17.0" "sentence-transformers==6.1.0" "colpali-engine==0.3.18" "peft==0.20.0" \
    accelerate pillow numpy einops

echo "=== [3/3] 自检"
$V/bin/python - <<'PY'
import torch, transformers, sentence_transformers, peft, colpali_engine
print("  torch", torch.__version__, "| arch", torch.cuda.get_arch_list() if torch.cuda.is_available() else "(登录节点无 GPU)")
print("  transformers", transformers.__version__, "| sentence-transformers", sentence_transformers.__version__)
print("  peft", peft.__version__, "| colpali_engine", colpali_engine.__version__)
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from transformers import Qwen2_5_VLForConditionalGeneration
print("  imports OK")
PY
echo "=== 完成 $(date +%H:%M:%S)"
