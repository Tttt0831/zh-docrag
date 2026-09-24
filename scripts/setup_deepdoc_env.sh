#!/bin/bash
# DeepDoc 独立运行环境：不含 torch，不需要 docker，不需要 ES/MySQL/MinIO。
# 只为跑 deepdoc/vision 的 OCR + 版面 + 表格结构识别，以及 PDF 解析。
set -eu
V=/home1/jiajunjie/envs/deepdoc
export TMPDIR=/home1/jiajunjie/tmp        # /tmp 在已满的根分区上
mkdir -p "$TMPDIR"
export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
export PIP_TRUSTED_HOST=mirrors.aliyun.com
export PIP_CONSTRAINT=/home1/jiajunjie/Projects/zh-docrag/constraints.txt   # 防 pip 偷换 torch/cu13x

[ -d "$V" ] || /usr/bin/python3.12 -m venv "$V" 2>/dev/null || python3 -m venv "$V"
"$V/bin/pip" install -q --upgrade pip

# deepdoc 的实际 import 清单（逐个从源码里 grep 出来的，不是照抄 207 个依赖的 pyproject）
"$V/bin/pip" install \
  "onnxruntime-gpu==1.23.2" \
  "opencv-python-headless==4.10.0.84" \
  "shapely==2.0.5" \
  "pdfplumber==0.11.10" \
  "pypdf>=5.0" \
  "pyclipper>=1.4.0,<2.0.0" \
  "xgboost==1.6.0" \
  "scikit-learn==1.5.0" \
  "numpy<2" \
  "pandas" \
  "huggingface_hub" \
  "xpinyin==0.7.6" \
  "python-docx" "python-pptx" "openpyxl" \
  "chardet" "demjson3" "beautifulsoup4" "markdown" "six" \
  "infinity-sdk==0.7.3"

echo "=== 安装完成，自检 ==="
"$V/bin/python" - <<'PY'
import onnxruntime as ort
print("  onnxruntime", ort.__version__)
print("  providers  ", [p for p in ort.get_available_providers()])
PY
