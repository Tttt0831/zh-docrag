#!/bin/bash
# RAGFlow 源码不入库，用这个脚本拉。版本对齐本次验证过的 commit 所在分支。
set -eu
cd "$(dirname "$0")/.."
[ -d ragflow ] && { echo "ragflow/ 已存在"; exit 0; }
git clone --depth 1 https://github.com/infiniflow/ragflow.git
echo "已克隆。镜像版本见 ragflow/docker/.env 的 RAGFLOW_IMAGE（本次验证：v0.27.2）"
