#!/bin/bash
# 通过 daocloud 镜像源拉取 RAGFlow 所需镜像，再 retag 回原名
set -u
M=docker.m.daocloud.io
declare -A IMGS=(
  ["infiniflow/ragflow:v0.27.2"]="$M/infiniflow/ragflow:v0.27.2"
  ["infiniflow/infinity:v0.7.3-x64-v3"]="$M/infiniflow/infinity:v0.7.3-x64-v3"
  ["mysql:8.0.40"]="$M/library/mysql:8.0.40"
  ["pgsty/silo:RELEASE.2026-08-06T00-00-00Z"]="$M/pgsty/silo:RELEASE.2026-08-06T00-00-00Z"
)
for orig in "${!IMGS[@]}"; do
  src="${IMGS[$orig]}"
  echo "=== [$(date +%H:%M:%S)] pull $src"
  if docker pull "$src"; then docker tag "$src" "$orig"; echo "=== OK  $orig"
  else echo "=== FAIL $orig"; fi
done
echo "=== 全部结束 $(date +%H:%M:%S)"
