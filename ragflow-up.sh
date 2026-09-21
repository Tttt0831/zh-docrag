#!/bin/bash
# RAGFlow 启动脚本
#
# 前置（需要 root，只做一次）：
#   sudo sysctl -w vm.max_map_count=262144                      # ES 要求 >=262144，本机默认 65530
#   sudo sh -c 'echo vm.max_map_count=262144 >> /etc/sysctl.conf'   # 重启后保持
#   sudo nvidia-ctk runtime configure --runtime=docker           # 注册 nvidia 容器运行时
#   sudo systemctl restart docker
#
# 镜像来自 docker.m.daocloud.io 并已 retag —— Docker Hub 直连不通。
set -eu
cd "$(dirname "$0")/ragflow/docker"
export DOC_ENGINE=elasticsearch DEVICE=gpu
exec docker compose -f docker-compose.yml "$@"
