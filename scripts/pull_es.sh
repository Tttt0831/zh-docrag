#!/bin/bash
L=/home1/jiajunjie/Projects/zh-docrag/logs
while ! grep -q "^=== 全部结束" $L/pull_mirror.log 2>/dev/null; do sleep 30; done
src=docker.m.daocloud.io/library/elasticsearch:8.11.3
echo "=== [$(date +%H:%M:%S)] pull $src"
docker pull "$src" && docker tag "$src" elasticsearch:8.11.3 && echo "=== OK elasticsearch:8.11.3" || echo "=== FAIL elasticsearch"
echo "=== ES 镜像结束 $(date +%H:%M:%S)"
