#!/bin/bash
# 起 RAGFlow 之前的自检：三项都 OK 才能起
ok=0
v=$(sysctl -n vm.max_map_count)
[ "$v" -ge 262144 ] && echo "  ✓ vm.max_map_count = $v" || { echo "  ✗ vm.max_map_count = $v（需 >=262144）"; ok=1; }
docker info 2>/dev/null | grep -q "nvidia" && echo "  ✓ nvidia 容器运行时已注册" || { echo "  ✗ nvidia 运行时未注册"; ok=1; }
for i in infiniflow/ragflow:v0.27.2 elasticsearch:8.11.3 mysql:8.0.40 valkey/valkey:8 pgsty/silo:RELEASE.2026-08-06T00-00-00Z; do
  docker image inspect "$i" >/dev/null 2>&1 && echo "  ✓ 镜像 $i" || { echo "  ✗ 镜像缺失 $i"; ok=1; }
done
exit $ok
