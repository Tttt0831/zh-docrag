# zh-docrag 集群公共环境变量。所有 sbatch 与手工运行都 source 这个文件。
# /home 配额 50G，HF 缓存、临时目录、pip/uv 缓存全部指到 /projects（本身是 HDD 层软链，配额 300G）。
export ZHDOC=$HOME/zh-docrag
export PY=/projects/$USER/envs/zhdocrag/bin/python
export HF_HOME=/projects/$USER/hf_cache
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_OFFLINE=1
export TMPDIR=/projects/$USER/tmp
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$ZHDOC/logs"
