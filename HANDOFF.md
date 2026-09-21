# 交接文档 — 环境搭建阶段

**日期** 2026-09-21 · **阶段** 0 未开始（正在拉两套方案的依赖）
**机器** 4 × RTX 4090D 24GB，`/home1` 剩 8.4T
**项目背景与研究计划见 [plan.md](plan.md)。本文只讲环境现状、踩过的坑、以及怎么接着干。**

---

## 一、一句话现状

要对测的两条路线——解析式（RAGFlow）和视觉式（ColQwen / Qwen3-VL-Embedding）——依赖正在下载。
**有一个硬阻塞：根分区 100% 满，RAGFlow 主镜像拉不下来。** 详见第四节，需要你决定怎么腾空间。

---

## 二、当前进度（2026-09-21 22:10 快照）

### 模型（HuggingFace 缓存 `~/.cache/huggingface`）

| 模型 | 用途 | 体积 | 状态 |
|---|---|---|---|
| `vidore/colqwen2.5-v0.2` | 视觉式·多向量，LoRA adapter | 0.26 GB | ✅ 完成 |
| `vidore/colqwen2.5-base` | 上者的基座（**不是** Qwen2.5-VL-3B-Instruct） | 7.5 GB | ✅ 完成 |
| `OpenSearch-AI/Ops-Colqwen3-4B` | 视觉式·多向量，ViDoRe SOTA | 8.9 GB | ⏳ 约 3.8/8.9 GB |
| `Qwen/Qwen3-VL-Embedding-2B` | 视觉式·单向量 | 4.3 GB | ⏳ 排队中 |
| `Qwen/Qwen3-VL-Embedding-8B` | 视觉式·单向量，分数最高 | 16.3 GB | ⏳ 排队中（`scripts/fetch_models_8b.sh` 等前序完成后自动开始） |

已在机器上、可直接用的：`Qwen/Qwen2.5-VL-7B-Instruct`（阶段 1 生成伪查询用）。

### Docker 镜像（RAGFlow）

| 镜像 | 状态 |
|---|---|
| `elasticsearch:8.11.3` | ✅ |
| `mysql:8.0.40` | ✅ |
| `valkey/valkey:8` | ✅ |
| `pgsty/silo:RELEASE.2026-08-06T00-00-00Z` | ✅ |
| `infiniflow/infinity:v0.7.3-x64-v3` | ✅（换 ES 后其实用不上，留着无妨） |
| **`infiniflow/ragflow:v0.27.2`** | ❌ **失败：`unexpected EOF`，根盘写满** |

### Python 环境

| 环境 | 用途 | 状态 |
|---|---|---|
| `~/envs/receiptvlm` | 旧环境，transformers 4.53.2 + torch 2.7.1+cu126 | 不要动。receipt-vlm 验证过的，改坏了没处找 |
| `~/envs/zhdocrag` | 新建，给 Ops-Colqwen3 用（要 transformers≥4.57） | ⏳ torch wheel 下到一半，**且会因 /tmp 满而失败**，见第四节 |

---

## 三、已确认的五个坑（都实际撞过，别重走）

### 1. HuggingFace 的 Xet 传输在本机卡死

小文件走普通 CDN 正常，大的 `.safetensors` 走 Xet，**45 秒零字节**，进程不报错也不退出。

```bash
export HF_HUB_DISABLE_XET=1     # 必须。加上立刻恢复正常速度
```

顺带：`hf-mirror.com` 实测 2 KB/s，不可用；`huggingface.co` 直连反而正常。

### 2. Docker Hub 直连不通

`registry-1.docker.io` 超时。测了 8 个镜像源，可用的：

| 源 | 结果 |
|---|---|
| `docker.m.daocloud.io` | ✅ 本次全部镜像都从这里拉的 |
| `registry.cn-hangzhou.aliyuncs.com` | ✅ 但只有 `infiniflow/ragflow`，没有 `infinity` 等 |
| `dockerproxy.net` / `docker.xuanyuan.me` | ✅ 可达，未实测拉取 |
| `hub-mirror.c.163.com` / `mirror.ccs.tencentyun.com` / `docker.1panel.live` | ❌ |

做法：`docker pull docker.m.daocloud.io/<原路径>` 后 `docker tag` 回原名，compose 就能用本地镜像。见 `scripts/pull_images.sh`。

### 3. `colqwen2.5-v0.2` 的基座不是 Qwen2.5-VL-3B-Instruct

adapter_config 里写的是 `base_model_name_or_path: vidore/colqwen2.5-base`——一个单独的 7.5 GB 仓库。
我一开始下了 `Qwen/Qwen2.5-VL-3B-Instruct`，白下。

### 4. `Ops-Colqwen3-4B` 要 transformers≥4.57 + torch 2.8.0

模型卡明写。它是 `trust_remote_code` 型（自带 `modeling_ops_colqwen3.py`），架构 `OpsColQwen3Model`，`model_type: ops_colqwen3`。
现有 `receiptvlm` 环境是 4.53.2，没有 `qwen3_vl` 模块，跑不了。所以新建了 `zhdocrag` venv。

**torch 我锁的是 2.7.1+cu126 而不是模型卡要的 2.8.0**——本机驱动 CUDA 上限 12.4，2.7.1+cu126 是已验证可用的组合。
如果 Ops-Colqwen3 实际跑起来报错再升 2.8.0，**升的时候必须带 `PIP_CONSTRAINT`**（见 `constraints.txt`），
否则 pip 会静默把 torch 换成 cu13x，装得上、跑不了。

pip 源也有坑：`download.pytorch.org` 实测 **4.8 KB/s**，得用阿里云；而阿里云的 pytorch-wheels 是**扁平目录不是 PEP503 布局**，
要用 `PIP_FIND_LINKS` 而不是 `--extra-index-url`，否则 `ResolutionImpossible`。

### 5. 根分区 100% 满（**当前阻塞**）

```
/dev/nvme0n1p4  864G  817G  3.4G  100%  /
/dev/sda1        19T  8.9T  8.4T   52%  /home1
```

Docker 数据目录在 `/var/lib/docker`，也就是在满的那块盘上：

- 镜像 71 个，230 GB
- 已停止容器 53 个，**405 GB 可写层**（最大的：`tester` 63.5G、`GS-ios4.5` 59.5G、`isaac-sim` 45.6G、`cuda118-ip188` 44.8G，都是 10~21 个月前的）
- 悬空镜像 41 个，25.1 GB
- 运行中的容器只有一个：`galbot-agent-db`

`ragflow:v0.27.2` 解压后约 15~20 GB，塞不下 → `unexpected EOF`。
**同样的原因，`zhdocrag` venv 的 pip 也会失败**——pip 默认往 `/tmp` 解包，已经占了 1.3 GB，torch 的 nvidia 依赖还有约 3 GB。
我已经把 `scripts/setup_venv.sh` 改成 `TMPDIR=/home1/jiajunjie/tmp`，重跑即可绕开。

---

## 四、需要你做的决定

**这些是你别的项目的东西（isaac-sim、galbot、curobo 等），我没动，也不建议我替你决定。**

### A. 腾空间（必须，否则 RAGFlow 起不来）

三个选项：

| 方案 | 回收 | 风险 |
|---|---|---|
| **迁 docker 数据目录到 `/home1`**（推荐） | 根盘腾出约 634 GB，且以后不再撞墙 | 低。不删任何东西，rsync 过去改 `data-root`。约 634 GB 要拷，半小时以上 |
| `docker image prune` 删悬空镜像 | 25 GB | 低，但可能不够——ragflow 要 15~20 GB，删完只剩约 28 GB |
| `docker container prune` 删停止容器 | 405 GB | **高**。镜像还在，但容器里装的环境、改过的配置、生成的数据全没 |

迁移方案的命令：

```bash
sudo systemctl stop docker
sudo rsync -aP /var/lib/docker/ /home1/docker/
echo '{"data-root": "/home1/docker"}' | sudo tee /etc/docker/daemon.json
sudo nvidia-ctk runtime configure --runtime=docker   # 顺手把 B 也做了
sudo systemctl start docker
docker images | head          # 验证镜像还在
# 确认无误后再： sudo rm -rf /var/lib/docker
```

### B. RAGFlow 跑 ES + GPU 的前置（你已选定这条路）

```bash
sudo sysctl -w vm.max_map_count=262144                          # ES 要求 ≥262144，本机默认 65530
sudo sh -c 'echo vm.max_map_count=262144 >> /etc/sysctl.conf'   # 重启后保持
sudo nvidia-ctk runtime configure --runtime=docker              # nvidia 运行时没注册，GPU profile 起不来
sudo systemctl restart docker
```

做完跑 `./preflight.sh` 自检，六项全 ✓ 才能起 RAGFlow。

---

## 五、恢复工作的命令

```bash
cd ~/Projects/zh-docrag

# 1) 继续下模型（断点续传，已完成的会跳过）
./scripts/fetch_models.sh &        # 前四个
./scripts/fetch_models_8b.sh &     # 8B，等前者完成后自动开始

# 2) 建 Python 环境（已改好 TMPDIR，腾出空间后重跑）
./scripts/setup_venv.sh

# 3) 拉 RAGFlow 源码（不入库）
./scripts/clone_ragflow.sh

# 4) 补拉失败的 ragflow 镜像（**必须先腾出根盘空间**）
docker pull docker.m.daocloud.io/infiniflow/ragflow:v0.27.2 \
  && docker tag docker.m.daocloud.io/infiniflow/ragflow:v0.27.2 infiniflow/ragflow:v0.27.2

# 5) 自检 → 起 RAGFlow
./preflight.sh && ./ragflow-up.sh up -d
```

Web UI 在 `http://<本机IP>`（nginx 80 端口），后端 9380。

---

## 六、仓库里有什么

| 路径 | 说明 |
|---|---|
| `plan.md` | **研究计划**：项目要回答什么、六源交叉验证的事实、阶段 0/1/2、终止条件 |
| `HANDOFF.md` | 本文，环境现状 |
| `preflight.sh` | 起 RAGFlow 前的六项自检 |
| `ragflow-up.sh` | RAGFlow 启动封装（ES + GPU，含前置说明） |
| `constraints.txt` | **pip 约束文件**，锁死 torch 防 cu13x，任何 pip install 都要带 |
| `scripts/` | 下模型、拉镜像、建 venv 的脚本，坑都写在注释里 |
| `reference/receipt-vlm/` | 前身项目源码 40 个 `.py`。**每个文件都踩过坑并修好了**，改造时别退回去。哪个文件修过什么见 plan.md 第六节 |
| `assets/why-visual.png` | 中粮生物 2023 年报第 60 页，项目动机的那张表 |

不入库：`ragflow/`（第三方克隆）、`logs/`（运行日志）。

---

## 七、下一步

环境齐了就进 **阶段 0a**：巨潮抓 ~200 份年报 → 渲染页面图 → 按难度分层 → 生成并人工核验 300~500 条查询。
方法论直接移植 `reference/receipt-vlm/scripts/ds_annotate.py` 的**门控 + 接地校验**——那里有实测数据：
LLM 约 12% 的输出是推断而非抽取（票面只印「山东高速」，它补成「山东高速集团有限公司」），必须过滤。

阶段 0 的**终止条件写在 plan.md 里，是写死的，不许事后放宽**。这是 receipt-vlm 该做没做的事。
