# 交接文档

**更新** 2026-09-24 · **阶段** 0 执行中（语料与解析已完成，评测集生成中）
**机器** 4 × RTX 4090D 24GB · **研究计划见 [DESIGN.md](DESIGN.md)**

---

## 一、现在到哪一步了

| 环节 | 状态 |
|---|---|
| 无 docker 的解析链路 | ✅ DeepDoc 脱离 RAGFlow 服务栈独立运行，GPU 生效 |
| 语料抓取 | ✅ 40 份 2024 年报（四板块各 10 家），174 MB |
| 页面渲染 | ✅ 2400 页，589 MB |
| **DeepDoc 全量解析** | ✅ **2400 页，3.60 h，5.4 s/页，零失败**，12969 块 |
| 接地参照文本 | ✅ pdfplumber 抽取，2392 页可用，8 页乱码 |
| 评测集生成 | ⏳ 视觉源重跑中，文本源待跑 |
| 五系统索引 | ⬜ 脚本已就位（`build_indexes.py`） |
| 对比评测 | ⬜ 脚本已就位（`evaluate_all.py`） |

## 二、唯一的结论性数据（MVP，n=15，**不可外推**）

31 页候选池、15 条人工查询：

| | 解析式 | 视觉式 | 差 |
|---|---|---|---|
| 未受控（BGE-m3 vs ColQwen2.5） | 0.892 | 1.000 | 0.108 |
| **受控（同为 Qwen3-VL-Emb-2B）** | **0.901** | **0.967** | **0.065** |

**原差距的约 40% 来自模型差异，不是表示方式。** 差距集中在跨页三级表头（0.754 vs 0.875）和普通表格（0.875 vs 1.000），纯文字与多级表头两层完全打平。

n=15、单文档、31 页候选池——这个差距完全可能是噪声。**真实规模的数一个都还没有。**

## 三、已确认的机制（与原假设不同）

原假设是「表格被拆烂 → 关键词丢失 → 检索不到」。实测**关键词没丢**：`COD`、`氨氮` 在 50-60 页九个页面上都在。

真实机制是：**跨页表格让连续多页的文本高度雷同**，加上整张表被切成一个 2139 字符的大块，单向量被稀释，具体是哪家公司哪一行反而淹没了。视觉式赢在页面图保留了「这一页有三级表头」这个版面级信号。

由此导出消融组 **A′（表格按 `<tr>` 拆行）**：解析式的劣势有多少来自切块粒度而非解析本身，必须测出来，否则对解析式不公平。

## 四、三个会让结论失真的坑（都已修，但接手时要知道）

### 1. 模型混淆（已修）
两条链路一次变了两个东西——表示方式不同，编码模型也不同，视觉侧底座大 5 倍新 1 年。**修法**：改用 Qwen3-VL-Embedding-2B 的双模态输入，同一套权重编码文字和图片，唯一变量只剩表示形式。

### 2. 接地校验参照偏差（已修）
若用 DeepDoc 解析文本做答案校验参照，「图上清晰但被解析坏了」的页会被判未接地而丢弃——**恰好剔掉解析式最难的样本，等于给解析式放水**。**修法**：改用 pdfplumber 抽的 PDF 内嵌文本，独立于两条路线。

### 3. 查询集退化与真值不唯一（已修）
首轮生成 201 条，**110 条是退化样本**（query 与 answer 完全相同，模型把原文陈述句当查询）——这类样本天然利好 BM25，会把词法系统的分抬虚。另有一批查询不含公司名（「银行承兑票据期末终止确认金额」），40 家年报每家都有这行，正确召回会被判错。

**修法**：提示词强制提问句 + 强制带公司名 + 给出正反例；退化检查前置到生成时；`filter_queries.py` 做唯一性检查——答案串在多少页出现就把这些页全记为真值，超过 8 页的判为无区分度丢弃。

## 五、环境（与上一版的差异）

| 环境 | 用途 | 关键点 |
|---|---|---|
| `~/envs/deepdoc` | 解析链路 | **不装 torch**（用 stub），20 个依赖，ONNX Runtime CUDA EP |
| `~/envs/zhdocrag` | 检索与生成 | torch 2.7.1+cu126、transformers 5.17.0、sentence-transformers 6.1.0、colpali-engine 0.3.18、peft 0.20.0 |
| `~/envs/receiptvlm` | 旧环境 | 不要动 |

### DeepDoc 脱离 RAGFlow 的三处关键改造（见 `scripts/deepdoc_standalone.py`）

1. **stub 掉 `common.settings`** —— 该模块在顶层 import 了 RAGFlow 整个存储层（es/infinity/oceanbase/opensearch/gaussdb/azure/gcs/minio/opendal/redis/s3/oss）。但 deepdoc 只用到其中**一个变量** `PARALLEL_DEVICES`，外加 `DOC_ENGINE_INFINITY`。
2. **stub 掉 `torch`** —— DeepDoc 判断能否用 GPU 的唯一依据是 `import torch` 能否成功（`ocr.py:89-93`），之后直接用 onnxruntime 建 CUDA 会话，torch 再无用途。为一个布尔值装 4 GB 不值得。**副带解掉一颗雷**：`pip_install_torch()` 在 `DEVICE != "cpu"` 时会在运行时执行 `pip install "torch>=2.5.0,<3.0.0"`，没有 CUDA 构建上界，会拉 cu13x 的 wheel，本机驱动上限 12.4，装得上跑不了。
3. **ctypes 预加载 cuDNN/cuBLAS** —— `onnxruntime.get_available_providers()` 里有 CUDAExecutionProvider 只代表编译时支持；运行时找不到 `libcudnn.so.9` 会**静默回落 CPU**，只在 stderr 打一行 W 级日志。实测回落后 45s/页，正常 GPU 下 5.4s/页。

## 六、环境坑速查

| 坑 | 解法 |
|---|---|
| HF Xet 传输卡死（45 秒零字节，不报错） | `export HF_HUB_DISABLE_XET=1` |
| Docker Hub 直连不通 | `docker.m.daocloud.io/<原路径>` 再 retag |
| 根分区 100% 满 | **他人的 6 个 18GB 日志占 108GB**；所有任务须 `TMPDIR=/home1/jiajunjie/tmp`，否则 torch 建临时目录直接崩 |
| `colqwen2.5-v0.2` 用 repo id 加载崩 | 用本地快照路径（transformers 解析 `additional_chat_templates/` 返回 None） |
| `PIP_CONSTRAINT` 只锁了 torch | transformers 会被 sentence-transformers 顺带升级，连带 peft 需同步升级 |
| 用 `pgrep -f`/`pkill -f` 匹配进程 | **必须用 `[x]` 括号写法排除自身**，否则会匹配到自己：等待循环会死等，pkill 会自杀 |

## 七、恢复工作

```bash
cd ~/Projects/zh-docrag
export TMPDIR=/home1/jiajunjie/tmp        # 根分区已满，必须

# 1) 评测集（视觉源约 20 min，文本源约 20 min）
./scripts/gen_queries.py --source image --target 200 --max-pages 500
./scripts/gen_queries.py --source text  --target 200 --max-pages 500
python scripts/filter_queries.py          # 退化过滤 + 唯一性检查

# 2) 建五系统索引（E 多向量最重，约 8 min）
python scripts/build_indexes.py

# 3) 对比评测 + 配对自助法检验
python scripts/evaluate_all.py
```

语料本身不入库（174MB PDF + 589MB 页面图），用 `scripts/fetch_corpus.py` + `scripts/build_pages.py` 可从巨潮完整重建；随机种子已固定。

## 八、下一步

跑完评测后按 DESIGN.md 第七节的**终止条件**判断——那是写死的，不许事后放宽。特别是：**双源结果若方向相反，说明差距主要来自出题偏差，必须重做查询集，不许挑对自己有利的那组报。**
