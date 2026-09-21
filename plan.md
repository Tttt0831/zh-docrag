# 中文视觉文档 RAG — 计划与交接文档

> 状态：**阶段 0 未开始**。文档更新于 2026-09-21。
> 先读第二节「已核实的事实」——那是唯一不用重新验证的部分，其余都是待办。
> **前身 `Projects/receipt-vlm` 已删除**（见第五节），可复用代码留在 `reference/receipt-vlm/`。

---

## 一、这个项目回答什么问题

> **中文文档 RAG，解析式（RAGFlow/MinerU）和视觉式（ColQwen），到底哪个好？
> 在什么类型的文档上好？**

这个问题没人回答过（第二节有六源交叉验证），而每个做中文文档 RAG 的团队都要做这个选择。

- **解析式**：PDF → OCR → 版面分析 → 转 Markdown → 文本切块 → 文本检索。RAGFlow/MinerU 走这条。
- **视觉式**：页面直接当图片编码，多向量后期交互（MaxSim）检索，不做 OCR。ColPali/ColQwen 走这条。

理由见 `assets/why-visual.png`（中粮生物 2023 年报第 60 页）：跨页多级表头、合并单元格、
竖排表头。解析式流水线会把这种表拆烂；视觉式绕过这个问题，但代价是每页约 1000 个向量。
**哪种代价更划算，在中文上是未知数。**

### 重要：不要一上来就训模型

初版计划写的是「训一个中文 ColQwen」，**那个前提站不住**——阿里云已经开源了
`Ops-Colqwen3-4B`（基座 Qwen3-VL-4B，ViDoRe v1/v2/v3 全 SOTA），中文能力大概率够用，
个人算力训不过它。

正确顺序是**测量先行**：先测清楚两条路线在中文上的实际表现，测出缺口再决定训不训。
这样无论结果朝哪边倒都有产出。

---

## 二、已核实的事实

**每条都实际查过，接手时可直接采信。核查日期 2026-09-21。**

### 2.1 中文视觉文档检索的数据与评测是空白（六源交叉验证）

| 检查项 | 结论 | 来源 |
|---|---|---|
| **VisR-Bench** | 16 语种、1,286 文档、**35,571 个问题**（Adobe Research, 2025-08）。15 个非英语语种是西、意、德、法、荷、阿拉伯、克罗地亚、**日语**、瑞典、越南、葡、芬兰、捷克、斯洛文尼亚、丹麦——**没有中文**。全文 "Chinese" 只出现 1 次，还是参考文献里的 C-Pack。日语都收了，说明不是 CJK 的技术问题，是他们爬的语料里中文文档不足 500 份 | [arXiv 2508.07493](https://arxiv.org/pdf/2508.07493) |
| ColPali 训练集 | 127,460 对，**纯英文 by design**（作者刻意留空非英语以研究零样本迁移）。63% 来自 DocVQA/InfoVQA/TAT-DQA/arXivQA，**37% 是爬 PDF + VLM 生成伪问题** | [HF blog](https://huggingface.co/blog/manu/colpali) |
| `llamaindex/vdr-multilingual-train` | 语言标签 `de, it, fr, es, en`——无中文 | HF API |
| `tsystems/colqwen2.5-3b-multilingual-v1.0` | 训练源无中文；**未报任何评测分数**；自述「聚焦高资源语言，泛化受限」 | 模型卡 |
| `nvidia/miracl-vision` | 18 语种含中文，但中文子集只有 **189 条查询 / 8,672 篇文档**，且是 Wikipedia 网页截图（Playwright 截前 2048px 裁到 980×980）——纯文本渲染，无表格图表。既不够训练也不代表真实中文版式 | [arXiv 2505.11651](https://arxiv.org/html/2505.11651) |
| ViDoRe V2 | 覆盖 EN/ES/FR/DE，无中文 | [arXiv 2505.17166](https://arxiv.org/pdf/2505.17166) |
| HF 中文文档数据集 | 只有 `CDLA`(dl=178)、`M6Doc`(dl=512) 等**版面分析**集——标的是框不是查询，下载量三位数 | HF API 多关键词搜索 |

### 2.2 成熟方案盘点：成熟的全是解析式

| 方案 | 可用性（已核实 2026-09-21） | 路线 | 能否当基线 |
|---|---|---|---|
| **RAGFlow** (infiniflow) | Apache-2.0，**91,092 星**，2026-09-21 当天 5 次提交，48 个版本 | **解析式**。README 里 `colpali`/`visual retrieval`/`multimodal`/`VLM` **一个都没有**。核心是 DeepDoc：OCR + 版面 + 模板切块。VLM 只用来给图片配说明（见 v0.27.1「OCR text lost when no image2text model is configured」） | ✅ **唯一活着的自托管解析式基线，用它** |
| MinerU (OpenDataLab) | Apache-2.0 + 附加条款，80,378 星，2026-09-21 在提交 | 解析式，转 Markdown/JSON/LaTeX | ⚠️ 只是解析器不是完整 RAG，且**已内置为 RAGFlow 的 PDF 解析器**，测 RAGFlow 即覆盖 |
| QAnything (网易有道) | AGPL-3.0，14,184 星。**真实开发停在 2025-03-12**；2026 年仅两次提交，都是往 README 加商务联系方式 | 解析式 | ❌ **实质停更 18 个月，不要用** |
| TextIn (合合信息) | 商业 API，新用户 **1000 次免费额度**，私有化部署需企业采购 | 解析式 | ⚠️ 只提供**解析这一步**，不含检索——要当基线得自己接切块+向量化+检索。可作「最好的商业解析器下，解析式的上限」抽样对照 |

**结论：中文文档 RAG 的成熟方案，没有一个在用视觉检索。**
**基线选型：RAGFlow 是唯一真正可用的对照组。**

### 2.3 视觉检索模型：成熟，但都没测过中文

| 模型 | 更新 | 下载 | 大小 | 形态 |
|---|---|---|---|---|
| `OpenSearch-AI/Ops-Colqwen3-4B` | 2026-01 | 6.7k | — | **阿里云 OpenSearch 团队**，ColPali 架构 + MaxSim，基座 Qwen3-VL-4B，**ViDoRe v1/v2/v3 全 SOTA** |
| `Qwen/Qwen3-VL-Embedding-8B` | 2026-04 | 117 万 | 16.3GB | **单向量稠密**，ViDoRe 84.4~87.2，MMEB-V2 77.8，有配套 reranker |
| `Qwen/Qwen3-VL-Embedding-2B` | 2026-04 | 112 万 | 4.3GB | 单向量稠密，MMEB-V2 73.2 |
| `vidore/colqwen2.5-v0.2` | 2026-08 | 32 万 | 0.3GB | 多向量，**LoRA on Qwen2.5-VL-3B** |
| `vidore/colqwen2-v1.0` | 2026-08 | 39 万 | 0.2GB | 多向量，LoRA(r=32) on Qwen2-VL-2B |
| `athrael-soju/colqwen3.5-4.5B-v3` | 2026-08 | 38 万 | 27.3GB | 多向量，完整权重 |

要点：
- ColQwen 系发布物只有 0.2~0.3GB，因为**是 LoRA adapter + 128 维投影头**，基座单独下。
- Ops-Colqwen3-4B 和 Qwen3-VL-Embedding **都声称 30+ 语种，都只在 ViDoRe（欧洲语种）上报分**。
  「支持」和「测过」是两回事——这正是本项目的切入点。
- ColPali 架构细节：SigLIP-So400m，448×448 输入切 32×32 = 1,024 个 patch，各投影到 128 维。

### 2.4 不要做文档解析成 Markdown

[OmniDocBench 已饱和](https://www.llamaindex.ai/blog/omnidocbench-is-saturated-what-s-next-for-ocr-benchmarks)。
那条路会精确复现 receipt-vlm 的失败模式：指标冲高但涨幅全来自工程修补。

### 2.5 数据链路已端到端验证

| 环节 | 验证结果 |
|---|---|
| 巨潮查询接口 | `POST http://www.cninfo.com.cn/new/hisAnnouncement/query` 可用。仅深市 2024 年报 `category_ndbg_szsh` 就 **11,427 份** |
| PDF 下载 | `http://static.cninfo.com.cn/<adjunctUrl>` 可直取。实测 4.2MB / 243 页 |
| 渲染 | `pdftoppm -png -r 110` 实测 **5 页 1.52 秒**。横纵混排（宽表转横向页），符合真实版式 |
| 网络 | cninfo / gov.cn / stats.gov.cn / arxiv / huggingface 全部直连，**无需代理** |

合规：巨潮是证监会指定的信息披露平台，年报是面向投资者的公开文件。
但仍应**限速抓取**，量控制在项目所需（~200~1000 份）。

**备选数据源**：VisR-Bench 的多语种部分是从 **CCpdf**（Common Crawl 的 PDF 语料）筛出来的
——他们按「该语种文档数 > 500」筛，中文没进去。值得去查 CCpdf 里中文 PDF 到底有多少：
若够用，可省掉自己爬；若确实不足 500，那本身就是「中文缺口」的又一个佐证。**未验证，待办。**

### 2.6 环境

```
python 3.12.6   torch 2.7.1+cu126   transformers 4.53.2   peft 0.19.1
accelerate 1.13.0   datasets 5.0.1
未装：vllm  flash_attn  faiss  sentence_transformers  bitsandbytes
GPU：4 × RTX 4090D 24GB   磁盘：/home1 剩约 8.4T
```

**三个环境坑（过往教训，务必遵守）**：
1. 驱动 CUDA 上限 12.4。`pip install` 任何 `torch>=x` 无上界的包都可能把 torch 换成
   cu130、毁掉环境。**装 vllm / flash-attn 必须用 `PIP_CONSTRAINT` 锁死 torch，
   装完立刻 `python -c "import torch;print(torch.__version__)"` 自检。**
2. 4 卡 P2P 关闭、all-reduce 约 8GB/s。**LoRA DDP 可行，全参微调 / FSDP 不可行**。
   跑 DDP 需 `NCCL_P2P_DISABLE=1`。
3. `transformers 4.53.2` 大概率不支持 Qwen3-VL 系列，升级时同样要锁 torch。

---

## 三、阶段计划

### 阶段 0 — 建评测集 + 测量（约 1 周）★ 下一步

**不训模型。** 目标是一张 **路线 × 文档类型** 的表。

**0a 建中文评测集**（这是项目的核心资产）
- 巨潮抓 ~200 份年报 → 渲染页面图
- 按文档难度分层：纯文字页 / 普通表格 / **跨页多级表头大表** / 图表页
- 每页生成查询，人工核验 ~300~500 条。数据量不大，质量要硬
- 方法论直接移植 `reference/receipt-vlm/scripts/ds_annotate.py`：
  **门控**（只对页面上确实有可问内容的页生成）→ **接地校验**（查询的依据必须能在页面上定位）
- 教训：LLM 会**推断而非抽取**（票面只印「山东高速」，它补成「山东高速集团有限公司」），
  实测约 12% 的输出在原文里找不到依据，**必须过滤**

**0b 三方对测**

| 系统 | 类型 |
|---|---|
| RAGFlow（本地部署，DeepDoc 解析） | 解析式基线（**唯一真正活跃的**） |
| `Ops-Colqwen3-4B` / `colqwen2.5-v0.2` | 视觉式·多向量 |
| `Qwen3-VL-Embedding-2B` | 视觉式·单向量 |
| TextIn API（可选，1000 次免费） | 解析式上限对照·仅抽样 |

指标：nDCG@5、Recall@1，**按文档类型分组报**。
另记：索引体积、建索引耗时、检索延迟——多向量的代价要量化，不能只报精度。

**这一步的价值在于：无论结果朝哪边倒都有产出。**
- 视觉式在复杂表格上明显赢 → 阶段 1 有了明确动机
- 两者相当 → 结论本身就有价值（省下所有人的试错），项目转为「中文文档 RAG 选型实证」
- 解析式全面赢 → 说明中文场景下视觉路线不成立，这也是结论

> 这是 receipt-vlm 该做没做的事：**先做能证伪自己的检查**，而不是先干活再找理由。

### 阶段 1 — 只在阶段 0 测出缺口后才做（约 2 周）

若视觉式在某类中文文档上落后于它在英文上的表现，说明缺的是中文训练数据：

1. 扩大抓取 → 本地 Qwen2.5-VL-7B 生成伪查询（**不要用 API**：100k 次 DeepSeek 调用约 ¥1200，
   本地跑只花电费。这也正是 ColPali 那 37% 合成数据的做法）
2. `colqwen2.5-v0.2` 基座 + LoRA 微调，四组消融：

| 训练数据 | 想回答的问题 |
|---|---|
| 零样本 | 基线 |
| 仅英文 ColPali 集 | 跨语种迁移能带来多少 |
| 仅中文合成 | 合成数据本身的价值 |
| 中英混合 | 是否互补 |

### 阶段 2 — 端到端 + 溯源（约 1 周）

检索 top-k 页 → VLM 作答 + 引用页码与区域。核心是**归因分析**：
答错是检索没召回，还是召回了但生成错。这个拆解是文档 RAG 岗位的日常，
也最不容易靠「修个 bug 涨 40 个点」糊弄过去。

---

## 四、可复用资产（`reference/receipt-vlm/`，40 个 .py，约 1MB）

前身项目已删除，只留了源码。**注意：里面每个文件都踩过坑并修好了，改造时别退回去。**

| 文件 | 行数 | 怎么用 / 里面修过什么 |
|---|---|---|
| `src/train_qwen2vl_lora.py` | 484 | 改造成 ColQwen 训练器的骨架。**已修**：左 padding 时 label mask 的 prompt 起点算错 |
| `src/model/vision.py` | 240 | SigLIP2 封装。**已修**：内层 `Siglip2VisionTransformer` 参数名是 `attention_mask` 不是 `pixel_attention_mask`，用签名内省兼容 |
| `src/model/projection.py` | — | 投影头 + LayerNorm + 可学习 scale。**已诊断**：视觉 embedding 范数是文本的 10 倍，不归一化会导致 loss 好看但检索/抽取全错。ColPali 的 128 维投影头是同类问题 |
| `src/model/llm_hf.py` | 265 | **已修**：PEFT 会包一层 wrapper，取 `.layers`/`.norm` 要向下钻 |
| `src/eval.py` | 288 | **已修一个严重 bug**：抽错的值原先只记 FN 不记 FP，导致精确率恒等于 100%。写检索指标时同样要做四象限自检 |
| `src/pretrain_lm.py` | 301 | **已修**：DDP teardown 死锁（只有 rank 0 调用含 all_reduce 的 evaluate） |
| `scripts/ds_annotate.py` | 275 | **LLM 伪标注的方法论模板**：分层门控、接地校验、成本核算、空响应识别（reasoning 模型 content 为空但 HTTP 200） |
| `scripts/scid_finalize.py` | — | 接地过滤实现参考（最佳子串相似度 < 0.7 判为无依据） |
| `src/data/synth.py` | 2383 | 合成文档生成器（票据向）。补长尾版式可参考，不直接用 |

---

## 五、前身项目 receipt-vlm

**已于 2026-09-21 删除**（147G：129G 数据 + 18G 权重）。用户明确决定不再找回。

- 提交历史在 GitHub `https://github.com/Tttt0831/receipt-vlm.git`，最后提交 `ad797e6`，
  分支 `feat/synth-realism-upgrade`。**13 个文件的未提交改动和 19 个未跟踪脚本没有推上去**，
  但源码已复制到 `reference/receipt-vlm/`。
- 一并删除的还有：SCID 数据集（15G，需申请获取）、自预训练的 437M 中文 LM
  （`route_c_437m`，15G，约 2 天训练）、111G 预训练语料、4.3G 合成票据。
- **它为什么失败**（这是最该带走的教训）：
  - 指标从 58% 冲到 99%，涨幅**全部来自修 bug**，任务本身早就饱和
  - SCID 4 万条真实数据，六字段齐全的只有 **0.7%**——而这是票种结构决定的，不是标注不足，
    再花钱补标也救不回来
  - 根因：**选了一个会饱和的窄任务，且没在投入前验证数据可用性**

### 安全

DeepSeek API key 曾以明文出现在对话中，**应去后台轮换**。
本项目一律走环境变量 `DEEPSEEK_API_KEY`，不写进任何文件、脚本或日志。

---

## 六、待决问题

1. 阶段 0 评测集多大才够判断？初定 300~500 条人工核验查询，可能偏少。
2. 多向量的存储方案未定（faiss 不直接支持 MaxSim，可能要自己实现或用 PLAID 类方案）。
3. RAGFlow 本地部署要 Docker，资源占用和部署成本未评估。
4. `Ops-Colqwen3-4B` 用 Qwen3-VL 基座，需要升 transformers——与坑 #3 冲突，先在独立 venv 验证。

---

## 附：复现本文档中的验证

```bash
# 巨潮接口（应返回 11427）
python -c "
import urllib.request,urllib.parse,json
d=urllib.parse.urlencode({'pageNum':1,'pageSize':10,'column':'szse','tabName':'fulltext',
 'category':'category_ndbg_szsh','seDate':'2024-01-01~2024-12-31','isHLtitle':'true'}).encode()
r=urllib.request.Request('http://www.cninfo.com.cn/new/hisAnnouncement/query',data=d,
 headers={'User-Agent':'Mozilla/5.0','Content-Type':'application/x-www-form-urlencoded'})
print(json.load(urllib.request.urlopen(r,timeout=30))['totalAnnouncement'])"

# PDF → 页面图
curl -A "Mozilla/5.0" -o t.pdf http://static.cninfo.com.cn/finalpage/2024-12-26/1222143681.PDF
pdftoppm -png -r 110 -f 60 -l 64 t.pdf pg

# VisR-Bench 语种（确认无中文）
curl -sL -o visr.pdf https://arxiv.org/pdf/2508.07493
pdftotext visr.pdf - | sed -n '/15 non-English languages/,+4p'
pdftotext visr.pdf - | grep -ci chinese     # 应为 1，且是参考文献

# RAGFlow 是否支持视觉检索（应无输出）
curl -sL https://raw.githubusercontent.com/infiniflow/ragflow/main/README.md \
  | grep -iE "colpali|visual retriev|multimodal|VLM"
```
