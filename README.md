# zh-docrag

中文视觉文档 RAG：**解析式（RAGFlow / MinerU）和视觉式（ColQwen / Qwen3-VL-Embedding），在中文文档上到底哪个好、在什么类型的文档上好。**

这个问题没人回答过——ViDoRe、VisR-Bench、ColPali 训练集、vdr-multilingual 全都没有中文（六源交叉验证见 [plan.md](plan.md)），
而每个做中文文档 RAG 的团队都要做这个选择。

## 从哪读起

| 文档 | 内容 |
|---|---|
| [PRINCIPLES.md](PRINCIPLES.md) | **原理**：两条路线怎么工作、比较怎么做到公平、首批结果说明了什么 |
| [plan.md](plan.md) | 研究计划：要回答什么、已核实的事实、阶段 0/1/2、**写死的终止条件** |
| [HANDOFF.md](HANDOFF.md) | 环境现状：进度快照、五个已确认的坑、当前阻塞、恢复命令 |

## 现在处于哪一步

阶段 0 的首批真实规模评测已完成（2026-09-29，440 条查询 × 2400 页），**正在做查询集人工抽检**，抽检后数字可能修正。

| 结论 | 数字 |
|---|---|
| 同一模型下，页面图比解析文本好 | +0.104 nDCG@5，p < 0.001，两个出题源方向一致 |
| 优势集中在复杂表格页 | +0.126（pdfplumber 口径，n=239） |
| 纯文字页两者持平 | 两套分层口径下均不显著 |
| 预先登记的「最强对最强」（ColQwen 多向量 vs 解析式混合） | +0.026，**不显著** |

原理与完整结果见 [PRINCIPLES.md](PRINCIPLES.md)。

## 快速开始

```bash
./scripts/fetch_models.sh      # 下模型（注意：必须 HF_HUB_DISABLE_XET=1，脚本里已设）
./scripts/setup_venv.sh        # 建 Python 环境（torch 锁 cu126，见 constraints.txt）
./scripts/clone_ragflow.sh     # 拉 RAGFlow 源码
./preflight.sh                 # 六项自检
./ragflow-up.sh up -d          # 起 RAGFlow
```
