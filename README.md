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

阶段 0 评测完成（2026-09-29，483 条双源查询 × 2400 页，已修正真值并完成抽检）。

| 结论 | 数字 |
|---|---|
| 同一模型下，页面图比解析文本好 | +0.142 nDCG@5，p < 0.001，两个出题源方向一致 |
| 优势集中在复杂表格页 | +0.154（pdfplumber 口径，n=268） |
| 纯文字、普通表格页两者打平 | pdfplumber 口径下均为 ±0.002，不显著 |
| 预先登记的「最强对最强」（ColQwen 多向量 vs 解析式混合） | +0.041，p = 0.042，**只在图像源显著** |
| 查询集残余噪声 | 按答题约 35%（出题模型读错表格行列），按找页约 8%–12% |

原理与完整结果见 [PRINCIPLES.md](PRINCIPLES.md)。

## 快速开始

```bash
./scripts/fetch_models.sh      # 下模型（注意：必须 HF_HUB_DISABLE_XET=1，脚本里已设）
./scripts/setup_venv.sh        # 建 Python 环境（torch 锁 cu126，见 constraints.txt）
./scripts/clone_ragflow.sh     # 拉 RAGFlow 源码
./preflight.sh                 # 六项自检
./ragflow-up.sh up -d          # 起 RAGFlow
```
