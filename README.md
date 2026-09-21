# zh-docrag

中文视觉文档 RAG：**解析式（RAGFlow / MinerU）和视觉式（ColQwen / Qwen3-VL-Embedding），在中文文档上到底哪个好、在什么类型的文档上好。**

这个问题没人回答过——ViDoRe、VisR-Bench、ColPali 训练集、vdr-multilingual 全都没有中文（六源交叉验证见 [plan.md](plan.md)），
而每个做中文文档 RAG 的团队都要做这个选择。

## 从哪读起

| 文档 | 内容 |
|---|---|
| [plan.md](plan.md) | 研究计划：要回答什么、已核实的事实、阶段 0/1/2、**写死的终止条件** |
| [HANDOFF.md](HANDOFF.md) | 环境现状：进度快照、五个已确认的坑、当前阻塞、恢复命令 |

## 现在处于哪一步

阶段 0 未开始，正在搭环境。**有一个硬阻塞（根分区满导致 RAGFlow 镜像拉不下来），见 HANDOFF.md 第四节。**

## 快速开始

```bash
./scripts/fetch_models.sh      # 下模型（注意：必须 HF_HUB_DISABLE_XET=1，脚本里已设）
./scripts/setup_venv.sh        # 建 Python 环境（torch 锁 cu126，见 constraints.txt）
./scripts/clone_ragflow.sh     # 拉 RAGFlow 源码
./preflight.sh                 # 六项自检
./ragflow-up.sh up -d          # 起 RAGFlow
```
