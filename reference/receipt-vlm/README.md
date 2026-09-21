# 轻量级中英文票据 VLM

> 从票据图片端到端抽取结构化字段（JSON），无需外部 OCR。围绕三条并行训练路线对比"自建轻量 VLM" vs "微调现成 VLM"。

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/pytorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 项目概述

从中英文票据/发票图片中直接输出 6 个关键字段的结构化 JSON。项目以**三条并行路线**对比不同的实现方式，使用同一套合成数据（3000/500/500）与同一套评估指标，便于横向比较。

抽取字段 schema：

```json
{
  "merchant_name": "string | null",
  "date":          "YYYY-MM-DD | null",
  "total_amount":  "number | null",
  "tax_amount":    "number | null",
  "tax_id":        "string | null",
  "invoice_no":    "string | null"
}
```

---

## 🧭 三条路线

| 路线 | 架构 | 可训练部分 | 状态 | F1 |
|------|------|-----------|------|:--:|
| **A** | SigLIP2 + MLP + **Qwen2-1.5B** LoRA | projection + LoRA + 解冻视觉顶层(消融) | ✅ | 15.3% (4层JFT) |
| **B** | **Qwen2-VL-2B-Instruct** + LoRA | LoRA 适配器 | ✅ | **58.5%** |
| **C** | SigLIP2 + MLP + **自制 MiniLLM 214M** | LM预训练 + projection + 端层 | ✅ | 45.8% |

三条路线均采用 **LLaVA 风格**多模态融合：文本序列里放 1 个 `<image>` 占位 token，前向时展开成 N 个视觉 patch embedding。

### 路线 A：SigLIP2 + Qwen2-1.5B（自建 VLM）
- 视觉：SigLIP2 NaFlex，动态分辨率
- 投影：2 层 MLP，`768 → 3072 → 1536`
- 语言：Qwen2-1.5B，LoRA(r=16) + 解冻 embedding/首末层
- 消融实验：冻结视觉(基线) → 解冻2层 → 解冻4层

### 路线 B：Qwen2-VL-2B + LoRA（微调现成 VLM）
- 直接在原生 Qwen2-VL-2B-Instruct 上加 LoRA(r=16)
- 训练/推理最简单，泛化最好，效果最优

### 路线 C：SigLIP2 + 自制 MiniLLM 214M（从零训练）
- 语言模型 `MiniLLM`：decoder-only，对齐 MiniMind 架构 —— RoPE + SDPA + RMSNorm + SwiGLU，`h1024/L16/heads16/inter2752 ≈ 214M`
- 自训练 ~12k BPE tokenizer（`tokenizers/receipt-bpe/`）
- **Stage 0**：MiniMind 中文语料全量语言预训练（~281M tokens/epoch × 2 epoch）
- **Stage 1+2**：VLM 对齐 + 票据精调（8 epoch）

---

## 📊 评估结果（500 合成测试集，同一指标口径）

指标定义见 [`src/eval.py`](src/eval.py)：字段 exact-match 准确率、Value Match、P/R/F1、1-NED、JSON 合法率、幻觉率。

### 五组实验完整对比

| 字段 | A(冻结) | A(2层JFT) | **A(4层JFT)** | 路线B | **路线C** |
|------|:------:|:---------:|:------------:|:-----:|:--------:|
| merchant_name | 2.80% | 38.40% | **41.00%** | 23.40% | **41.60%** |
| date | 0.00% | 0.40% | **0.60%** | 36.60% | **52.40%** |
| total_amount | 0.00% | 0.00% | 0.00% | **67.20%** | 43.80% |
| tax_amount | 20.60% | 20.60% | **21.60%** | **74.60%** | 41.20% |
| tax_id | 33.40% | 33.80% | **35.00%** | **41.60%** | 34.60% |
| invoice_no | 22.20% | 12.00% | **13.00%** | **41.60%** | 9.80% |

| 总体 | A(冻结) | A(2层JFT) | **A(4层JFT)** | 路线B | **路线C** |
|------|:------:|:---------:|:------------:|:-----:|:--------:|
| Value Match | 13.17% | 17.53% | **18.53%** | **47.50%** | 37.23% |
| Precision | 72.22% | 77.32% | 75.09% | **94.70%** | 89.95% |
| Recall | 0.99% | 7.94% | **8.49%** | **42.27%** | 30.76% |
| **F1** | 1.96% | 14.39% | **15.26%** | **58.45%** | 45.84% |
| 1-NED(merchant)↓ | 0.7969 | 0.4746 | **0.4493** | 0.6555 | **0.4435** |
| JSON 合法率 | 100% | 100% | 100% | 100% | 100% |
| 幻觉率 | 2.64% | 16.09% | 17.85% | 16.36% | 21.76% |

> 结果文件：`evaluation_results/route_a/` · `route_a_jointft/` · `route_a_jointft_4l/` · `route_b/` · `route_c/`

### 核心发现

1. **路线 B 总体最强**（F1 58.5%）：原生 VLM 的视觉理解能力无可替代，金额字段优势突出
2. **路线 C 性价比最优**：214M 小模型 F1 达 45.8%，超越路线 A 的 1.5B 参数方案；文本字段（merchant_name/date）甚至超过路线 B
3. **路线 A 离不开视觉解冻**：冻结 → 完全幻觉（F1 1.96%）；解冻 2 层 → 14.4%；4 层 → 15.3%
4. **路线 A 瓶颈不在视觉层数**：4 层 vs 2 层仅 +0.9pp F1，total_amount 始终 0%。瓶颈在 projection 容量和训练数据
5. **金额字段是路线 A/C 的共同短板**：路线 B 的 total_amount 67.2% vs 路线 C 43.8% vs 路线 A 0%
6. **路线 C 幻觉率偏高**（21.76%）：小模型更倾向"编造合理内容"

---

## 📈 训练曲线

### 路线 A 基线（冻结视觉，5 epoch）

| Epoch | Train Loss | Val Loss |
|:-----:|:----------:|:--------:|
| 1 | 1.0185 | 0.8931 |
| 2 | 0.8939 | 0.8820 |
| 3 | 0.8825 | 0.8763 |
| 4 | 0.8751 | 0.8729 |
| 5 | 0.8687 | 0.8727 |

低 loss ≠ 高准确率——缺乏视觉接地导致输出为幻觉。

### 路线 A 联合微调 2层（解冻视觉顶部 2 层，5 epoch）

| Epoch | Train Loss | Val Loss |
|:-----:|:----------:|:--------:|
| 1 | 1.0670 | 0.8944 |
| 2 | 0.8863 | 0.8704 |
| 3 | 0.8660 | 0.8517 |
| 4 | 0.8513 | 0.8471 |
| 5 | 0.8452 | **0.8469** |

解冻后 val loss（0.847）< 基线（0.873），merchant_name 2.8% → 38.4%。

### 路线 A 联合微调 4层（解冻视觉顶部 4 层，5 epoch，从 epoch 2 恢复）

| Epoch | Train Loss | Val Loss |
|:-----:|:----------:|:--------:|
| 1 | 1.0607 | 0.8897 |
| 2 | 0.8918 | 0.8656 |
| 3 | 0.8744 | 0.8584 |
| 4 | 0.8699 | **0.8574** |
| 5 | 0.8688 | 0.8576 |

4 层 val loss（0.8574）略高于 2 层（0.8469），纯看 loss 4 层略差，但抽取指标小幅提升（F1 14.4% → 15.3%）。边际收益递减。

### 路线 B（5 epoch, bs=2, grad_accum=8）

训练 5 epoch 后达到上表指标（per-epoch 曲线未单独留存）。

### 路线 C — Stage 0 语言预训练（~281M tokens/epoch × 2 epoch）

预训练数据：MiniMind `pretrain_t2t_mini` 全量 130 万行  
配置：`block_size=512, batch_size=24, lr=1e-4, warmup=1000`  
规模：22,856 step/epoch × 24 × 512 = **~281M tokens/epoch**，2 epoch ≈ **562M tokens 总训练量**

| Epoch | Steps | Start Loss (ppl) | End Loss (ppl) |
|:-----:|:-----:|:----------------:|:--------------:|
| 1 | 22,856 | 9.60 (14707) | ~3.8 |
| 2 | 22,856 | ~3.8 | **~2.08 (8.0)** |

训练耗时约 3.5 小时（RTX 4090D 24GB），无 NaN/发散。

**预训练后评估**（next-token 准确率 42.9%，14 token 中文样本）：

| Prompt | 模型续写 | 判断 |
|--------|----------|:--:|
| 今天天气真好， | "我想出去散步。" | ✅ |
| 人工智能的发展将 | "带来更多的机遇和挑战。" | ✅ |
| 中国的四大发明包括 | "造纸术、印刷术、指南针和火药。" | ✅ |
| 如何做好一顿美味的 | "午餐。" | 🟡 |
| 昨天我去超市买了 | "两瓶牛奶…" | ✅ |

无重复模式，语言能力初步建立。

### 路线 C — Stage 1+2 VLM 训练（8 epoch, 3000/500 合成票据）

| Epoch | Train Loss | Val Loss |
|:-----:|:----------:|:--------:|
| 1 | 1.5156 | 1.0205 |
| 2 | 1.0172 | 0.9826 |
| 3 | 0.9826 | 0.9548 |
| 4 | 0.9191 | 0.8848 |
| 5 | 0.7947 | 0.7424 |
| 6 | 0.6645 | 0.6807 |
| 7 | 0.5905 | **0.6763** |
| 8 | 0.5567 | 0.6789 |

最佳 val loss 0.6763（Epoch 7），远优于路线 A 基线 0.873。仅 214M 参数达到 F1=45.8%。

---

## 🗂️ 项目结构

```
receipt-vlm/
├── README.md                       # 本文档（唯一文档）
├── requirements.txt
├── configs/
│   ├── route_a.yaml                # 路线A: SigLIP2(冻结) + Qwen2-1.5B LoRA
│   ├── route_a_jointft.yaml         # 路线A消融: 解冻2层 + 联合微调
│   ├── route_a_jointft_4l.yaml      # 路线A消融: 解冻4层 + 联合微调
│   ├── route_b.yaml                 # 路线B: Qwen2-VL-2B LoRA
│   └── route_c.yaml                 # 路线C: SigLIP2 + 自制 MiniLLM 214M
├── src/
│   ├── train.py                    # 路线 A/C 训练（--resume 恢复, --init-llm 载入预训练LLM）
│   ├── train_qwen2vl_lora.py       # 路线 B 训练
│   ├── pretrain_lm.py              # 路线 C: MiniLLM 语言预训练
│   ├── train_tokenizer.py          # 路线 C: BPE tokenizer 训练
│   ├── run_eval.py / eval.py       # 推理评估 + 指标计算
│   ├── infer.py                    # 推理引擎（自动识别路线 A/B/C）
│   ├── model/                      # vision / projection / llm / vlm / llm_hf / tokenizer
│   ├── data/                       # synth（合成票据）/ dataset
│   └── utils/                      # normalize / preprocessing
├── tokenizers/receipt-bpe/         # 路线C 自训练 BPE（已入库）
└── evaluation_results/             # 五组评估指标（已入库）
    ├── route_a/ route_a_jointft/ route_a_jointft_4l/
    ├── route_b/
    └── route_c/
```

> 数据集与 checkpoint 不入库（`.gitignore`）：`data/`、`checkpoints/`、`*.pt`、`*.log`。

---

## 🚀 快速开始

```bash
pip install -r requirements.txt

# 0) 生成合成数据（3000/500/500）
python -m src.data.synth

# 路线 A：训练（含消融变体）+ 评估
python -m src.train --config configs/route_a.yaml --epochs 5 --batch-size 2
python -m src.train --config configs/route_a_jointft.yaml --epochs 5 --batch-size 2
python -m src.train --config configs/route_a_jointft_4l.yaml --epochs 5 --batch-size 2
python -m src.run_eval --checkpoint checkpoints/route_a/best_model.pt \
    --data data/synthetic/test/test.jsonl --tokenizer Qwen/Qwen2-1.5B \
    --output evaluation_results/route_a

# 路线 B：训练 + 评估
python src/train_qwen2vl_lora.py --config configs/route_b.yaml
python -m src.run_eval --checkpoint checkpoints/route_b/best_lora \
    --data data/synthetic/test/test.jsonl --output evaluation_results/route_b

# 路线 C：① 训 tokenizer ② 语言预训练 ③ VLM 训练
python -m src.train_tokenizer --vocab-size 12000
python -m src.pretrain_lm --max-lines 1300000 --epochs 2 --block-size 512 --batch-size 24
python -m src.train --config configs/route_c.yaml \
    --init-llm checkpoints/route_c/llm_pretrained.pt
```

环境：Python ≥ 3.10，PyTorch 2.x（bf16），单卡 RTX 4090 24GB 即可。

---

## 🧪 评估指标体系

| 指标 | 说明 |
|------|------|
| Field Accuracy | 6 字段各自的 exact-match 准确率 |
| Value Match Rate | 6 字段同时完全匹配的比例 |
| Precision / Recall / F1 | 字段级，反映漏抽 vs 幻觉 |
| 1-NED | 归一化编辑距离（merchant_name），越低越好 |
| JSON Validity Rate | 输出为合法 JSON 的比例 |
| Hallucination Rate | 真值为 null 却被模型填值的比例 |

归一化（[`src/utils/normalize.py`](src/utils/normalize.py)）：日期 → `YYYY-MM-DD`；金额去符号/千分位；字符串 strip + 全角转半角。

---

## 🙏 致谢

- [SigLIP](https://github.com/google-research/big_vision) — 视觉编码器
- [MiniMind](https://github.com/jingyaogong/minimind) — 自制 LLM 架构参考 + 语言预训练语料
- [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) — 路线 B 基座
- [LLaVA](https://github.com/haotian-liu/LLaVA) — Projection 融合设计

## 📄 许可证

MIT License
