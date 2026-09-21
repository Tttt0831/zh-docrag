#!/bin/bash
# 路线C 自动化流水线：等预训练完成 → 评估 → VLM训练 → 评估
set -e
cd /root/autodl-tmp/receipt-vlm
mkdir -p logs checkpoints/route_c

LOG="logs/auto_pipeline.log"
echo "=== 路线C 自动化流水线 $(date) ===" | tee -a "$LOG"

# ── Step 0: 等待预训练完成 ──
PRETRAIN_LOG="logs/pretrain_full.log"
PRETRAIN_CKPT="checkpoints/route_c/llm_pretrained.pt"

echo "[0/4] 等待预训练完成..." | tee -a "$LOG"
while ! grep -q "预训练完成" "$PRETRAIN_LOG" 2>/dev/null; do
    sleep 30
done
echo "  ✓ 预训练完成 $(date)" | tee -a "$LOG"

# ── Step 1: 评估预训练模型质量 ──
echo "[1/4] 评估预训练模型生成质量..." | tee -a "$LOG"
python3 -c '
import torch, sys, json, re
sys.path.insert(0, "/root/autodl-tmp/receipt-vlm")
from src.model.llm import MiniLLM, LLMConfig
from transformers import AutoTokenizer
device = torch.device("cuda")
CKPT = "/root/autodl-tmp/receipt-vlm/checkpoints/route_c/llm_pretrained.pt"
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
cfg = LLMConfig(**{k:v for k,v in ckpt["llm_config"].items() if k in LLMConfig.__dataclass_fields__})
model = MiniLLM(cfg).to(device)
model.load_state_dict(ckpt["llm_state_dict"])
model.eval()
tok = AutoTokenizer.from_pretrained("/root/autodl-tmp/receipt-vlm/tokenizers/receipt-bpe", trust_remote_code=True)

results = {}
# Next-token accuracy
prompts = ["今天天气很好，","人工智能是","中国的首都是","最近，","由于经济","教育对于","根据相关法律法规，","机器学习是人工智能的"]
correct, total = 0, 0
with torch.no_grad():
    for p in prompts:
        ids = tok.encode(p, add_special_tokens=False)
        if len(ids) < 2: continue
        inp = torch.tensor([ids[:-1]], device=device)
        target = ids[1:]
        logits = model(inp)
        preds = logits[0].argmax(dim=-1)
        for pi, ti in zip(preds[-len(target):], target):
            correct += (pi == ti).item(); total += 1
results["next_token_acc"] = f"{100*correct/max(total,1):.1f}% ({correct}/{total})"

# Generation quality
test_prompts = ["今天天气真好，","人工智能的发展将","中国的四大发明包括","如何做好一顿美味的","昨天我去超市买了"]
gen_results = []
for prompt in test_prompts:
    ids = tok.encode(prompt, add_special_tokens=False)
    if not ids: continue
    inp = torch.tensor([ids], device=device)
    with torch.no_grad():
        for _ in range(60):
            logits = model(inp)
            nxt = torch.argmax(logits[0,-1,:]).item()
            inp = torch.cat([inp, torch.tensor([[nxt]], device=device)], dim=1)
    text = tok.decode(inp[0, len(ids):].tolist(), skip_special_tokens=True)
    for cut in ["\n","。",".","！","？"]:
        if cut in text: text = text[:text.index(cut)+1]; break
    gen_results.append({"prompt": prompt, "completion": text[:200]})
    # Check for repeats
    has_repeat = bool(re.search(r"(.)\1{15,}", text))
    gen_results[-1]["repeat"] = has_repeat

results["generations"] = gen_results
results["repeat_count"] = sum(1 for g in gen_results if g["repeat"])

import os
os.makedirs("/root/autodl-tmp/receipt-vlm/logs", exist_ok=True)
with open("/root/autodl-tmp/receipt-vlm/logs/pretrain_eval.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(json.dumps(results, ensure_ascii=False, indent=2))
' 2>&1 | tee -a "$LOG"
echo "  ✓ 预训练评估完成" | tee -a "$LOG"

# ── Step 2: VLM 训练 (Stage 1+2) ──
echo "[2/4] 启动 VLM 训练..." | tee -a "$LOG"
python -m src.train \
    --config configs/route_c.yaml \
    --init-llm checkpoints/route_c/llm_pretrained.pt \
    > logs/vlm_route_c.log 2>&1
echo "  ✓ VLM 训练完成 $(date)" | tee -a "$LOG"

# ── Step 3: VLM 评估 ──
echo "[3/4] 运行 VLM 评估..." | tee -a "$LOG"
python -m src.run_eval \
    --checkpoint checkpoints/route_c/best_model.pt \
    --data data/synthetic/test/test.jsonl \
    --tokenizer tokenizers/receipt-bpe \
    --output evaluation_results/route_c \
    2>&1 | tee -a "$LOG"
echo "  ✓ 评估完成" | tee -a "$LOG"

echo "=== 流水线全部完成 $(date) ===" | tee -a "$LOG"
