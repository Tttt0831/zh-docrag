"""
HF CausalLM + LoRA 封装

将 HuggingFace Qwen2ForCausalLM 封装为 VLM 可用的 LLM 组件。
暴露与 MiniLLM 一致的接口：
  - forward_with_embeddings(embeddings, attention_mask) -> logits
  - embed_tokens(input_ids) -> embeddings
  - generate(inputs_embeds, ...) -> token_ids

用法：
    from src.model.llm_hf import build_hf_llm
    llm = build_hf_llm("Qwen/Qwen2-1.5B", lora=True)
    logits = llm.forward_with_embeddings(embeds, mask)
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class HFLlmConfig:
    """HF LLM 配置。"""
    model_name: str = "Qwen/Qwen2-1.5B"
    torch_dtype: str = "bfloat16"       # "bfloat16" | "float16" | "float32"
    device_map: str = "auto"
    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )


class HFLlamaWrapper(nn.Module):
    """
    封装 HuggingFace CausalLM，暴露与 MiniLLM 兼容的接口。

    关键方法:
      - embed_tokens(input_ids) → [B, L, H]
      - forward_with_embeddings(embeddings, attention_mask) → [B, L, V]
      - generate(inputs_embeds, attention_mask, ...) → [B, generated_len]
    """

    def __init__(self, config: HFLlmConfig):
        super().__init__()
        self.config = config

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(config.torch_dtype, torch.bfloat16)

        # 加载基座模型
        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch_dtype,
            device_map=config.device_map,
            trust_remote_code=True,
        )

        # LoRA
        self.lora = None
        if config.use_lora:
            self._apply_lora()

        # 缓存常用组件引用
        self._embed_tokens = self.model.get_input_embeddings()
        self._lm_head = self.model.get_output_embeddings()

        # vocab
        self.vocab_size = self.model.config.vocab_size

    def _apply_lora(self):
        from peft import get_peft_model, LoraConfig, TaskType

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=list(self.config.lora_target_modules),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_config)
        self.lora = lora_config
        print(f"✓ LoRA applied: r={self.config.lora_r}, "
              f"alpha={self.config.lora_alpha}, "
              f"targets={self.config.lora_target_modules}")

    # ------------------------------------------------------------------ #
    # 属性（兼容 MiniLLM 接口）
    # ------------------------------------------------------------------ #

    @property
    def embed_tokens(self):
        return self._embed_tokens

    @property
    def lm_head(self):
        return self._lm_head

    def _backbone(self):
        """
        取到真正带 .layers / .norm 的那层 backbone。

        层级会随是否启用 LoRA 而变：
          不加 LoRA: self.model = Qwen2ForCausalLM     -> .model 是 Qwen2Model
          加了 LoRA: self.model = PeftModelForCausalLM -> .model 是 Qwen2ForCausalLM
                                                       -> .model.model 才是 Qwen2Model
        原实现写死 self.model.model，加 LoRA 后就拿不到 layers，
        导致 freeze_strategy="projection_llm_ends" 在 HF+LoRA 下直接 AttributeError
        （这条路径之前从没被执行过）。这里逐层下钻，两种情况都成立。
        """
        m = self.model
        for _ in range(4):
            if hasattr(m, "layers") and hasattr(m, "norm"):
                return m
            if not hasattr(m, "model"):
                break
            m = m.model
        raise AttributeError(f"未能在 {type(self.model).__name__} 下找到带 layers/norm 的 backbone")

    @property
    def layers(self):
        """返回 HF 模型的 decoder layers（用于冻结策略）。"""
        return self._backbone().layers

    @property
    def ln_f(self):
        """返回 HF 模型的 final layernorm。"""
        return self._backbone().norm

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------ #
    # 核心接口
    # ------------------------------------------------------------------ #

    def forward_with_embeddings(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        使用 pre-computed embeddings 前向传播。
        跳过 embedding 层，直接喂入 transformer。

        Args:
            embeddings: [B, L, hidden_size]
            attention_mask: [B, L]

        Returns:
            logits: [B, L, vocab_size]
        """
        outputs = self.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return outputs.logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """标准前向（从 token ids）。"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return outputs.logits

    @torch.no_grad()
    def generate_from_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        eos_token_id: int = 5,
        pad_token_id: int = 0,
    ) -> torch.Tensor:
        """
        从 embeddings 开始自回归生成。

        Returns:
            generated_ids: [B, generated_len]
        """
        self.eval()
        device = next(self.parameters()).device

        # 移到正确设备
        if inputs_embeds.device != device:
            inputs_embeds = inputs_embeds.to(device)
        if attention_mask.device != device:
            attention_mask = attention_mask.to(device)

        generated = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            do_sample=temperature > 0,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            use_cache=True,
        )
        return generated


def build_hf_llm(model_name: str = "Qwen/Qwen2-1.5B",
                 use_lora: bool = True,
                 lora_r: int = 16,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.05,
                 torch_dtype: str = "bfloat16",
                 device_map: str = "auto") -> HFLlamaWrapper:
    """工厂函数：构建 HF LLM（可选 LoRA）。"""
    config = HFLlmConfig(
        model_name=model_name,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    return HFLlamaWrapper(config)


# ═══════════════════════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    print("=" * 50)
    print("Testing HFLlamaWrapper (offline smoke test)")
    print("=" * 50)

    # 不实际下载模型，只验证接口
    print("\n⚠️  在线测试需要网络加载 Qwen2-1.5B (~3GB)")
    print("如需离线测试，请使用: python -c 'from src.model.llm_hf import HFLlmConfig; print(HFLlmConfig())'")
    print("✓ llm_hf.py 语法正确，可导入")
