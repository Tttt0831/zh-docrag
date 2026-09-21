"""
VLM 模型 - 组装 Vision + Projection + LLM

多模态融合采用 LLaVA 风格：文本序列里只放 1 个 <image> 占位 token，
在 forward 时把它**展开**成 N 个视觉 patch embedding（而非就地覆盖后续文本），
并同步展开 labels / attention_mask。
"""
import json
import torch
import torch.nn as nn
from typing import Optional, Union, Dict, Any, List
from dataclasses import dataclass

from .vision import SigLIP2VisionEncoder, VisionOutput
from .projection import MLPProjection
from .llm import MiniLLM, LLMConfig


@dataclass
class VLMConfig:
    """VLM 配置"""
    # Vision
    vision_model_name: str = "google/siglip2-base-patch16-naflex"
    vision_hidden_size: int = 768
    freeze_vision: bool = True
    vision_trainable_layers: int = 0   # >0 时解冻 vision 顶部若干层（联合微调）
    max_num_patches: int = 256

    # Projection
    projection_intermediate_dim: int = 2048
    projection_out_norm: bool = True          # projection 输出做 LayerNorm+缩放
    projection_target_row_norm: float = 0.65  # 目标行范数（对齐文本 embedding）
    projection_pool: int = 1                  # patch 序列平均池化倍数（pool²）
    projection_dropout: float = 0.0

    # LLM
    llm_type: str = "mini"           # "mini" | "hf" | "hf_lora"
    llm_vocab_size: int = 32000
    llm_hidden_size: int = 512
    llm_num_layers: int = 6
    llm_num_heads: int = 8
    llm_intermediate_size: int = 2048
    # HF LLM 专用
    hf_model_name: str = "Qwen/Qwen2-1.5B"
    hf_lora_r: int = 16
    hf_lora_alpha: int = 32
    hf_lora_dropout: float = 0.05

    # Training
    max_sequence_length: int = 1024
    image_token_sequence_length: int = 576  # 仅用于 fallback 估计
    gradient_checkpointing: bool = False    # 梯度检查点以节省显存

    # 冻结策略: "projection_only" | "projection_llm_ends"
    freeze_strategy: str = "projection_llm_ends"

    # 特殊 token id（运行时由 tokenizer 覆盖）
    pad_token_id: int = 0
    image_token_id: int = 3
    boa_token_id: int = 4
    eoa_token_id: int = 5


class ReceiptVLM(nn.Module):
    """轻量级票据 VLM: Vision Encoder -> Projection -> LLM -> JSON。"""

    def __init__(self, config: VLMConfig):
        super().__init__()
        self.config = config

        # 1. Vision Encoder（冻结）
        self.vision_encoder = SigLIP2VisionEncoder(
            model_name=config.vision_model_name,
            freeze=config.freeze_vision,
            max_num_patches=config.max_num_patches,
        )

        # 2. Projection（可训练）
        self.projection = MLPProjection(
            vision_dim=self.vision_encoder.hidden_size,
            llm_hidden_dim=config.llm_hidden_size,
            intermediate_dim=config.projection_intermediate_dim,
            out_norm=getattr(config, "projection_out_norm", True),
            target_row_norm=getattr(config, "projection_target_row_norm", 0.65),
            pool=getattr(config, "projection_pool", 1),
            dropout=config.projection_dropout,
        )

        # 3. LLM — 支持 mini / hf / hf_lora
        self._hf_llm = None  # 标记是否使用 HF LLM
        if config.llm_type in ("hf", "hf_lora"):
            from .llm_hf import HFLlmConfig, HFLlamaWrapper
            hf_cfg = HFLlmConfig(
                model_name=config.hf_model_name,
                use_lora=(config.llm_type == "hf_lora"),
                lora_r=config.hf_lora_r,
                lora_alpha=config.hf_lora_alpha,
                lora_dropout=config.hf_lora_dropout,
            )
            self.llm = HFLlamaWrapper(hf_cfg)
            self._hf_llm = True
        else:
            llm_config = LLMConfig(
                vocab_size=config.llm_vocab_size,
                hidden_size=config.llm_hidden_size,
                num_layers=config.llm_num_layers,
                num_heads=config.llm_num_heads,
                intermediate_size=config.llm_intermediate_size,
                max_position_embeddings=config.max_sequence_length,
            )
            self.llm = MiniLLM(llm_config, gradient_checkpointing=config.gradient_checkpointing)

        # 冻结策略
        self.set_trainable(config.freeze_strategy)

        # 特殊 token id
        self.pad_token_id = config.pad_token_id
        self.image_token_id = config.image_token_id
        self.boa_token_id = config.boa_token_id
        self.eoa_token_id = config.eoa_token_id

    # ------------------------------------------------------------------ #
    # 冻结策略
    # ------------------------------------------------------------------ #
    def set_trainable(self, strategy: str = "projection_llm_ends"):
        """
        设置可训练参数。Vision 始终冻结（在 encoder 内完成），Projection 始终可训练。

        - "projection_only": 只训练 projection。
        - "projection_llm_ends": projection + LLM(embedding + 首层 + 末层 + ln_f + lm_head)。
          对 HF+LoRA 模式，"projection_only" 只训练 projection+LoRA 适配器；
          "projection_llm_ends" 额外解冻 embedding 和首末层。
        """
        # projection 恒可训练
        for p in self.projection.parameters():
            p.requires_grad = True

        # 视觉部分解冻（顶部若干层）——用于联合微调增强视觉接地
        n_vis = getattr(self.config, "vision_trainable_layers", 0)
        if n_vis and n_vis > 0:
            n = self.vision_encoder.set_top_trainable(n_vis)
            print(f"✓ 视觉部分解冻: 顶部 {n_vis} 层 + post_layernorm，可训练 {n/1e6:.2f}M")

        if self._hf_llm:
            # HF LLM: LoRA 适配器已由 PEFT 设为 trainable
            # 先全冻结基座
            for p in self.llm.model.parameters():
                p.requires_grad = False
            # 重新启用 PEFT 适配器（它们被标记为 trainable）
            for n, p in self.llm.model.named_parameters():
                if "lora_" in n:
                    p.requires_grad = True

        else:
            # MiniLLM: 先全冻结
            for p in self.llm.parameters():
                p.requires_grad = False

        if strategy == "projection_only":
            return

        if strategy == "projection_llm_ends":
            if self._hf_llm:
                # 解冻 embedding + first/last layer + norm + lm_head
                for p in self.llm.model.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.llm.layers[0].parameters():
                    p.requires_grad = True
                for p in self.llm.layers[-1].parameters():
                    p.requires_grad = True
                for p in self.llm.ln_f.parameters():
                    p.requires_grad = True
                for p in self.llm.model.get_output_embeddings().parameters():
                    p.requires_grad = True
            else:
                # MiniLLM
                for p in self.llm.embed_tokens.parameters():
                    p.requires_grad = True
                for p in self.llm.layers[0].parameters():
                    p.requires_grad = True
                for p in self.llm.layers[-1].parameters():
                    p.requires_grad = True
                for p in self.llm.ln_f.parameters():
                    p.requires_grad = True
                for p in self.llm.lm_head.parameters():
                    p.requires_grad = True
        else:
            raise ValueError(f"未知冻结策略: {strategy}")

        self.config.freeze_strategy = strategy

    # ------------------------------------------------------------------ #
    # 多模态融合：把单个 <image> 占位展开为 N 个 patch embedding
    # ------------------------------------------------------------------ #
    def _merge_multimodal(self,
                          input_ids: torch.Tensor,
                          image_features: torch.Tensor,
                          image_mask: torch.Tensor,
                          attention_mask: Optional[torch.Tensor],
                          labels: Optional[torch.Tensor]):
        """
        Args:
            input_ids: [B, L]，每个样本恰好含 1 个 <image> token
            image_features: [B, P, H]（已投影到 LLM 空间）
            image_mask: [B, P]，1=有效 patch
            attention_mask: [B, L] 或 None
            labels: [B, L] 或 None
        Returns:
            inputs_embeds [B, L', H], attention_mask [B, L'], labels [B, L'] 或 None
        """
        device = input_ids.device
        B, L = input_ids.shape
        H = image_features.shape[-1]

        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).long()

        text_embeds = self.llm.embed_tokens(input_ids)  # [B, L, H]

        merged_embeds, merged_masks, merged_labels = [], [], []
        for b in range(B):
            pos_list = (input_ids[b] == self.image_token_id).nonzero(as_tuple=True)[0]
            n_valid = int(image_mask[b].sum().item()) if image_mask is not None else image_features.shape[1]
            patches = image_features[b][:n_valid] if image_mask is None else image_features[b][image_mask[b].bool()]

            if len(pos_list) == 0:
                # 没有 <image>，退化为纯文本
                merged_embeds.append(text_embeds[b])
                merged_masks.append(attention_mask[b])
                if labels is not None:
                    merged_labels.append(labels[b])
                continue

            pos = pos_list[0].item()
            emb = torch.cat([text_embeds[b, :pos], patches, text_embeds[b, pos + 1:]], dim=0)
            msk = torch.cat([
                attention_mask[b, :pos],
                torch.ones(patches.shape[0], dtype=attention_mask.dtype, device=device),
                attention_mask[b, pos + 1:],
            ], dim=0)
            merged_embeds.append(emb)
            merged_masks.append(msk)
            if labels is not None:
                lab = torch.cat([
                    labels[b, :pos],
                    torch.full((patches.shape[0],), -100, dtype=labels.dtype, device=device),
                    labels[b, pos + 1:],
                ], dim=0)
                merged_labels.append(lab)

        # 右侧 padding 到统一长度
        max_len = max(e.shape[0] for e in merged_embeds)
        embeds_out = torch.zeros(B, max_len, H, dtype=text_embeds.dtype, device=device)
        mask_out = torch.zeros(B, max_len, dtype=attention_mask.dtype, device=device)
        labels_out = torch.full((B, max_len), -100, dtype=torch.long, device=device) if labels is not None else None

        for b in range(B):
            l = merged_embeds[b].shape[0]
            embeds_out[b, :l] = merged_embeds[b]
            mask_out[b, :l] = merged_masks[b]
            if labels is not None:
                labels_out[b, :l] = merged_labels[b]

        return embeds_out, mask_out, labels_out

    def _encode_vision(self, vision_inputs):
        """vision encoder + projection，返回 (image_features [B,P,H], image_mask [B,P])。"""
        vision_output: VisionOutput = self.vision_encoder(vision_inputs)
        image_features = self.projection(vision_output.pixel_values)  # [B, P, llm_hidden]
        return image_features, vision_output.attention_mask

    # ------------------------------------------------------------------ #
    # 前向（训练）
    # ------------------------------------------------------------------ #
    def forward(self,
                vision_inputs: Union[torch.Tensor, Dict[str, Any]],
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        image_features, image_mask = self._encode_vision(vision_inputs)

        inputs_embeds, attn, merged_labels = self._merge_multimodal(
            input_ids, image_features, image_mask, attention_mask, labels
        )

        logits = self.llm.forward_with_embeddings(inputs_embeds, attn)

        loss = None
        if merged_labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = merged_labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {'logits': logits, 'loss': loss}

    # ------------------------------------------------------------------ #
    # 自回归生成
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(self,
                 vision_inputs: Union[torch.Tensor, Dict[str, Any]],
                 prompt: str,
                 tokenizer,
                 max_new_tokens: int = 256,
                 temperature: float = 0.7,
                 top_p: float = 0.9) -> str:
        self.eval()
        device = next(self.parameters()).device

        image_features, image_mask = self._encode_vision(vision_inputs)

        encoded = tokenizer.encode_prompt(prompt, image_token=True, max_length=1024,
                                          add_boa=True, padding=False)
        input_ids = torch.tensor([encoded['input_ids']], device=device)
        attention_mask = torch.tensor([encoded['attention_mask']], device=device)

        inputs_embeds, attn, _ = self._merge_multimodal(
            input_ids, image_features, image_mask, attention_mask, None
        )

        # 使用 HF 的 generate（如果可用）
        if self._hf_llm:
            generated_ids = self.llm.generate_from_embeddings(
                inputs_embeds=inputs_embeds,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=self.eoa_token_id,
                pad_token_id=self.pad_token_id,
            )
            # 只取新生成的 token（去掉 prompt/vision 部分）
            gen_len = generated_ids.shape[1]
            # generated_ids 包含全部序列，取最后 max_new_tokens 以内的新 token
            new_tokens = generated_ids[0, -max_new_tokens:]
            generated_ids_list = new_tokens.tolist()
            # 截断到 eoa
            if self.eoa_token_id in generated_ids_list:
                idx = generated_ids_list.index(self.eoa_token_id)
                generated_ids_list = generated_ids_list[:idx + 1]
        else:
            # MiniLLM 手写生成
            cur_embeds, cur_attn = inputs_embeds, attn
            generated_ids_list = []

            for _ in range(max_new_tokens):
                logits = self.llm.forward_with_embeddings(cur_embeds, cur_attn)
                next_logits = logits[0, -1, :] / max(temperature, 1e-6)

                # top-p sampling
                probs = torch.softmax(next_logits, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumprobs = torch.cumsum(sorted_probs, dim=-1)
                remove = cumprobs > top_p
                remove[1:] = remove[:-1].clone()
                remove[0] = False
                next_logits[sorted_idx[remove]] = float('-inf')

                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                tok_id = next_token.item()
                generated_ids_list.append(tok_id)

                if tok_id == self.eoa_token_id or tok_id == getattr(tokenizer, 'eos_token_id', -999):
                    break

                next_embed = self.llm.embed_tokens(next_token.unsqueeze(0))
                cur_embeds = torch.cat([cur_embeds, next_embed], dim=1)
                cur_attn = torch.cat([cur_attn, torch.ones((1, 1), dtype=cur_attn.dtype, device=device)], dim=1)

        output_text = tokenizer.decode(generated_ids_list, skip_special_tokens=False)
        extracted = tokenizer.extract_json_from_output(output_text)
        if extracted is not None:
            return json.dumps(extracted, ensure_ascii=False, indent=2)
        return output_text

    # ------------------------------------------------------------------ #
    # 参数统计
    # ------------------------------------------------------------------ #
    def print_trainable_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'\n{"="*50}')
        print(f'Total parameters: {total:,}')
        print(f'Trainable parameters: {trainable:,}')
        print(f'Trainable ratio: {100 * trainable / total:.2f}%')
        print(f'{"="*50}')
        print('Module-wise:')
        print(f'  Vision Encoder: total={self.vision_encoder.num_parameters:,}, trainable={self.vision_encoder.trainable_parameters:,}')
        print(f'  Projection: {self.projection.num_parameters:,} (trainable)')
        print(f'  LLM trainable: {self.llm.trainable_parameters:,}')

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_vlm_from_config(config_dict: Dict[str, Any]) -> ReceiptVLM:
    """从配置字典创建 VLM。"""
    valid = {k: v for k, v in config_dict.items() if k in VLMConfig.__dataclass_fields__}
    return ReceiptVLM(VLMConfig(**valid))


if __name__ == '__main__':
    print('Testing ReceiptVLM (fallback vision)...')
    config = VLMConfig(vision_model_name="fallback", llm_vocab_size=6167, llm_hidden_size=512, llm_num_layers=6)
    vlm = ReceiptVLM(config)
    vlm.print_trainable_parameters()

    # 构造一个含单个 <image> 的输入
    B, L = 2, 16
    input_ids = torch.randint(6, 6167, (B, L))
    input_ids[:, 0] = config.image_token_id  # 第 0 位为 <image>
    labels = input_ids.clone()
    labels[:, :3] = -100
    images = torch.randn(B, 3, 384, 384)

    out = vlm(images, input_ids, attention_mask=torch.ones(B, L), labels=labels)
    print(f'\nForward: logits {out["logits"].shape}, loss {out["loss"].item():.4f}')
    assert out['logits'].shape[1] == L - 1 + 576  # 展开 576 个 patch
    print('VLM forward test passed!')
