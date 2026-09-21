"""
SigLIP2 NaFlex 视觉编码器封装
使用动态分辨率，保留原生宽高比，支持可变序列长度

两种工作模式：
  1. 真·SigLIP2 NaFlex（model_name="google/siglip2-base-patch16-naflex"）：
     用官方 processor 产出 pixel_values / pixel_attention_mask / spatial_shapes，
     喂给 vision_model，返回可变长度的 patch 序列 + attention_mask。
  2. fallback 卷积编码器（model_name="fallback"）：离线/CPU 快速路径，
     固定 384×384 → 576 patch，输出维度同为 768。
"""
import torch
import torch.nn as nn
from typing import Optional, Union, Dict, Any
from dataclasses import dataclass


@dataclass
class VisionOutput:
    """视觉编码器输出"""
    pixel_values: torch.Tensor          # [batch, num_patches, hidden_dim]
    attention_mask: Optional[torch.Tensor]  # [batch, num_patches] 1=valid, 0=pad
    num_patches: int                     # patch 数（可变，等于序列维度长度）


class SigLIP2VisionEncoder(nn.Module):
    """SigLIP2 NaFlex 视觉编码器（冻结），输出视觉特征供投影到 LLM 空间。"""

    def __init__(self,
                 model_name: str = "google/siglip2-base-patch16-naflex",
                 freeze: bool = True,
                 max_num_patches: int = 256):
        super().__init__()

        self.model_name = model_name
        self.max_num_patches = max_num_patches

        # fallback 卷积编码器（始终创建；当真模型不可用或输入为裸 tensor 时使用）
        self._fallback_encoder = nn.Sequential(
            nn.Conv2d(3, 768, kernel_size=16, stride=16, padding=0),
            nn.ReLU(),
        )

        self.model = None
        self.processor = None

        # model_name == "fallback" 时直接走 fallback，不尝试联网
        if model_name and model_name != "fallback":
            from transformers import AutoModel, AutoProcessor
            try:
                self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
                print(f"Successfully loaded vision model: {model_name}")
            except Exception as e:
                print(f"Warning: Could not load {model_name}: {e}")
                print("Using fallback encoder for demonstration...")
            try:
                self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            except Exception as e:
                print(f"Warning: Could not load processor: {e}")
                self.processor = None

        # 输出维度
        if self.model is not None:
            self.hidden_size = self.model.config.vision_config.hidden_size
        else:
            self.hidden_size = 768

        # 冻结
        if freeze and self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def set_top_trainable(self, num_layers: int) -> int:
        """部分解冻：放开 vision_model 顶部 num_layers 个 encoder 层 + post_layernorm。

        用于路线 A 的视觉+投影联合微调，增强视觉接地。返回解冻的参数量。
        """
        if self.model is None or num_layers <= 0:
            return 0
        vm = self.model.vision_model
        trainable = 0
        for layer in vm.encoder.layers[-num_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
                trainable += p.numel()
        if hasattr(vm, "post_layernorm"):
            for p in vm.post_layernorm.parameters():
                p.requires_grad = True
                trainable += p.numel()
        return trainable

    # ---------------------------------------------------------------- #
    # 预处理：把 PIL 图片列表转成编码器需要的输入
    # ---------------------------------------------------------------- #
    def process_images(self, images, device: Optional[torch.device] = None) -> Dict[str, Any]:
        """
        将 PIL 图片（单张或列表）转成编码器输入。

        Returns:
            dict:
              - 真模型: {pixel_values, pixel_attention_mask, spatial_shapes}
              - fallback: {pixel_values: [B,3,384,384]}
        """
        if not isinstance(images, (list, tuple)):
            images = [images]

        if self.model is not None and self.processor is not None:
            inputs = self.processor(
                images=list(images),
                return_tensors="pt",
                max_num_patches=self.max_num_patches,
            )
            out = {k: v for k, v in inputs.items()}
        else:
            out = {"pixel_values": self._fallback_preprocess(images)}

        if device is not None:
            out = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in out.items()}
        return out

    @staticmethod
    def _fallback_preprocess(images) -> torch.Tensor:
        """fallback：resize 到 384 + 归一化，返回 [B,3,384,384]。"""
        import torchvision.transforms as T
        transform = T.Compose([
            T.Resize((384, 384)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        return torch.stack([transform(img.convert("RGB")) for img in images])

    # ---------------------------------------------------------------- #
    # 前向
    # ---------------------------------------------------------------- #
    def forward(self, vision_inputs: Union[torch.Tensor, Dict[str, Any]]) -> VisionOutput:
        """
        Args:
            vision_inputs:
              - dict 且含 spatial_shapes → 真·NaFlex 路径
              - dict 仅含 pixel_values / 或裸 Tensor → fallback 路径
        """
        # 真·NaFlex
        if isinstance(vision_inputs, dict) and "spatial_shapes" in vision_inputs and self.model is not None:
            return self._naflex_forward(vision_inputs)

        # fallback：取出 tensor
        if isinstance(vision_inputs, dict):
            images = vision_inputs["pixel_values"]
        else:
            images = vision_inputs
        return self._fallback_forward(images)

    def _naflex_forward(self, inputs: Dict[str, Any]) -> VisionOutput:
        """真·SigLIP2 NaFlex 前向。"""
        vision_model = self.model.vision_model
        pixel_values = inputs["pixel_values"]
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = inputs.get("pixel_attention_mask")
        spatial_shapes = inputs.get("spatial_shapes")

        # 如果 processor 没有提供这些必需的参数，需要构造默认值
        if attention_mask is None:
            if pixel_values.dim() == 3:
                b, p, _ = pixel_values.shape
                attention_mask = torch.ones((b, p), dtype=torch.long, device=pixel_values.device)
            else:
                b, _, h, w = pixel_values.shape
                attention_mask = torch.ones((b, h * w), dtype=torch.long, device=pixel_values.device)

        if spatial_shapes is None:
            if pixel_values.dim() == 4:
                b, _, h, w = pixel_values.shape
                spatial_shapes = torch.tensor([[h, w]], dtype=torch.long, device=pixel_values.device)
                if b > 1:
                    spatial_shapes = spatial_shapes.repeat(b, 1)
            else:
                b, p, _ = pixel_values.shape
                spatial_shapes = torch.tensor([[1, p]], dtype=torch.long, device=pixel_values.device)
                if b > 1:
                    spatial_shapes = spatial_shapes.repeat(b, 1)

        # SigLIP2 的两层包装用了不同的参数名：
        #   Siglip2VisionModel.forward       -> pixel_attention_mask
        #   Siglip2VisionTransformer.forward -> attention_mask
        # self.model.vision_model 拿到的是内层的 Transformer，所以要用
        # attention_mask。这里按真实签名探测，兼容两种情况，避免 transformers
        # 版本变动再次踩坑。
        import inspect
        _params = inspect.signature(vision_model.forward).parameters
        _mask_key = "pixel_attention_mask" if "pixel_attention_mask" in _params else "attention_mask"
        outputs = vision_model(
            pixel_values=pixel_values,
            spatial_shapes=spatial_shapes,
            **{_mask_key: attention_mask},
        )
        last_hidden = outputs.last_hidden_state  # [B, P, hidden]

        # 构造序列级别的 attention_mask（用于 LLM）
        b, p, _ = last_hidden.shape
        sequence_attention_mask = torch.ones((b, p), dtype=torch.long, device=last_hidden.device)

        return VisionOutput(
            pixel_values=last_hidden,
            attention_mask=sequence_attention_mask.long(),
            num_patches=p,
        )

    def _fallback_forward(self, images: torch.Tensor) -> VisionOutput:
        """fallback 卷积前向：[B,3,384,384] → [B,576,768]。"""
        features = self._fallback_encoder(images)          # [B,768,24,24]
        features = features.flatten(2).transpose(1, 2)      # [B,576,768]
        b, p, _ = features.shape
        attention_mask = torch.ones((b, p), dtype=torch.long, device=features.device)
        return VisionOutput(pixel_values=features, attention_mask=attention_mask, num_patches=p)

    @property
    def num_parameters(self) -> int:
        if self.model is not None:
            return sum(p.numel() for p in self.model.parameters())
        return sum(p.numel() for p in self._fallback_encoder.parameters())

    @property
    def trainable_parameters(self) -> int:
        if self.model is not None:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return 0  # fallback 不计入可训练（演示用）


if __name__ == '__main__':
    print('Testing fallback vision encoder...')
    encoder = SigLIP2VisionEncoder(model_name="fallback", freeze=True)
    print(f'hidden size: {encoder.hidden_size}')

    images = torch.randn(2, 3, 384, 384)
    out = encoder(images)
    print(f'Output: {out.pixel_values.shape}, mask: {out.attention_mask.shape}, patches: {out.num_patches}')
    assert out.pixel_values.shape == (2, 576, 768)
    print('Fallback vision encoder test passed!')
