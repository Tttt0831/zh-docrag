"""
真实推理模块
实现完整的图片 → JSON 推理流程。

设计要点：
- 优先用 checkpoint 内嵌的 model_config 精确重建模型（保证视觉编码器/词表/层数一致）；
- 没有 checkpoint 时用默认轻量配置 + 本地 BPE tokenizer 构建（输出无意义，仅验证管线）；
- 视觉预处理交给 vision_encoder.process_images，与训练完全一致。
"""
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.vlm import ReceiptVLM, VLMConfig, create_vlm_from_config
from src.model.tokenizer import get_tokenizer


DEFAULT_PROMPT = """请从这张票据中抽取以下字段并以 JSON 输出:
merchant_name, date, total_amount, tax_amount, tax_id, invoice_no。
缺失字段填 null,不要编造。"""

QWEN2_VL_MODEL = "Qwen/Qwen2-VL-2B-Instruct"


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """扫描出第一个**配对完整**的 {...} 对象并解析。

    模型常在合法 JSON 之后继续生成（重复、特殊 token、乱码），导致整体串无法解析；
    贪婪的 `\\{.*\\}` 会从第一个 { 跨到最后一个 }，反而把多个对象粘在一起。
    这里按花括号配平取第一个对象，能从尾部带垃圾的输出里救回结果。
    """
    start = text.find('{')
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == '\\':
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i + 1])
                        except json.JSONDecodeError:
                            break
        start = text.find('{', start + 1)
    return None


def is_route_b_lora_path(checkpoint_path: Optional[str]) -> bool:
    """Route B checkpoint: directory with adapter_config.json at root (old) or lora_adapter/ subdir + no meta.pt."""
    if not checkpoint_path:
        return False
    path = Path(checkpoint_path)
    if not path.is_dir():
        return False
    # 轻量 Route A 也有 lora_adapter/ 子目录，但还有 meta.pt + projection.pt
    if (path / "meta.pt").exists() and (path / "projection.pt").exists():
        return False  # Route A lightweight
    # Old Route B: adapter files at root
    if (path / "adapter_config.json").exists() and (path / "adapter_model.safetensors").exists():
        return True
    # New Route B (if migrated to lora_adapter/ subdir)
    if (path / "lora_adapter" / "adapter_config.json").exists():
        return True
    return False


def is_hf_lora_lightweight(checkpoint_path: Optional[str]) -> bool:
    """检测 Route A 轻量 checkpoint 格式：目录 + meta.pt + projection.pt"""
    if not checkpoint_path:
        return False
    path = Path(checkpoint_path)
    return path.is_dir() and (path / "meta.pt").exists() and (path / "projection.pt").exists()


def build_vlm(checkpoint, tokenizer, device, fallback_vision="google/siglip2-base-patch16-naflex"):
    """根据 checkpoint（可为 None）构建 VLM。"""
    if checkpoint and "model_config" in checkpoint:
        model = create_vlm_from_config(checkpoint["model_config"]).to(device)
    else:
        cfg = VLMConfig(
            vision_model_name=fallback_vision,
            llm_vocab_size=tokenizer.vocab_size,
            llm_hidden_size=512,
            llm_num_layers=6,
            llm_num_heads=8,
            llm_intermediate_size=2048,
            max_sequence_length=1024,
            pad_token_id=tokenizer.pad_token_id,
            image_token_id=tokenizer.image_token_id,
            boa_token_id=tokenizer.boa_token_id,
            eoa_token_id=tokenizer.eoa_token_id,
        )
        model = ReceiptVLM(cfg).to(device)

    # 特殊 token 对齐 tokenizer
    model.pad_token_id = tokenizer.pad_token_id
    model.image_token_id = tokenizer.image_token_id
    model.boa_token_id = tokenizer.boa_token_id
    model.eoa_token_id = tokenizer.eoa_token_id
    return model


class ReceiptInference:
    """票据推理引擎"""

    def __init__(self,
                 checkpoint_path: Optional[str],
                 device: Optional[str] = None,
                 tokenizer_name: Optional[str] = None):
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.checkpoint_path = checkpoint_path
        self.mode = "route_b" if is_route_b_lora_path(checkpoint_path) else "route_a"
        self.has_checkpoint = False
        self.processor = None
        self.tokenizer = None

        if self.mode == "route_b":
            print(f'Loading Qwen2-VL route B model on {self.device}...')
            base_model = Qwen2VLForConditionalGeneration.from_pretrained(
                QWEN2_VL_MODEL,
                torch_dtype=torch.bfloat16 if self.device.type == 'cuda' else torch.float32,
                trust_remote_code=True,
            )
            lora_dir = Path(checkpoint_path) / "lora_adapter"
            self.model = PeftModel.from_pretrained(
                base_model, str(lora_dir) if lora_dir.exists() else checkpoint_path)
            self.model.to(self.device)
            self.processor = AutoProcessor.from_pretrained(QWEN2_VL_MODEL, trust_remote_code=True)
            self.has_checkpoint = True
            self.model.eval()
            print(f'✓ Loaded route B LoRA adapter from {checkpoint_path}')
            print(f'✓ Inference engine ready on {self.device}')
            return

        # Route A / C: check for lightweight HF LoRA format
        ckpt_path_obj = Path(checkpoint_path) if checkpoint_path else None
        if ckpt_path_obj and is_hf_lora_lightweight(str(ckpt_path_obj)):
            print(f'✓ 检测到轻量 HF LoRA checkpoint: {ckpt_path_obj}')
            meta = torch.load(ckpt_path_obj / "meta.pt", map_location=self.device, weights_only=False)
            print('Loading tokenizer...')
            self.tokenizer = get_tokenizer(tokenizer_name)
            model = create_vlm_from_config(meta["model_config"]).to(self.device)
            # 加载 LoRA adapter — 需要先解包 PeftModel 拿到基座
            lora_dir = ckpt_path_obj / "lora_adapter"
            if lora_dir.exists():
                current_model = model.llm.model
                if hasattr(current_model, 'get_base_model'):
                    base_transformer = current_model.get_base_model()
                elif hasattr(current_model, 'model'):
                    base_transformer = current_model.model
                else:
                    base_transformer = current_model
                model.llm.model = PeftModel.from_pretrained(base_transformer, str(lora_dir))
                # 更新缓存的组件引用
                model.llm._embed_tokens = model.llm.model.get_input_embeddings()
                model.llm._lm_head = model.llm.model.get_output_embeddings()
            # 加载 projection
            model.projection.load_state_dict(torch.load(ckpt_path_obj / "projection.pt", map_location=self.device))
            # 加载视觉可训练部分（如有）
            vis_path = ckpt_path_obj / "vision_trainable.pt"
            if vis_path.exists():
                vis_sd = torch.load(vis_path, map_location=self.device)
                missing, unexpected = model.load_state_dict(vis_sd, strict=False)
                print(f'  ✓ 加载视觉可训练权重 (missing={len(missing)}, unexpected={len(unexpected)})')
            model.pad_token_id = self.tokenizer.pad_token_id
            model.image_token_id = self.tokenizer.image_token_id
            model.boa_token_id = self.tokenizer.boa_token_id
            model.eoa_token_id = self.tokenizer.eoa_token_id
            self.model = model
            self.has_checkpoint = True
            epoch = meta.get("epoch")
            tail = f" (epoch {epoch + 1})" if epoch is not None else ""
            print(f'✓ Loaded lightweight checkpoint{tail}')
            self.model.eval()
            print(f'✓ Inference engine ready on {self.device}')
            return

        print('Loading tokenizer...')
        self.tokenizer = get_tokenizer(tokenizer_name)

        checkpoint = None
        if checkpoint_path and Path(checkpoint_path).exists():
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            except Exception as e:
                print(f'Warning: 读取 checkpoint 失败: {e}')

        print(f'Building model on {self.device}...')
        self.model = build_vlm(checkpoint, self.tokenizer, self.device)

        if checkpoint is not None:
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.has_checkpoint = True
                epoch = checkpoint.get('epoch')
                tail = f" (epoch {epoch + 1})" if epoch is not None else ""
                print(f'✓ Loaded checkpoint{tail}')
            except Exception as e:
                print(f'Warning: 加载权重失败: {e}')
                print('使用随机初始化权重（输出无意义，仅验证管线）')
        else:
            print('未找到 checkpoint；使用随机初始化权重（输出无意义，仅验证管线）')

        self.model.eval()
        print(f'✓ Inference engine ready on {self.device}')

    def generate_json(self,
                      image: Image.Image,
                      prompt: str = DEFAULT_PROMPT,
                      max_new_tokens: int = 256,
                      temperature: float = 0.7,
                      top_p: float = 0.9) -> Dict[str, Any]:
        """图片 → 预测 JSON dict。"""
        if self.mode == "route_b":
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image.convert('RGB')},
                    {"type": "text", "text": prompt.replace('<image>', '').strip()},
                ],
            }]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=[image_inputs],
                padding=True,
                return_tensors='pt',
            )
            inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
            )
            prompt_len = inputs['input_ids'].shape[1]
            output_text = self.processor.tokenizer.decode(
                generated_ids[0][prompt_len:],
                skip_special_tokens=True,
            )
        else:
            vision_inputs = self.model.vision_encoder.process_images([image.convert('RGB')], device=self.device)
            output_text = self.model.generate(
                vision_inputs=vision_inputs,
                prompt=prompt,
                tokenizer=self.tokenizer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        try:
            return json.loads(output_text)
        except json.JSONDecodeError:
            # 优先取第一个配平完整的 JSON 对象（容忍尾部重复/乱码）
            first = extract_first_json_object(output_text)
            if first is not None:
                return first
            extractor = self.tokenizer.extract_json_from_output if self.tokenizer is not None else None
            if extractor is not None:
                extracted = extractor(output_text)
                if extracted is not None:
                    return extracted
            return {'error': 'Failed to parse model output as JSON', 'raw_output': output_text}

    def batch_infer(self,
                    image_paths: List[str],
                    prompts: Optional[List[str]] = None,
                    **generation_kwargs) -> List[Dict[str, Any]]:
        if prompts is None:
            prompts = [DEFAULT_PROMPT] * len(image_paths)
        results = []
        for img_path, prompt in zip(image_paths, prompts):
            try:
                image = Image.open(img_path).convert('RGB')
                results.append(self.generate_json(image, prompt, **generation_kwargs))
            except Exception as e:
                print(f'Error processing {img_path}: {e}')
                results.append({'error': str(e), 'image_path': img_path})
        return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='测试推理')
    parser.add_argument('--checkpoint', default=None, help='模型checkpoint路径')
    parser.add_argument('--data', default='data/processed/test.jsonl', help='测试数据路径')
    parser.add_argument('--max-samples', type=int, default=5, help='最大测试样本数')
    args = parser.parse_args()

    print('=' * 50)
    print('Testing Real Inference Pipeline')
    print('=' * 50)

    repo_root = Path(__file__).resolve().parent.parent

    # 确定checkpoint路径
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = str(repo_root / 'checkpoints' / 'route_a' / 'best_model.pt')

    print(f'使用 checkpoint: {checkpoint_path}')
    inferencer = ReceiptInference(checkpoint_path)

    # 测试单张图片
    test_image_path = repo_root / 'data' / 'synthetic' / 'test_invoice.png'
    if not test_image_path.exists():
        # 用合成器即时生成一张
        from src.data.synth import generate_invoice_data, draw_receipt
        data = generate_invoice_data('vat_general')
        img = draw_receipt(data, template='vat_general')
        test_image_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(test_image_path)

    image = Image.open(test_image_path).convert('RGB')
    print(f'\nImage size: {image.size}')
    result = inferencer.generate_json(image, max_new_tokens=128)
    print('\n🔍 Prediction Result:')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('\n✓ Inference test completed')


if __name__ == '__main__':
    main()
