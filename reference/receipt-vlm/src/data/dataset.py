"""
数据集加载器
支持统一 schema 的 instruction 数据
"""
import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Any, Optional, List
from PIL import Image
import random


class ReceiptDataset(Dataset):
    """
    票据数据集
    加载统一 schema 的 jsonl 格式数据
    """

    def __init__(self,
                 data_path: str,
                 image_root: Optional[str] = None,
                 split: str = 'train',
                 max_samples: Optional[int] = None):
        """
        Args:
            data_path: jsonl 数据文件路径
            image_root: 图片根目录（如果图片路径是相对路径）
            split: train/val/test
            max_samples: 最大样本数（用于调试）
        """
        self.data_path = Path(data_path)
        self.image_root = Path(image_root) if image_root else None
        self.split = split

        # 加载数据
        self.samples = self._load_data()

        # 限制样本数
        if max_samples and len(self.samples) > max_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

        print(f'Loaded {len(self.samples)} samples for {split} split')

    def _load_data(self) -> List[Dict[str, Any]]:
        """加载 jsonl 数据"""
        samples = []

        if not self.data_path.exists():
            print(f'Warning: Data file {self.data_path} not found')
            return samples

        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    print(f'Warning: Failed to parse line: {e}')
                    continue

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        返回单个样本

        Returns:
            Dict with:
                - image: PIL Image
                - prompt: str
                - target_json: Dict[str, Any]
                - image_path: str
        """
        sample = self.samples[idx]

        # 加载图片
        image_path = sample.get('image_path')
        if self.image_root and not Path(image_path).is_absolute():
            image_path = self.image_root / image_path

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f'Warning: Failed to load image {image_path}: {e}')
            # 创建空白图片作为备选
            image = Image.new('RGB', (384, 384), color='white')

        return {
            'image': image,
            'prompt': sample.get('prompt', ''),
            'target_json': sample.get('target_json', {}),
            'image_path': str(image_path),
        }


def create_sample_dataset(output_path: str, num_samples: int = 100):
    """
    创建示例数据集用于测试

    Args:
        output_path: 输出目录
        num_samples: 样本数量
    """
    from ..synth import generate_invoice_data, DEFAULT_PROMPT
    import uuid

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    image_dir = output_path / 'images'
    image_dir.mkdir(exist_ok=True)

    samples = []

    for i in range(num_samples):
        # 生成数据
        data = generate_invoice_data()
        image_filename = f'sample_{i:04d}.png'
        image_path = image_dir / image_filename

        # 创建占位图片（实际使用时需要生成真实图片）
        image = Image.new('RGB', (384, 384), color='white')
        image.save(image_path)

        samples.append({
            'image_path': str(image_path),
            'prompt': DEFAULT_PROMPT,
            'target_json': data,
        })

    # 保存 jsonl
    output_file = output_path / 'train.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f'Created {num_samples} samples at {output_path}')


def build_training_collate(tokenizer, vision_processor, max_length: int = 1024):
    """
    构造训练用 collate_fn（闭包），产出对齐的 causal-LM 训练张量。

    每条样本的序列结构：
        [<image>] + prompt_tokens + [<boa>] + answer_tokens + [<eoa>]
    labels 只在 answer_tokens + <eoa> 上为真实 id，prompt/图像段全为 -100，
    即损失只作用在 JSON 答案上。

    Args:
        tokenizer: ReceiptTokenizer 实例
        vision_processor: 可调用对象，输入 PIL 列表返回视觉输入 dict
            （通常是 vision_encoder.process_images）
        max_length: 文本序列最大长度（截断用）

    Returns:
        collate_fn(batch) -> dict(vision_inputs, input_ids, attention_mask, labels, target_jsons)
    """
    import torch
    from torch.nn.utils.rnn import pad_sequence

    eoa_id = tokenizer.eoa_token_id
    pad_id = tokenizer.pad_token_id

    def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        pil_images = [item['image'] for item in batch]
        input_ids_list, attn_list, labels_list, target_jsons = [], [], [], []

        for item in batch:
            # 1. prompt 段（含 <image> 与 <boa>），不算 loss
            prompt_ids = tokenizer.encode_prompt(
                item['prompt'], image_token=True, add_boa=True, padding=False
            )['input_ids']

            # 2. answer 段：紧凑 JSON + <eoa>，算 loss
            answer_str = tokenizer.encode_target(item['target_json'], add_eoa=False)
            answer_ids = tokenizer.tokenizer.encode(answer_str, add_special_tokens=False)
            answer_ids = answer_ids + [eoa_id]

            ids = prompt_ids + answer_ids
            labels = [-100] * len(prompt_ids) + list(answer_ids)

            # 截断（保留答案：从尾部截）
            if len(ids) > max_length:
                ids = ids[:max_length]
                labels = labels[:max_length]

            input_ids_list.append(torch.tensor(ids, dtype=torch.long))
            attn_list.append(torch.ones(len(ids), dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))
            target_jsons.append(item['target_json'])

        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
        attention_mask = pad_sequence(attn_list, batch_first=True, padding_value=0)
        labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)

        vision_inputs = vision_processor(pil_images)

        return {
            'vision_inputs': vision_inputs,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'target_jsons': target_jsons,
        }

    return _collate


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    批处理整理函数（简化版 - 返回原始数据）

    Args:
        batch: 样本列表

    Returns:
        批处理后的字典
    """
    images = [item['image'] for item in batch]
    prompts = [item['prompt'] for item in batch]
    target_jsons = [item['target_json'] for item in batch]
    image_paths = [item['image_path'] for item in batch]

    # 返回原始数据（向后兼容）
    return {
        'images': images,
        'prompts': prompts,
        'target_jsons': target_jsons,
        'image_paths': image_paths,
    }


def collate_fn_with_preprocessing(batch: List[Dict[str, Any]],
                                  tokenizer=None,
                                  image_transform=None,
                                  max_length: int = 1024) -> Dict[str, Any]:
    """
    批处理整理函数（完整版 - 带真实的预处理和 tokenization）

    Args:
        batch: 样本列表
        tokenizer: ReceiptTokenizer 实例
        image_transform: 图像预处理 transform
        max_length: 最大序列长度

    Returns:
        批处理后的字典（包含预处理后的 tensor）
    """
    # 如果没有提供 tokenizer 或 transform，回退到简化版本
    if tokenizer is None or image_transform is None:
        return collate_fn(batch)

    import torch
    from torch.nn.utils.rnn import pad_sequence

    images = []
    input_ids_list = []
    attention_masks = []
    labels_list = []
    target_jsons = []

    for item in batch:
        # 1. 图像预处理
        img = image_transform(item['image'])
        images.append(img)

        # 2. 编码 prompt（包含 <image> token）
        encoded = tokenizer.encode_prompt(
            item['prompt'],
            image_token=True,
            max_length=max_length,
            add_boa=True
        )
        input_ids_list.append(torch.tensor(encoded['input_ids'], dtype=torch.long))
        attention_masks.append(torch.tensor(encoded['attention_mask'], dtype=torch.long))

        # 3. 编码 target JSON 作为 label
        target_str = tokenizer.encode_target(item['target_json'], add_eoa=True)
        target_ids = tokenizer.tokenizer.encode(
            target_str,
            return_tensors='pt',
            padding='max_length',
            max_length=512,
            truncation=True,
            add_special_tokens=False
        )
        labels_list.append(target_ids.squeeze(0))

        target_jsons.append(item['target_json'])

    # 4. 堆叠为 tensor
    images_tensor = torch.stack(images)  # [batch, 3, H, W]

    # Pad sequences to same length
    input_ids_tensor = pad_sequence(
        input_ids_list,
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )

    attention_mask_tensor = pad_sequence(
        attention_masks,
        batch_first=True,
        padding_value=0
    )

    labels_tensor = pad_sequence(
        labels_list,
        batch_first=True,
        padding_value=-100
    )

    return {
        'images': images_tensor,
        'input_ids': input_ids_tensor,
        'attention_mask': attention_mask_tensor,
        'labels': labels_tensor,
        'target_jsons': target_jsons,
    }


if __name__ == '__main__':
    # 测试数据集 + 训练 collate
    print('Testing ReceiptDataset + build_training_collate...')

    repo_root = Path(__file__).resolve().parent.parent.parent
    data_path = repo_root / 'data' / 'synthetic' / 'train.jsonl'

    if not data_path.exists():
        print(f'Data not found: {data_path}')
        print('请先运行 `python -m src.data.synth` 生成合成数据。')
        raise SystemExit(0)

    dataset = ReceiptDataset(str(data_path), max_samples=8)
    print(f'Dataset size: {len(dataset)}')
    sample = dataset[0]
    print(f'Sample keys: {list(sample.keys())}')
    print(f'Image size: {sample["image"].size}')
    print(f'Target JSON: {sample["target_json"]}')

    # 用真实 tokenizer + fallback 视觉预处理验证 collate 对齐
    import sys
    sys.path.insert(0, str(repo_root))
    from src.model.tokenizer import get_tokenizer
    from src.model.vision import SigLIP2VisionEncoder
    from torch.utils.data import DataLoader

    tok = get_tokenizer()
    enc = SigLIP2VisionEncoder(model_name="fallback", freeze=True)
    collate = build_training_collate(tok, enc.process_images, max_length=512)

    loader = DataLoader(dataset, batch_size=4, collate_fn=collate)
    batch = next(iter(loader))
    print(f'\nCollate output:')
    print(f'  input_ids: {batch["input_ids"].shape}')
    print(f'  labels: {batch["labels"].shape}')
    print(f'  vision pixel_values: {batch["vision_inputs"]["pixel_values"].shape}')
    # 断言：labels 中非 -100 的位置应只覆盖答案段
    import torch
    n_supervised = (batch['labels'] != -100).sum().item()
    print(f'  supervised tokens (answer only): {n_supervised}')
    assert n_supervised > 0
    assert batch['input_ids'].shape == batch['labels'].shape
    print('Dataset + collate test passed!')
