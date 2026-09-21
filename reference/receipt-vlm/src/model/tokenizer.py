"""
票据 VLM 专用 Tokenizer
基于 Qwen2 tokenizer，支持中英文和特殊 tokens
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from transformers import AutoTokenizer

# 本项目自训练的小词表 BPE tokenizer 默认位置（由 scripts/train_tokenizer.py 生成）
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LOCAL_TOKENIZER_DIR = _REPO_ROOT / "tokenizers" / "receipt-bpe"


class ReceiptTokenizer:
    """
    票据 VLM 专用 tokenizer

    加载优先级：
      1. 本项目自训练的小词表 BPE（tokenizers/receipt-bpe/）——让模型保持轻量
      2. 显式传入的 base_model_name（如 Qwen/Qwen2-1.5B）
      3. 回退 bert-base-chinese

    功能：
    - 支持中英文 tokenization
    - 特殊 tokens: <image>, <boa>, <eoa>
    - 编码 prompt 和 target JSON
    """

    def __init__(self, base_model_name: Optional[str] = None):
        """
        初始化 tokenizer

        Args:
            base_model_name: 显式指定的 tokenizer 名称/路径。
                为 None 时优先使用本地 BPE，没有则回退 Qwen/Qwen2-1.5B。
        """
        source = self._resolve_source(base_model_name)
        self.model_name = source

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                source,
                trust_remote_code=True
            )
            print(f"✓ Loaded tokenizer from {source}")
        except Exception as e:
            print(f"Warning: Failed to load tokenizer from {source}: {e}")
            # 回退到默认 tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
            self.model_name = "bert-base-chinese"
            print("Using bert-base-chinese as fallback")

        # 配置特殊 tokens
        self._setup_special_tokens()

    @staticmethod
    def _resolve_source(base_model_name: Optional[str]) -> str:
        """决定从哪里加载 tokenizer。"""
        if base_model_name:
            return base_model_name
        if LOCAL_TOKENIZER_DIR.exists() and (LOCAL_TOKENIZER_DIR / "tokenizer.json").exists():
            return str(LOCAL_TOKENIZER_DIR)
        # 没有本地 BPE 时回退到 Qwen2（需联网下载）
        return "Qwen/Qwen2-1.5B"

    def _setup_special_tokens(self):
        """配置特殊 tokens"""
        # 定义特殊 tokens
        special_tokens = {
            'image_token': '<image>',
            'boa_token': '<boa>',   # begin of answer
            'eoa_token': '<eoa>',   # end of answer
        }

        # 添加到 tokenizer（如果不存在）
        special_tokens_list = list(special_tokens.values())
        existing_special = self.tokenizer.special_tokens_map.get('additional_special_tokens', [])

        tokens_to_add = [t for t in special_tokens_list if t not in existing_special]
        if tokens_to_add:
            self.tokenizer.add_special_tokens({
                'additional_special_tokens': tokens_to_add
            })

        # 获取特殊 token IDs
        self.image_token_id = self.tokenizer.convert_tokens_to_ids('<image>')
        self.boa_token_id = self.tokenizer.convert_tokens_to_ids('<boa>')
        self.eoa_token_id = self.tokenizer.convert_tokens_to_ids('<eoa>')

        # 获取词汇表大小
        self.vocab_size = len(self.tokenizer)

        print(f"Special tokens: <image>={self.image_token_id}, <boa>={self.boa_token_id}, <eoa>={self.eoa_token_id}")
        print(f"Vocab size: {self.vocab_size}")

    def encode_prompt(self,
                      prompt: str,
                      image_token: bool = True,
                      max_length: int = 1024,
                      add_boa: bool = True,
                      padding: str = 'max_length') -> Dict[str, List[int]]:
        """
        编码提示为 input_ids

        Args:
            prompt: 文本提示
            image_token: 是否在开头添加 <image> token
            max_length: 最大序列长度
            add_boa: 是否在末尾添加 <boa> token（JSON 开始）
            padding: 填充策略。训练时用 'max_length' 对齐 batch；
                     自回归生成时应传 False（不填充），否则生成会从 pad token 续写

        Returns:
            Dict with input_ids and attention_mask
        """
        # 去掉传入文本里可能已存在的 <image>，避免出现多个占位符
        prompt = prompt.replace("<image>", "").strip()

        # 构造完整 prompt
        full_prompt = ""
        if image_token:
            full_prompt += "<image> "

        full_prompt += prompt

        if add_boa:
            full_prompt += " <boa>"

        # 使用 tokenizer 编码
        encoding = self.tokenizer(
            full_prompt,
            return_tensors='pt',
            padding=padding,
            max_length=max_length,
            truncation=True,
            add_special_tokens=True
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0).tolist(),
            'attention_mask': encoding['attention_mask'].squeeze(0).tolist()
        }

    def encode_target(self, target_json: Dict[str, Any], add_eoa: bool = True) -> str:
        """
        将目标 JSON 编码为字符串

        Args:
            target_json: 目标 JSON 字典
            add_eoa: 是否在末尾添加 <eoa> token

        Returns:
            编码后的字符串
        """
        # 转换为 JSON 字符串（紧凑格式）
        json_str = json.dumps(target_json, ensure_ascii=False, separators=(',', ':'))

        if add_eoa:
            json_str += " <eoa>"

        return json_str

    def encode_target_with_prompt(self,
                                   target_json: Dict[str, Any],
                                   max_length: int = 512) -> Dict[str, List[int]]:
        """
        编码目标 JSON（用于训练 label）

        Args:
            target_json: 目标 JSON 字典
            max_length: 最大序列长度

        Returns:
            Dict with input_ids and attention_mask
        """
        target_str = self.encode_target(target_json, add_eoa=True)

        encoding = self.tokenizer(
            target_str,
            return_tensors='pt',
            padding='max_length',
            max_length=max_length,
            truncation=True,
            add_special_tokens=False
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0).tolist(),
            'attention_mask': encoding['attention_mask'].squeeze(0).tolist()
        }

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        解码 token IDs 为文本

        Args:
            token_ids: token ID 列表
            skip_special_tokens: 是否跳过特殊 tokens

        Returns:
            解码后的文本
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def extract_json_from_output(self, output_text: str) -> Optional[Dict[str, Any]]:
        """
        从模型输出中提取 JSON

        Args:
            output_text: 模型输出的文本

        Returns:
            解析后的 JSON 字典，失败返回 None
        """
        import re

        # 尝试找到 JSON 部分（在 { 和 } 之间）
        json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    @property
    def pad_token_id(self) -> int:
        """获取 padding token ID"""
        if self.tokenizer.pad_token is not None:
            return self.tokenizer.pad_token_id
        return self.tokenizer.eos_token_id

    @property
    def eos_token_id(self) -> int:
        """获取结束 token ID"""
        return self.tokenizer.eos_token_id


# 全局 tokenizer 实例（延迟加载）
_global_tokenizer: Optional[ReceiptTokenizer] = None


def get_tokenizer(model_name: Optional[str] = None) -> ReceiptTokenizer:
    """
    获取全局 tokenizer 实例

    Args:
        model_name: tokenizer 名称/路径。None 时优先用本地 BPE，回退 Qwen2。

    Returns:
        ReceiptTokenizer 实例
    """
    global _global_tokenizer
    if _global_tokenizer is None:
        _global_tokenizer = ReceiptTokenizer(model_name)
    return _global_tokenizer


def reset_tokenizer():
    """重置全局 tokenizer（用于测试）"""
    global _global_tokenizer
    _global_tokenizer = None


if __name__ == '__main__':
    # 测试 tokenizer
    print('='*50)
    print('Testing ReceiptTokenizer')
    print('='*50)

    tokenizer = ReceiptTokenizer()

    # 测试编码 prompt
    prompt = "请从这张票据中抽取以下字段并以 JSON 输出: merchant_name, date, total_amount"
    encoded = tokenizer.encode_prompt(prompt, image_token=True)

    print(f'\nEncoded prompt:')
    print(f'  Input IDs length: {len(encoded["input_ids"])}')
    print(f'  First 20 tokens: {encoded["input_ids"][:20]}')
    print(f'  Image token at position: {encoded["input_ids"].index(tokenizer.image_token_id) if tokenizer.image_token_id in encoded["input_ids"] else "Not found"}')

    # 测试编码 target JSON
    target_json = {
        'merchant_name': '北京科技有限公司',
        'date': '2025-01-15',
        'total_amount': 1250.00
    }
    target_str = tokenizer.encode_target(target_json)
    print(f'\nEncoded target:')
    print(f'  String: {target_str}')

    # 测试解码
    decoded = tokenizer.decode(encoded['input_ids'][:50])
    print(f'\nDecoded (first 50 tokens):')
    print(f'  {decoded}')

    # 测试 JSON 提取
    test_output = "这里有一些文本 {\"merchant_name\": \"北京科技有限公司\", \"date\": \"2025-01-15\"} 更多文本"
    extracted = tokenizer.extract_json_from_output(test_output)
    print(f'\nExtracted JSON:')
    print(f'  {json.dumps(extracted, ensure_ascii=False, indent=2)}')

    print('\n✓ Tokenizer test passed!')
