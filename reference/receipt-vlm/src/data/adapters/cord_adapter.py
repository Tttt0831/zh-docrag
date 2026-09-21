"""
CORD 数据集适配器

支持两种数据源:
  1. 本地 CORD 目录（原始 .jpg/.json 对）
  2. HuggingFace datasets (naver-clova-ix/cord-v2)

CORD (Receipt OCR Dataset) 数据集结构:
- 图片: JPG 格式的小票图片
- 标注: JSON 格式，包含:
  - menu: 商品列表 [{nm, unitprice, cnt, price}, ...]
  - subtotal: 小计金额
  - total: 总金额
  - tax: 税额 (可选)
  - date: 日期 (可选)
  - company: 商户名 (可选)

HF 版本 (gt_parse 格式):
  - gt_parse.menu: [{nm, unitprice, cnt, price}, ...]
  - gt_parse.subtotal: {subtotal_price, tax_price, ...}
  - gt_parse.total: {total_price, cashprice, changeprice}

映射策略:
- merchant_name: 从 company 或 menu.sub_nm 推断，缺失则为 null
- date: CORD 无日期字段 → null
- total_amount: 从 total.total_price 提取
- tax_amount: 从 subtotal.tax_price 提取
- tax_id: CORD 数据集不包含税号，固定为 null
- invoice_no: CORD 数据集不包含发票号，固定为 null
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

# Ensure project root is on path (needed when running this file directly)
_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


class CORDAdapter:
    """CORD 数据集适配器"""

    def __init__(self, data_root: Path):
        """
        Args:
            data_root: CORD 数据集根目录
                预期结构:
                data_root/
                ├── train/          # 训练集
                │   ├── xxx.jpg
                │   └── xxx.json
                └── test/           # 测试集
                    ├── xxx.jpg
                    └── xxx.json
        """
        self.data_root = Path(data_root)

    def load_annotation(self, json_path: Path) -> Dict[str, Any]:
        """
        加载 CORD JSON 标注文件

        Args:
            json_path: JSON 文件路径

        Returns:
            原始标注数据
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def extract_merchant_name(self, annotation: Dict[str, Any]) -> Optional[str]:
        """
        提取商户名称

        CORD 字段: company (可选)
        映射决策: 直接使用 company 字段，缺失则返回 null
        """
        company = annotation.get('company')
        if company and isinstance(company, str) and company.strip():
            return company.strip()
        return None

    def extract_date(self, annotation: Dict[str, Any]) -> Optional[str]:
        """
        提取并归一化日期

        CORD 字段: date (可选，格式多样)
        映射决策: 尝试解析多种日期格式，统一为 YYYY-MM-DD
        """
        date_str = annotation.get('date')
        if not date_str or not isinstance(date_str, str):
            return None

        date_str = date_str.strip()

        # 尝试多种日期格式
        date_formats = [
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%d-%m-%Y',
            '%d/%m/%Y',
            '%m-%d-%Y',
            '%m/%d/%Y',
            '%Y%m%d',
            '%d.%m.%Y',
            '%Y.%m.%d',
        ]

        for fmt in date_formats:
            try:
                date_obj = datetime.strptime(date_str, fmt)
                return date_obj.strftime('%Y-%m-%d')
            except ValueError:
                continue

        # 如果常规格式都失败，尝试提取数字
        import re
        numbers = re.findall(r'\d+', date_str)
        if len(numbers) >= 3:
            # 尝试多种数字组合
            combinations = [
                f"{numbers[0]}-{numbers[1]}-{numbers[2]}",
                f"{numbers[2]}-{numbers[1]}-{numbers[0]}",
                f"{numbers[2]}-{numbers[0]}-{numbers[1]}",
            ]
            for combo in combinations:
                try:
                    date_obj = datetime.strptime(combo, '%Y-%m-%d')
                    if 2000 <= date_obj.year <= 2100:  # 合理年份范围
                        return date_obj.strftime('%Y-%m-%d')
                except ValueError:
                    continue

        return None

    def extract_total_amount(self, annotation: Dict[str, Any]) -> Optional[float]:
        """
        提取总金额

        CORD 字段: total
        映射决策: 直接使用 total 字段
        """
        total = annotation.get('total')
        if total is not None:
            try:
                # 处理字符串形式的金额 (如 "$12.34", "1,234.56")
                if isinstance(total, str):
                    total = total.replace(',', '').replace('$', '').strip()
                return float(total)
            except (ValueError, TypeError):
                pass
        return None

    def extract_tax_amount(self, annotation: Dict[str, Any]) -> Optional[float]:
        """
        提取税额

        CORD 字段: tax (可选)
        映射决策: 使用 tax 字段，缺失则返回 null
        """
        tax = annotation.get('tax')
        if tax is not None:
            try:
                if isinstance(tax, str):
                    tax = tax.replace(',', '').replace('$', '').strip()
                return float(tax)
            except (ValueError, TypeError):
                pass
        return None

    def extract_tax_id(self, annotation: Dict[str, Any]) -> Optional[str]:
        """
        提取税号

        CORD 字段: 无
        映射决策: CORD 是小票数据集，不包含税号，固定返回 null
        """
        return None

    def extract_invoice_no(self, annotation: Dict[str, Any]) -> Optional[str]:
        """
        提取发票号码

        CORD 字段: 无
        映射决策: CORD 是小票数据集，不包含发票号，固定返回 null
        """
        return None

    def convert_to_unified_schema(self, annotation: Dict[str, Any]) -> Dict[str, Any]:
        """
        将 CORD 标注转换为统一的 6 字段 schema

        Args:
            annotation: 原始 CORD 标注

        Returns:
            统一 schema 的字典
        """
        return {
            'merchant_name': self.extract_merchant_name(annotation),
            'date': self.extract_date(annotation),
            'total_amount': self.extract_total_amount(annotation),
            'tax_amount': self.extract_tax_amount(annotation),
            'tax_id': self.extract_tax_id(annotation),
            'invoice_no': self.extract_invoice_no(annotation),
        }

    def process_dataset(self, split: str = 'train') -> List[Dict[str, Any]]:
        """
        处理整个数据集分割

        Args:
            split: 'train' 或 'test'

        Returns:
            处理后的样本列表，每个样本包含:
            {
                'image_path': str,
                'prompt': str,
                'target_json': dict
            }
        """
        split_dir = self.data_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"CORD {split} directory not found: {split_dir}")

        samples = []
        image_files = list(split_dir.glob('*.jpg')) + list(split_dir.glob('*.png'))

        for img_path in image_files:
            json_path = img_path.with_suffix('.json')
            if not json_path.exists():
                # 尝试其他可能的 JSON 文件名
                json_path = split_dir / f"{img_path.stem}.json"
                if not json_path.exists():
                    continue

            try:
                # 加载标注
                annotation = self.load_annotation(json_path)

                # 转换为统一 schema
                target_json = self.convert_to_unified_schema(annotation)

                # 构造样本
                sample = {
                    'image_path': str(img_path),
                    'prompt': self.get_default_prompt(),
                    'target_json': target_json,
                    'dataset': 'cord',
                    'split': split,
                }
                samples.append(sample)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

        return samples

    @staticmethod
    def get_default_prompt() -> str:
        """获取默认 prompt 模板"""
        return """<image>请从这张票据中抽取以下字段并以 JSON 输出:
merchant_name, date, total_amount, tax_amount, tax_id, invoice_no。
缺失字段填 null,不要编造。"""

    # ── HuggingFace 格式支持 ──────────────────────────────────────────

    @staticmethod
    def convert_hf_sample(hf_sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        将 HuggingFace CORD v2 样本转换为统一 schema。

        HF 样本结构:
          {
            "image": PIL.Image,
            "ground_truth": "{\"gt_parse\": {...}}"
          }

        Returns:
            {"merchant_name", "date", "total_amount", "tax_amount", "tax_id", "invoice_no"}
        """
        gt_raw = hf_sample.get("ground_truth", "{}")
        if isinstance(gt_raw, str):
            try:
                gt = json.loads(gt_raw)
            except json.JSONDecodeError:
                gt = {}
        else:
            gt = gt_raw

        gt_parse = gt.get("gt_parse", {})

        # total_amount: total.total_price
        total_price = CORDAdapter._extract_hf_field(gt_parse, "total", "total_price")

        # tax_amount: subtotal.tax_price
        tax_price = CORDAdapter._extract_hf_field(gt_parse, "subtotal", "tax_price")

        # merchant_name: 尝试多个来源
        merchant = CORDAdapter._extract_hf_merchant(gt_parse)

        from src.utils.normalize import normalize_amount, normalize_string
        return {
            "merchant_name": normalize_string(merchant),
            "date": None,
            "total_amount": normalize_amount(total_price),
            "tax_amount": normalize_amount(tax_price),
            "tax_id": None,
            "invoice_no": None,
        }

    @staticmethod
    def _extract_hf_field(gt_parse: dict, block: str, field: str) -> Optional[str]:
        """从 HF gt_parse 中提字段值。"""
        b = gt_parse.get(block, {})
        if not isinstance(b, dict):
            return None
        val = b.get(field)
        if isinstance(val, list) and len(val) > 0:
            val = val[0]
        if isinstance(val, dict):
            for sub_key in ("text", "value", "price", "amount"):
                if sub_key in val:
                    return str(val[sub_key])
            return str(list(val.values())[0]) if val else None
        return str(val).strip() if val is not None else None

    @staticmethod
    def _extract_hf_merchant(gt_parse: dict) -> Optional[str]:
        """从 HF gt_parse 中提取商户名。"""
        candidates = []
        menu = gt_parse.get("menu", [])
        for item in (menu or []):
            sub_nm = item.get("sub_nm")
            if isinstance(sub_nm, str) and sub_nm.strip():
                candidates.append(sub_nm.strip())
            elif isinstance(sub_nm, list):
                for s in sub_nm:
                    if isinstance(s, dict):
                        for sk in ("text", "value"):
                            if sk in s and s[sk]:
                                candidates.append(str(s[sk]).strip())
                    elif isinstance(s, str) and s.strip():
                        candidates.append(s.strip())
        return max(candidates, key=len) if candidates else None

    @classmethod
    def from_huggingface(cls, split: str = "train",
                         max_samples: Optional[int] = None,
                         output_dir: Optional[Path] = None
                         ) -> List[Dict[str, Any]]:
        """
        从 HuggingFace 下载 CORD v2 并转换为统一 schema。

        Args:
            split: 数据分割 ('train', 'validation', 'test')
            max_samples: 最大样本数
            output_dir: 图片保存目录（None 则保存到 data/cord/images/）

        Returns:
            转换后的样本列表 [{image_path, prompt, target_json}]
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("请安装 datasets: pip install datasets")

        ds = load_dataset("naver-clova-ix/cord-v2", split=split)
        if max_samples and len(ds) > max_samples:
            ds = ds.select(range(max_samples))

        if output_dir is None:
            output_dir = Path(__file__).resolve().parent.parent.parent.parent / "data" / "cord"
        img_dir = Path(output_dir) / "images" / split
        img_dir.mkdir(parents=True, exist_ok=True)

        samples = []
        for i, hf_sample in enumerate(ds):
            # 保存图片
            img = hf_sample.get("image")
            fname = f"cord_{i:04d}.png"
            fpath = img_dir / fname
            if hasattr(img, "save"):
                img.save(fpath)
            elif isinstance(img, bytes):
                import io
                from PIL import Image as PILImage
                PILImage.open(io.BytesIO(img)).save(fpath)

            target_json = cls.convert_hf_sample(hf_sample)
            samples.append({
                "image_path": str(fpath),
                "prompt": cls.get_default_prompt(),
                "target_json": target_json,
                "dataset": "cord",
                "split": split,
            })

        return samples


def test_adapter():
    """测试 CORD 适配器"""
    # 创建一个模拟的 CORD 标注
    mock_annotation = {
        'company': 'Walmart Store',
        'date': '2023-05-15',
        'total': 125.67,
        'tax': 10.23,
        'menu': [
            {'nm': 'Milk', 'unitprice': 3.99, 'cnt': 2, 'price': 7.98},
            {'nm': 'Bread', 'unitprice': 2.50, 'cnt': 1, 'price': 2.50},
        ]
    }

    adapter = CORDAdapter(data_root=Path('/dummy/path'))
    result = adapter.convert_to_unified_schema(mock_annotation)

    print("CORD 适配器测试:")
    print(f"输入: {mock_annotation}")
    print(f"输出: {result}")
    print()

    expected = {
        'merchant_name': 'Walmart Store',
        'date': '2023-05-15',
        'total_amount': 125.67,
        'tax_amount': 10.23,
        'tax_id': None,
        'invoice_no': None,
    }

    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ CORD 适配器本地格式测试通过")


def test_hf_adapter():
    """测试 HuggingFace 格式的转换"""
    print("\n测试 HuggingFace CORD v2 格式转换:")

    # 模拟 HF 样本
    hf_sample = {
        "image": None,
        "ground_truth": json.dumps({
            "gt_parse": {
                "menu": [
                    {"nm": "Milk", "cnt": "2", "unitprice": "5000", "price": "10000",
                     "sub_nm": "SuperMart Inc."},
                    {"nm": "Bread", "cnt": "1", "unitprice": "3000", "price": "3000"},
                ],
                "subtotal": {
                    "subtotal_price": "13000",
                    "tax_price": "1300",
                },
                "total": {
                    "total_price": "14300",
                    "cashprice": "15000",
                    "changeprice": "700",
                },
            }
        }),
    }

    result = CORDAdapter.convert_hf_sample(hf_sample)
    print(f"  merchant_name: {result['merchant_name']}")
    print(f"  total_amount:  {result['total_amount']}")
    print(f"  tax_amount:    {result['tax_amount']}")

    assert result["merchant_name"] == "SuperMart Inc.", f"Bad merchant: {result['merchant_name']}"
    assert result["total_amount"] == 14300.0, f"Bad total: {result['total_amount']}"
    assert result["tax_amount"] == 1300.0, f"Bad tax: {result['tax_amount']}"
    assert result["tax_id"] is None
    assert result["invoice_no"] is None
    print("✓ CORD 适配器 HF 格式测试通过")


if __name__ == '__main__':
    test_adapter()
    test_hf_adapter()
