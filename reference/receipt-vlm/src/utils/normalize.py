"""
字段归一化工具函数
在 eval 和数据构造里都要用同一套
"""
import re
import json
from typing import Optional, Dict, Any
from datetime import datetime


def normalize_date(date_str: Optional[str]) -> Optional[str]:
    """
    将各种日期格式统一为 YYYY-MM-DD
    支持: 2025/1/1, 2025年1月1日, 01-Jan-2025, 2025-01-01 等
    """
    if date_str is None or not isinstance(date_str, str):
        return None

    date_str = date_str.strip()
    if not date_str:
        return None

    # 已经是标准格式
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return date_str

    # 尝试各种格式
    formats_to_try = [
        '%Y/%m/%d',
        '%Y/%d/%m',
        '%m/%d/%Y',
        '%d/%m/%Y',
        '%Y.%m.%d',
        '%d.%m.%Y',
        '%m.%d.%Y',
        '%Y年%m月%d日',
        '%Y年%m月%d',
        '%m/%d/%y',
        '%d/%m/%y',
        '%Y-%m-%d',
        '%d-%m-%Y',
        '%m-%d-%Y',
        '%d-%b-%Y',
        '%d-%B-%Y',
        '%b %d, %Y',
        '%B %d, %Y',
        '%d %b %Y',
        '%d %B %Y',
    ]

    for fmt in formats_to_try:
        try:
            parsed = datetime.strptime(date_str, fmt)
            return parsed.strftime('%Y-%m-%d')
        except ValueError:
            continue

    # 尝试从字符串中提取日期模式
    date_patterns = [
        r'(\d{4})[/-年](\d{1,2})[/-月](\d{1,2})',
        r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',
        r'(\d{1,2})月(\d{1,2})日(\d{4})',  # 5月15日2023
        r'(\d{1,2})月(\d{1,2})日[，,](\d{4})',  # 5月15日，2023
    ]

    # 先检查纯数字格式 YYYYMMDD
    if re.match(r'^\d{8}$', date_str):
        try:
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            if 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
                return f'{year:04d}-{month:02d}-{day:02d}'
        except ValueError:
            pass

    for pattern in date_patterns:
        match = re.search(pattern, date_str)
        if match:
            groups = match.groups()
            try:
                if len(groups[0]) == 4:  # 年份在前
                    year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                else:  # 年份在后
                    month, day, year = int(groups[0]), int(groups[1]), int(groups[2])

                # 处理两位年份
                if year < 100:
                    year += 2000 if year < 50 else 1900

                if 1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2100:
                    return f'{year:04d}-{month:02d}-{day:02d}'
            except (ValueError, IndexError):
                continue

    return None


def normalize_amount(amount: Any) -> Optional[float]:
    """
    将金额字符串归一化为 float，保留 2 位小数
    去掉货币符号、千分位逗号
    """
    if amount is None:
        return None

    # 如果已经是数字
    if isinstance(amount, (int, float)):
        return round(float(amount), 2)

    if not isinstance(amount, str):
        return None

    amount_str = amount.strip()
    if not amount_str or amount_str.lower() in ['null', 'none', 'n/a', '-']:
        return None

    # 去除货币符号和多余字符
    amount_str = re.sub(r'[¥$€£¢₹₩₽円￠]', '', amount_str)
    amount_str = amount_str.replace(',', '')
    amount_str = amount_str.replace(' ', '')

    try:
        return round(float(amount_str), 2)
    except ValueError:
        return None


def normalize_tax_id(tax_id: Optional[str]) -> Optional[str]:
    """
    税号归一化：去空格、统一为小写
    """
    if tax_id is None or not isinstance(tax_id, str):
        return None

    normalized = tax_id.strip().replace(' ', '').replace('-', '').lower()
    return normalized if normalized else None


def normalize_string(s: Optional[str]) -> Optional[str]:
    """
    字符串字段归一化：strip + 全角转半角 + 去多余空白
    """
    if s is None or not isinstance(s, str):
        return None

    # 全角标点符号映射表
    fullwidth_to_halfwidth = {
        '，': ',',  # U+FF0C
        '、': ',',  # U+3001 (顿号映射为逗号)
        '：': ':',  # U+FF1A
        '；': ';',  # U+FF1B
        '！': '!',  # U+FF01
        '？': '?',  # U+FF1F
        '（': '(',  # U+FF08
        '）': ')',  # U+FF09
        '【': '[',  # U+3010
        '】': ']',  # U+3011
        '「': '"',  # U+300C
        '」': '"',  # U+300D
        '『': "'",  # U+300E
        '』': "'",  # U+300F
    }

    # 全角转半角
    normalized = ''
    for char in s:
        # 先查映射表
        if char in fullwidth_to_halfwidth:
            normalized += fullwidth_to_halfwidth[char]
            continue

        code = ord(char)
        # 全角字符范围 (ASCII 对应的全角)
        if 0xFF01 <= code <= 0xFF5E:
            normalized += chr(code - 0xFEE0)
        elif code == 0x3000:  # 全角空格
            normalized += ' '
        else:
            normalized += char

    # 去多余空白
    normalized = ' '.join(normalized.split())
    return normalized if normalized else None


def normalize_invoice_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    对整条发票记录进行归一化
    """
    return {
        'merchant_name': normalize_string(record.get('merchant_name')),
        'date': normalize_date(record.get('date')),
        'total_amount': normalize_amount(record.get('total_amount')),
        'tax_amount': normalize_amount(record.get('tax_amount')),
        'tax_id': normalize_tax_id(record.get('tax_id')),
        'invoice_no': normalize_string(record.get('invoice_no')),
    }


def validate_schema(record: Dict[str, Any]) -> bool:
    """
    验证记录是否符合 schema
    """
    required_fields = {'merchant_name', 'date', 'total_amount', 'tax_amount', 'tax_id', 'invoice_no'}

    if not isinstance(record, dict):
        return False

    if not all(field in record for field in required_fields):
        return False

    for field in required_fields:
        value = record[field]

        # null 是合法的
        if value is None:
            continue

        # 类型检查
        if field == 'date':
            if not isinstance(value, str) or not re.match(r'^\d{4}-\d{2}-\d{2}$', value):
                return False
        elif field in ['total_amount', 'tax_amount']:
            if not isinstance(value, (int, float)):
                return False
        else:  # merchant_name, tax_id, invoice_no
            if not isinstance(value, str):
                return False

    return True


if __name__ == '__main__':
    # 测试归一化函数
    print("Testing normalization functions:")

    # 日期测试
    test_dates = ['2025/1/1', '2025年1月1日', '01-Jan-2025', '2025-01-01', None]
    for d in test_dates:
        print(f"  Date: {d!r} -> {normalize_date(d)!r}")

    # 金额测试
    test_amounts = ['¥1,234.56', '$1,234.56', '1234.56', None, 'N/A']
    for a in test_amounts:
        print(f"  Amount: {a!r} -> {normalize_amount(a)!r}")

    # 字符串测试
    test_strings = ['Ｔｅｓｔ　Ｓｔｒｉｎｇ', '  Test   String  ', None]
    for s in test_strings:
        print(f"  String: {s!r} -> {normalize_string(s)!r}")

    # 税号测试
    test_tax_ids = ['12345678901234', ' 1234-5678-9012-34 ', None]
    for t in test_tax_ids:
        print(f"  Tax ID: {t!r} -> {normalize_tax_id(t)!r}")
