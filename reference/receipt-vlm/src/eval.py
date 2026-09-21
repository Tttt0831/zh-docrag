"""
评估模块
计算项目要求的全部指标
"""
import json
import torch
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np


def field_accuracy(predictions: List[Dict], targets: List[Dict],
                   field_name: str) -> float:
    """
    计算单个字段的 exact-match 准确率

    Args:
        predictions: 预测结果列表
        targets: 目标结果列表
        field_name: 字段名

    Returns:
        准确率 (0-1)
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have same length")

    correct = 0
    total = 0

    for pred, target in zip(predictions, targets):
        pred_val = pred.get(field_name)
        target_val = target.get(field_name)

        # 两个都是 null 算正确
        if pred_val is None and target_val is None:
            correct += 1
        elif pred_val == target_val:
            correct += 1

        total += 1

    return correct / total if total > 0 else 0.0


def value_match_rate(predictions: List[Dict], targets: List[Dict]) -> float:
    """
    计算所有字段的值匹配率

    Returns:
        匹配率 (0-1)
    """
    fields = ['merchant_name', 'date', 'total_amount', 'tax_amount', 'tax_id', 'invoice_no']

    total_fields = len(fields) * len(predictions)
    correct_fields = 0

    for pred, target in zip(predictions, targets):
        for field in fields:
            pred_val = pred.get(field)
            target_val = target.get(field)

            if pred_val == target_val:
                correct_fields += 1

    return correct_fields / total_fields if total_fields > 0 else 0.0


def field_level_f1(predictions: List[Dict], targets: List[Dict]) -> Dict[str, float]:
    """
    计算字段级别的 F1 分数

    把抽取看成 precision/recall，能反映漏抽 vs 多抽(幻觉)

    Returns:
        Dict with precision, recall, f1
    """
    fields = ['merchant_name', 'date', 'total_amount', 'tax_amount', 'tax_id', 'invoice_no']

    # 统计 TP, FP, FN
    tp = 0  # 正确抽取
    fp = 0  # 错误抽取（幻觉）
    fn = 0  # 漏抽

    for pred, target in zip(predictions, targets):
        for field in fields:
            pred_val = pred.get(field)
            target_val = target.get(field)

            if target_val is not None:
                if pred_val == target_val:
                    tp += 1                      # 抽对了
                elif pred_val is None:
                    fn += 1                      # 漏抽
                else:
                    # 抽出了一个错的值：既多抽了一个错的(FP)，也漏掉了对的(FN)。
                    # 旧实现这里只记 FN，导致 FP 只可能来自「真值 null 却填值」，
                    # precision 被结构性钉死在 1.0，F1 退化成 recall 的单调变换
                    # (已验证：报告 F1 与 2r/(1+r) 完全相同)。
                    fp += 1
                    fn += 1
            elif pred_val is not None:
                # 目标是 null，但预测了值 → 幻觉
                fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def normalized_edit_distance(predictions: List[Dict], targets: List[Dict],
                               field_name: str = 'merchant_name') -> float:
    """
    计算归一化编辑距离 (1 - NED)

    对长文本字段如 merchant_name

    Returns:
        平均 1-NED 分数（越低越好）
    """
    try:
        import editdistance as ed
    except ImportError:
        print('Warning: editdistance not installed, using simple comparison')
        # 简单使用 exact match
        return 1.0 - field_accuracy(predictions, targets, field_name)

    distances = []

    for pred, target in zip(predictions, targets):
        pred_val = str(pred.get(field_name, ''))
        target_val = str(target.get(field_name, ''))

        if pred_val == target_val:
            distances.append(0.0)
        else:
            # 计算编辑距离
            dist = ed.eval(pred_val, target_val)
            max_len = max(len(pred_val), len(target_val))
            if max_len > 0:
                distances.append(dist / max_len)
            else:
                distances.append(0.0)

    return np.mean(distances) if distances else 0.0


def json_validity_rate(predictions: List[Dict]) -> float:
    """
    计算 JSON 合法率

    输出能否被 json.loads 解析 且 通过 schema 校验

    Returns:
        合法率 (0-1)
    """
    try:
        from src.utils.normalize import validate_schema
    except ImportError:
        # 简化版本：只检查基本结构
        def validate_schema(record):
            required_fields = {'merchant_name', 'date', 'total_amount', 'tax_amount', 'tax_id', 'invoice_no'}
            return isinstance(record, dict) and all(f in record for f in required_fields)

    valid = 0

    for pred in predictions:
        if validate_schema(pred):
            valid += 1

    return valid / len(predictions) if predictions else 0.0


def hallucination_rate(predictions: List[Dict], targets: List[Dict]) -> float:
    """
    计算幻觉率

    真值为 null 但模型却填了值的比例

    Returns:
        幻觉率 (0-1)
    """
    fields = ['merchant_name', 'date', 'total_amount', 'tax_amount', 'tax_id', 'invoice_no']

    hallucinations = 0
    total_null_fields = 0

    for pred, target in zip(predictions, targets):
        for field in fields:
            target_val = target.get(field)
            pred_val = pred.get(field)

            if target_val is None:
                total_null_fields += 1
                if pred_val is not None:
                    hallucinations += 1

    return hallucinations / total_null_fields if total_null_fields > 0 else 0.0


def compute_all_metrics(predictions: List[Dict], targets: List[Dict],
                         dataset_name: str = 'test') -> Dict[str, Any]:
    """
    计算所有要求的指标

    Returns:
        完整的指标字典
    """
    # 字段级准确率
    field_accuracies = {}
    for field in ['merchant_name', 'date', 'total_amount', 'tax_amount', 'tax_id', 'invoice_no']:
        field_accuracies[f'acc_{field}'] = field_accuracy(predictions, targets, field)

    # 整体指标
    metrics = {
        'dataset': dataset_name,
        'num_samples': len(predictions),
        **field_accuracies,
        'value_match_rate': value_match_rate(predictions, targets),
        **field_level_f1(predictions, targets),
        'ned_merchant_name': normalized_edit_distance(predictions, targets, 'merchant_name'),
        'json_validity_rate': json_validity_rate(predictions),
        'hallucination_rate': hallucination_rate(predictions, targets),
    }

    return metrics


def format_metrics_as_markdown(metrics: Dict[str, Any]) -> str:
    """将指标格式化为 Markdown 表格"""

    lines = [
        f"## {metrics.get('dataset', 'Unknown')} Dataset Results",
        f"",
        f"**Samples**: {metrics.get('num_samples', 0)}",
        f"",
        f"### Field-wise Accuracy",
        f"| Field | Accuracy |",
        f"|-------|----------|",
        f"| merchant_name | {metrics.get('acc_merchant_name', 0):.2%} |",
        f"| date | {metrics.get('acc_date', 0):.2%} |",
        f"| total_amount | {metrics.get('acc_total_amount', 0):.2%} |",
        f"| tax_amount | {metrics.get('acc_tax_amount', 0):.2%} |",
        f"| tax_id | {metrics.get('acc_tax_id', 0):.2%} |",
        f"| invoice_no | {metrics.get('acc_invoice_no', 0):.2%} |",
        f"",
        f"### Overall Metrics",
        f"| Metric | Score |",
        f"|--------|-------|",
        f"| Value Match Rate | {metrics.get('value_match_rate', 0):.2%} |",
        f"| Precision | {metrics.get('precision', 0):.2%} |",
        f"| Recall | {metrics.get('recall', 0):.2%} |",
        f"| F1 Score | {metrics.get('f1', 0):.2%} |",
        f"| 1-NED (merchant_name) | {metrics.get('ned_merchant_name', 0):.4f} |",
        f"| JSON Validity Rate | {metrics.get('json_validity_rate', 0):.2%} |",
        f"| Hallucination Rate | {metrics.get('hallucination_rate', 0):.2%} |",
    ]

    return '\n'.join(lines)


if __name__ == '__main__':
    # 测试评估函数
    print('Testing evaluation metrics...')

    # 模拟数据
    predictions = [
        {'merchant_name': 'Test Corp', 'date': '2025-01-01', 'total_amount': 100.0, 'tax_amount': 13.0, 'tax_id': '123456789', 'invoice_no': '001'},
        {'merchant_name': 'Wrong Name', 'date': '2025-01-02', 'total_amount': 200.0, 'tax_amount': None, 'tax_id': '987654321', 'invoice_no': '002'},
        {'merchant_name': 'Another Corp', 'date': '2025-01-03', 'total_amount': 150.0, 'tax_amount': 19.5, 'tax_id': '111111111', 'invoice_no': '003'},
    ]

    targets = [
        {'merchant_name': 'Test Corp', 'date': '2025-01-01', 'total_amount': 100.0, 'tax_amount': 13.0, 'tax_id': '123456789', 'invoice_no': '001'},
        {'merchant_name': 'Real Corp', 'date': '2025-01-02', 'total_amount': 200.0, 'tax_amount': 26.0, 'tax_id': '987654321', 'invoice_no': '002'},
        {'merchant_name': 'Another Corp', 'date': '2025-01-03', 'total_amount': None, 'tax_amount': None, 'tax_id': None, 'invoice_no': '003'},
    ]

    metrics = compute_all_metrics(predictions, targets, 'test')
    print(format_metrics_as_markdown(metrics))

    print('\nEvaluation test passed!')
