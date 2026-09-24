#!/usr/bin/env python
"""检索指标：nDCG@k、Recall@k、MRR。解析式与视觉式共用，保证两边算的是同一个数。

写这个文件时刻意做了四象限自检（见 self_test）——receipt-vlm 里
eval.py 有过一个严重 bug：抽错的值只记 FN 不记 FP，精确率恒等于 100%。
检索指标同样容易犯这种错，所以这里的每个函数都有已知答案的用例兜着。
"""
import numpy as np


def dcg(rels: np.ndarray) -> float:
    """rels 按检索排名顺序给出的相关性分数。"""
    return float(np.sum(rels / np.log2(np.arange(2, len(rels) + 2))))


def ndcg_at_k(ranked_ids, relevant_ids, k: int = 5) -> float:
    """ranked_ids: 检索返回的文档 id，按得分降序。relevant_ids: 真值集合。"""
    rel_set = set(relevant_ids)
    if not rel_set:
        return 0.0
    gains = np.array([1.0 if d in rel_set else 0.0 for d in ranked_ids[:k]])
    ideal = np.array([1.0] * min(len(rel_set), k))
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_ids, relevant_ids, k: int = 1) -> float:
    rel_set = set(relevant_ids)
    if not rel_set:
        return 0.0
    hit = len(rel_set & set(ranked_ids[:k]))
    return hit / len(rel_set)


def mrr(ranked_ids, relevant_ids) -> float:
    rel_set = set(relevant_ids)
    for i, d in enumerate(ranked_ids, start=1):
        if d in rel_set:
            return 1.0 / i
    return 0.0


def self_test():
    """已知答案的用例。改动上面任何函数后必须先跑通这里。"""
    # 命中在第 1 位
    assert recall_at_k(["a", "b", "c"], ["a"], k=1) == 1.0
    assert ndcg_at_k(["a", "b", "c"], ["a"], k=5) == 1.0
    assert mrr(["a", "b", "c"], ["a"]) == 1.0
    # 命中在第 3 位：recall@1 应为 0，不能因为「最终找到了」就算对
    assert recall_at_k(["x", "y", "a"], ["a"], k=1) == 0.0
    assert abs(mrr(["x", "y", "a"], ["a"]) - 1 / 3) < 1e-9
    assert abs(ndcg_at_k(["x", "y", "a"], ["a"], k=5) - 0.5) < 1e-9  # 1/log2(4)
    # 完全没命中
    assert recall_at_k(["x", "y"], ["a"], k=2) == 0.0
    assert ndcg_at_k(["x", "y"], ["a"], k=2) == 0.0
    assert mrr(["x", "y"], ["a"]) == 0.0
    # 多个真值，只召回一半
    assert recall_at_k(["a", "x"], ["a", "b"], k=2) == 0.5
    # 空真值不应该悄悄算成满分
    assert ndcg_at_k(["a"], [], k=5) == 0.0
    print("  metrics self_test 全部通过")


if __name__ == "__main__":
    self_test()
