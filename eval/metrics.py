"""RAG 检索评估指标模块。

提供经典的检索评估指标，全部基于召回结果列表与期望来源的匹配计算。

支持的指标：
    - Hit@K     : 前 K 条结果中是否至少命中一条正确来源（0/1）
    - MRR@K     : 第一条正确来源的排名倒数（排名越靠前分数越高）
    - Recall@K  : 前 K 条结果中覆盖了多少比例的正确来源

所有函数都是纯函数，输入是「召回来源列表」和「期望来源集合」，
输出是 0.0~1.0 之间的浮点数。不做任何 IO，方便单元测试。

来源匹配规则：
    - 每条召回结果的元信息里取 source_path + page（如果有 page 的话）
    - 期望来源是 (source_path, page) 元组集合
    - page 为 None 时表示「不关心页码，同一文件就算命中」
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def _normalize_source(source: dict[str, Any] | str) -> tuple[str, int | None]:
    """把召回结果的元信息统一成 (source_path, page) 元组。

    支持两种输入格式：
        1. 字典：从 metadata 里取 source_path 和 page
        2. 字符串：直接就是 source_path，page 为 None
    """
    if isinstance(source, str):
        return source, None
    source_path = str(source.get("source_path") or source.get("source") or "")
    page = source.get("page")
    if page is not None:
        try:
            page = int(page)
        except (ValueError, TypeError):
            page = None
    return source_path, page


def _matches(
    retrieved_source: tuple[str, int | None],
    expected_source: tuple[str, int | None],
) -> bool:
    """判断一条召回结果是否匹配一条期望来源。

    匹配规则：
        - source_path 必须完全相同
        - 如果 expected_source 的 page 是 None，表示「不关心页码」，只看文件
        - 否则 page 也必须相等
    """
    if retrieved_source[0] != expected_source[0]:
        return False
    if expected_source[1] is None:
        return True
    return retrieved_source[1] == expected_source[1]


def hit_at_k(
    retrieved: Sequence[dict[str, Any] | str],
    expected: Sequence[tuple[str, int | None] | str],
    k: int,
) -> float:
    """计算 Hit@K：前 K 条结果中是否至少命中一条正确来源。

    返回 1.0（命中）或 0.0（未命中）。

    :param retrieved: 召回结果列表，按相关性从高到低排序
    :param expected:  期望命中的来源列表
    :param k:         取前 K 条结果计算
    :return:          0.0 或 1.0
    """
    if k <= 0 or not expected:
        return 0.0

    expected_set = [
        (s, None) if isinstance(s, str) else s for s in expected
    ]

    for item in retrieved[:k]:
        src = _normalize_source(item)
        for exp in expected_set:
            if _matches(src, exp):
                return 1.0
    return 0.0


def mrr_at_k(
    retrieved: Sequence[dict[str, Any] | str],
    expected: Sequence[tuple[str, int | None] | str],
    k: int,
) -> float:
    """计算 MRR@K（Mean Reciprocal Rank）：第一条正确来源的排名倒数。

    排名从 1 开始。如果第 1 条就命中 → 1.0；第 3 条才命中 → 0.33；
    前 K 条都没命中 → 0.0。

    :param retrieved: 召回结果列表，按相关性从高到低排序
    :param expected:  期望命中的来源列表
    :param k:         只看前 K 条
    :return:          0.0 ~ 1.0 之间的浮点数
    """
    if k <= 0 or not expected:
        return 0.0

    expected_set = [
        (s, None) if isinstance(s, str) else s for s in expected
    ]

    for rank, item in enumerate(retrieved[:k], start=1):
        src = _normalize_source(item)
        for exp in expected_set:
            if _matches(src, exp):
                return 1.0 / rank
    return 0.0


def recall_at_k(
    retrieved: Sequence[dict[str, Any] | str],
    expected: Sequence[tuple[str, int | None] | str],
    k: int,
) -> float:
    """计算 Recall@K：前 K 条结果覆盖了多少比例的期望来源。

    例如期望有 4 条正确来源，前 K 条命中了 2 条 → 0.5。

    :param retrieved: 召回结果列表，按相关性从高到低排序
    :param expected:  期望命中的来源列表
    :param k:         只看前 K 条
    :return:          0.0 ~ 1.0 之间的浮点数
    """
    if k <= 0 or not expected:
        return 0.0

    expected_list = [
        (s, None) if isinstance(s, str) else s for s in expected
    ]

    hit_indices: set[int] = set()
    for item in retrieved[:k]:
        src = _normalize_source(item)
        for idx, exp in enumerate(expected_list):
            if idx in hit_indices:
                continue
            if _matches(src, exp):
                hit_indices.add(idx)
                break

    return len(hit_indices) / len(expected_list)


def compute_all_retrieval_metrics(
    retrieved: Sequence[dict[str, Any] | str],
    expected: Sequence[tuple[str, int | None] | str],
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float]:
    """一次性计算所有检索指标，返回字典便于批量写入报告。

    :param retrieved: 召回结果列表
    :param expected:  期望命中的来源列表
    :param ks:        要计算的 K 值列表，默认 (1, 3, 5)
    :return:          { "hit_at_3": 0.6, "mrr_at_5": 0.42, ... } 形式的字典
    """
    result: dict[str, float] = {}
    for k in ks:
        result[f"hit_at_{k}"] = hit_at_k(retrieved, expected, k)
        result[f"mrr_at_{k}"] = mrr_at_k(retrieved, expected, k)
        result[f"recall_at_{k}"] = recall_at_k(retrieved, expected, k)
    return result


def average_metrics(results: list[dict[str, float]]) -> dict[str, float]:
    """对多条样本的指标做均值，得到整体评估结果。

    :param results: 每条样本的指标字典列表
    :return:        各指标的平均值字典
    """
    if not results:
        return {}
    keys = results[0].keys()
    return {key: sum(r[key] for r in results) / len(results) for key in keys}
