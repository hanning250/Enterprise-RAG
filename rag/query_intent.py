"""Query 意图分类模块（规则驱动，零模型调用）。

当前为企业内部知识库场景：不再沿用旅游域意图（门票/住宿/美食等）。
统一返回 general_query，供 Answerability / ContextCompressor 走通用路径。
后续若要加企业意图（工资/制度/报销等），在此扩展规则表即可。
"""

from __future__ import annotations

from dataclasses import dataclass

IntentName = str

GENERAL: IntentName = "general_query"


@dataclass(frozen=True)
class QueryIntentResult:
    intent: IntentName
    scores: dict[IntentName, int]


def classify(query: str) -> QueryIntentResult:
    """对企业知识库 query 统一归为 general_query。"""
    _ = query  # 保留签名，便于后续接规则表
    return QueryIntentResult(intent=GENERAL, scores={GENERAL: 0})
