"""上下文预算模块。"""

from __future__ import annotations

import os
from typing import List, Tuple
from langchain_core.documents import Document


def _count_ascii_and_other(text: str) -> Tuple[int, int]:
    """统计文本中 ASCII 和非 ASCII 字符的数量。"""
    ascii_count = 0
    other_count = 0
    for ch in text:
        if ord(ch) < 128:
            ascii_count += 1
        else:
            other_count += 1
    return ascii_count, other_count


def estimate_tokens(text: str) -> int:
    """估算文本占用的 token 数（轻量级方案）。"""
    if not text:
        return 0

    ascii_count, other_count = _count_ascii_and_other(text)

    ascii_tokens = ascii_count / 4.0
    other_tokens = other_count / 1.5

    import math
    total_tokens = math.ceil(ascii_tokens + other_tokens)

    return total_tokens


def _format_citation(meta: dict, counter: int) -> str:
    """把 metadata 格式化成「资料N: 文件名 第X页/第X-Y页/章节X」的人可读引用。

    优先展示：
      1. 相对项目根的文件名（去掉前面的 data/ 前缀）
      2. page：若有 page_start/page_end 显示「第X-Y页」；否则显示单页 page
      3. section_title / section_path：Markdown 分章节级检索能展示面包屑
    """
    if not meta:
        return f"资料{counter}"

    raw_source = (
        meta.get("source_path")
        or meta.get("source")
        or ""
    )
    filename = os.path.basename(str(raw_source)) if raw_source else "未标注来源"

    p_start = meta.get("page_start")
    p_end = meta.get("page_end")
    page_single = meta.get("page")
    if p_start is not None and p_end is not None and p_start != p_end:
        page_part = f" 第{p_start}-{p_end}页"
    elif page_single is not None:
        page_part = f" 第{page_single}页"
    else:
        page_part = ""

    section_title = meta.get("section_title")
    if section_title:
        section_part = f"（章节：{section_title}）"
    else:
        section_part = ""

    return f"资料{counter}: {filename}{page_part}{section_part}"


def format_doc_context_block(meta: dict, counter: int, page_content: str) -> str:
    """单条参考资料的 prompt 块：仅 citation + 正文，不含 metadata dump。"""
    citation = _format_citation(meta or {}, counter)
    content = (page_content or "").strip()
    return f"【{citation}】: {content}\n"


def build_context_text(
    docs: List[Document],
    max_context_tokens: int | None = None,
) -> Tuple[str, int, int]:
    """拼装 LLM 上下文：有预算则裁剪，否则纳入全部 docs。

    Returns
    -------
    context, used_tokens, included_count
    """
    if not docs:
        return "", 0, 0

    if max_context_tokens is not None and max_context_tokens > 0:
        return build_context_with_budget(docs, max_context_tokens)

    context = ""
    for i, doc in enumerate(docs, start=1):
        context += format_doc_context_block(
            doc.metadata or {},
            i,
            doc.page_content or "",
        )
    used = estimate_tokens(context)
    return context, used, len(docs)


def build_context_with_budget(
    docs: List[Document],
    max_context_tokens: int,
) -> Tuple[str, int, int]:
    """按照 token 预算把检索到的文档拼装成上下文文本（citation + 正文）。"""
    if not docs:
        return "", 0, 0

    if max_context_tokens <= 0:
        return "", 0, 0

    context = ""
    used_tokens = 0
    included_count = 0
    counter = 0

    for doc in docs:
        counter += 1
        doc_str = format_doc_context_block(
            doc.metadata or {},
            counter,
            doc.page_content or "",
        )
        doc_tokens = estimate_tokens(doc_str)

        if used_tokens + doc_tokens <= max_context_tokens:
            context += doc_str
            used_tokens += doc_tokens
            included_count += 1
        else:
            break

    return context, used_tokens, included_count
