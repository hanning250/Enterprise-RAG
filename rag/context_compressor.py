"""检索后上下文压缩 + 答案能力校验模块。

企业知识库场景下意图统一为 general_query：
- AnswerabilityFilter：通用路径不做意图降权（原样返回候选）
- ContextCompressor：不做旅游域字段裁剪；仅在开启时对过长片段做 token 截断

二者仍挂在 Hybrid → Expand → Compress 链路上，便于日后扩展企业意图规则。
"""

from __future__ import annotations

from typing import Optional

from langchain_core.documents import Document

from rag.context_budget import estimate_tokens
from rag.query_intent import QueryIntentResult, classify, GENERAL
from utils.config_handler import rag_conf
from utils.logger_handler import logger


def _resolve_config() -> dict:
    cfg = rag_conf.get("post_retrieval", {}) or {}
    return {
        "answerability_enabled": bool(cfg.get("answerability_enabled", True)),
        "answerability_fail_penalty": float(cfg.get("answerability_fail_penalty", 0.12)),
        "compressor_enabled": bool(cfg.get("compressor_enabled", True)),
        "compressor_max_kept_tokens": int(cfg.get("compressor_max_kept_tokens", 1200)),
        "compressor_fallback_ratio": float(cfg.get("compressor_fallback_ratio", 0.35)),
    }


class AnswerabilityFilter:
    """按 query 意图做候选的回答价值检查（规则驱动，零模型）。

    当前仅 general_query：直接放行。保留接口供 HybridRetriever 调用。
    """

    def __init__(self):
        self.cfg = _resolve_config()

    def filter(
        self,
        query: str,
        candidates: list["RetrievalCandidate"],  # type: ignore[name-defined]
        *,
        intent_result: Optional[QueryIntentResult] = None,
    ) -> list["RetrievalCandidate"]:  # type: ignore[name-defined]
        if not self.cfg["answerability_enabled"] or not candidates:
            return list(candidates)

        intent = intent_result if intent_result is not None else classify(query)
        if intent.intent == GENERAL:
            return list(candidates)

        # 预留：非 general 意图时可在此按规则降权
        logger.debug(
            f"[Answerability] intent={intent.intent} 尚无企业规则，原样放行 "
            f"{len(candidates)} 条候选"
        )
        return list(candidates)


class ContextCompressor:
    """检索后上下文收紧：企业场景下不做旅游字段裁剪，仅做超长截断。"""

    def __init__(self):
        self.cfg = _resolve_config()

    def compress(
        self,
        query: str,
        docs: list[Document],
        *,
        intent_result: Optional[QueryIntentResult] = None,
    ) -> list[Document]:
        if not self.cfg["compressor_enabled"] or not docs:
            return list(docs)

        _ = query
        _ = intent_result
        max_tokens = self.cfg["compressor_max_kept_tokens"]
        if not max_tokens or max_tokens <= 0:
            return list(docs)

        result: list[Document] = []
        truncated = 0
        for doc in docs:
            raw = doc.page_content or ""
            if estimate_tokens(raw) <= max_tokens:
                result.append(doc)
                continue
            meta = dict(doc.metadata or {})
            compressed = self._truncate_tokens(raw, max_tokens)
            meta["_context_compressed"] = True
            meta["_context_compressed_intent"] = GENERAL
            meta["_context_compressed_saved_chars"] = max(0, len(raw) - len(compressed))
            truncated += 1
            result.append(Document(page_content=compressed, metadata=meta))

        if truncated:
            logger.info(
                f"[ContextCompressor] 通用截断：{truncated}/{len(docs)} 段超过 "
                f"{max_tokens} tokens，已裁剪"
            )
        return result

    @staticmethod
    def _truncate_tokens(text: str, max_tokens: int) -> str:
        if not text or estimate_tokens(text) <= max_tokens:
            return text or ""
        lines = text.split("\n")
        kept_lines: list[str] = []
        budget = max_tokens
        for line in lines:
            t = estimate_tokens(line)
            if budget - t >= 0:
                kept_lines.append(line)
                budget -= t
            else:
                break
        return "\n".join(kept_lines).strip()
