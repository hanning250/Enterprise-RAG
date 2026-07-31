"""相关性阈值过滤与兜底模块。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rag.types import RetrievalCandidate
from utils.config_handler import rag_conf
from utils.logger_handler import logger

FallbackStrategy = str
FALLBACK_EMPTY_SAFE: FallbackStrategy = "empty_safe"
FALLBACK_FALLBACK_RRF: FallbackStrategy = "fallback_rrf"


def _resolve_score_guard_config() -> dict:
    """从 rag_conf 中解析 ScoreGuard 相关配置。"""
    rerank_cfg = rag_conf.get("rerank", {}) or {}
    retrieval_cfg = rag_conf.get("retrieval", {}) or {}
    guard_cfg = rag_conf.get("score_guard", {}) or {}

    min_score = float(
        guard_cfg.get("min_score")
        or rerank_cfg.get("min_score")
        or 0.35
    )

    final_top_k = int(
        retrieval_cfg.get("final_top_k")
        or rerank_cfg.get("top_k")
        or 6
    )

    strategy = str(
        guard_cfg.get("strategy")
        or FALLBACK_EMPTY_SAFE
    ).lower()
    if strategy not in (FALLBACK_EMPTY_SAFE, FALLBACK_FALLBACK_RRF):
        logger.warning(
            f"[ScoreGuard] 未知兜底策略 strategy={strategy}，回退到 empty_safe"
        )
        strategy = FALLBACK_EMPTY_SAFE

    fallback_count = int(guard_cfg.get("fallback_count", 3))
    if fallback_count < 1:
        fallback_count = 3

    # 二次检索（auto_refine）配置：min_candidates_after_filter 不足该值时，
    # HybridRetriever 会自动扩大 top_k（×expansion_factor 再跑一次
    min_candidates = int(guard_cfg.get("min_candidates_after_filter", 2))
    if min_candidates < 0:
        min_candidates = 2
    expansion_factor = float(guard_cfg.get("second_pass_expansion_factor", 2.0))
    if expansion_factor < 1.0:
        expansion_factor = 1.0
    second_pass_relax_min_score = float(
        guard_cfg.get("second_pass_relax_min_score_factor", 0.8)
    )
    if not (0 < second_pass_relax_min_score <= 1.0):
        second_pass_relax_min_score = 0.8

    return {
        "min_score": min_score,
        "final_top_k": final_top_k,
        "strategy": strategy,
        "fallback_count": fallback_count,
        "min_candidates_after_filter": min_candidates,
        "second_pass_expansion_factor": expansion_factor,
        "second_pass_relax_min_score_factor": second_pass_relax_min_score,
    }


@dataclass
class ScoreGuardResult:
    """ScoreGuard 的输出包装类。"""
    candidates: list[RetrievalCandidate]
    filtered_count: int = 0
    used_fallback: bool = False
    fallback_reason: str = ""

    def to_log_line(self) -> str:
        """把结果格式化为一行日志文本。"""
        msg = (
            f"最终输出候选 {len(self.candidates)} 条，"
            f"过滤 {self.filtered_count} 条弱相关；"
        )
        if self.used_fallback:
            msg += f"触发兜底: {self.fallback_reason}"
        else:
            msg += "未触发兜底"
        return msg


class ScoreGuard:
    """相关性阈值卫士。"""

    def __init__(
        self,
        *,
        min_score: Optional[float] = None,
        final_top_k: Optional[int] = None,
        strategy: Optional[FallbackStrategy] = None,
        fallback_count: Optional[int] = None,
        min_candidates_after_filter: Optional[int] = None,
        second_pass_expansion_factor: Optional[float] = None,
        second_pass_relax_min_score_factor: Optional[float] = None,
    ):
        """初始化 ScoreGuard。"""
        cfg = _resolve_score_guard_config()
        self.min_score: float = float(min_score) if min_score is not None else cfg["min_score"]
        self.final_top_k: int = int(final_top_k) if final_top_k is not None else cfg["final_top_k"]
        self.strategy: FallbackStrategy = strategy or cfg["strategy"]
        self.fallback_count: int = (
            int(fallback_count) if fallback_count is not None else cfg["fallback_count"]
        )
        # 二次检索相关：由 HybridRetriever 读取这些配置来决定要不要再跑一轮
        self.min_candidates_after_filter: int = (
            int(min_candidates_after_filter)
            if min_candidates_after_filter is not None
            else cfg["min_candidates_after_filter"]
        )
        self.second_pass_expansion_factor: float = (
            float(second_pass_expansion_factor)
            if second_pass_expansion_factor is not None
            else cfg["second_pass_expansion_factor"]
        )
        self.second_pass_relax_min_score_factor: float = (
            float(second_pass_relax_min_score_factor)
            if second_pass_relax_min_score_factor is not None
            else cfg["second_pass_relax_min_score_factor"]
        )

        if self.final_top_k < 1:
            logger.warning(f"[ScoreGuard] final_top_k={self.final_top_k} 非法，已修正为 6")
            self.final_top_k = 6
        if self.fallback_count < 1:
            self.fallback_count = 3
        if self.min_candidates_after_filter < 0:
            self.min_candidates_after_filter = 0
        if self.second_pass_expansion_factor < 1.0:
            self.second_pass_expansion_factor = 1.0
        if not (0 < self.second_pass_relax_min_score_factor <= 1.0):
            self.second_pass_relax_min_score_factor = 0.8

    def needs_second_pass(self, kept_count_after_filter: int) -> bool:
        """给 HybridRetriever 判断是否需要二次检索扩大 top_k。"""
        if self.min_candidates_after_filter <= 0:
            return False
        return kept_count_after_filter < self.min_candidates_after_filter

    @staticmethod
    def _candidate_sort_key(cand: RetrievalCandidate) -> float:
        """为候选文档生成最终排序键（越大越靠前）。"""
        if cand.rerank_score is not None and cand.rerank_score > 0:
            return float(cand.rerank_score)
        if cand.fusion_score is not None:
            return float(cand.fusion_score)
        if cand.score is not None:
            return float(cand.score)
        return -999.0

    def filter(self, candidates: list[RetrievalCandidate]) -> ScoreGuardResult:
        """执行相关性阈值过滤 + 兜底策略 + 最终条数裁剪。"""
        total = len(candidates)
        if total == 0:
            return ScoreGuardResult(
                candidates=[],
                filtered_count=0,
                used_fallback=False,
                fallback_reason="上游没有候选文档",
            )

        sorted_cands = sorted(candidates, key=self._candidate_sort_key, reverse=True)

        has_any_rerank_score: bool = any(
            c.rerank_score is not None and c.rerank_score > 0
            for c in sorted_cands
        )

        kept_after_filter: list[RetrievalCandidate] = []
        if has_any_rerank_score:
            for c in sorted_cands:
                has_valid_rerank = (
                    c.rerank_score is not None and c.rerank_score > 0
                )
                score = c.rerank_score if has_valid_rerank else None
                if score is not None and score >= self.min_score:
                    kept_after_filter.append(c)
        else:
            kept_after_filter = list(sorted_cands)

        filtered_count = total - len(kept_after_filter)

        if kept_after_filter:
            final_cands = kept_after_filter[: self.final_top_k]
            result = ScoreGuardResult(
                candidates=final_cands,
                filtered_count=filtered_count,
                used_fallback=False,
            )
            logger.info(f"[ScoreGuard] {result.to_log_line()}")
            return result

        if self.strategy == FALLBACK_EMPTY_SAFE:
            result = ScoreGuardResult(
                candidates=[],
                filtered_count=filtered_count,
                used_fallback=True,
                fallback_reason=(
                    f"empty_safe：所有 {total} 条候选 rerank_score 均 < min_score={self.min_score}，"
                    f"为避免弱相关资料污染上下文，返回空。建议用户换个问法或补充知识库。"
                ),
            )
            logger.warning(f"[ScoreGuard] {result.to_log_line()}")
            return result

        fallback_cands = sorted_cands[: self.fallback_count]
        for c in fallback_cands:
            meta = dict(c.doc.metadata or {})
            meta["_score_guard_fallback"] = True
            meta["_score_guard_reason"] = (
                f"所有候选 rerank_score 均 < {self.min_score}，兜底取前 {self.fallback_count} 条"
            )
            c.doc.metadata = meta

        result = ScoreGuardResult(
            candidates=fallback_cands[: self.final_top_k],
            filtered_count=filtered_count,
            used_fallback=True,
            fallback_reason=(
                f"fallback_rrf：所有 {total} 条候选 rerank_score < min_score={self.min_score}，"
                f"回退前 {len(fallback_cands)} 条（已打低置信标记）。"
            ),
        )
        logger.warning(f"[ScoreGuard] {result.to_log_line()}")
        return result
