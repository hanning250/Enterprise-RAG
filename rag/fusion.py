"""候选融合模块。"""

from __future__ import annotations

from typing import Optional

from rag.types import RetrievalCandidate
from utils.config_handler import rag_conf
from utils.logger_handler import logger


def _resolve_fusion_config() -> dict:
    """从 rag_conf 中读取 fusion 配置。"""
    retrieval_cfg = rag_conf.get("retrieval", {}) or {}
    fusion_cfg = rag_conf.get("fusion", {}) or {}

    return {
        "rrf_k": int(fusion_cfg.get("rrf_k", 60)),
        "fusion_top_k": int(retrieval_cfg.get("fusion_top_k", 30)),
        "method": str(fusion_cfg.get("method", "rrf")).lower(),
    }


class CandidateFusion:
    """候选融合器（去重 + RRF 打分 + 截断）。"""

    def __init__(self, *, rrf_k: Optional[int] = None, fusion_top_k: Optional[int] = None):
        """初始化融合器。"""
        cfg = _resolve_fusion_config()
        self.rrf_k: int = int(rrf_k) if rrf_k is not None else cfg["rrf_k"]
        self.fusion_top_k: int = (
            int(fusion_top_k) if fusion_top_k is not None else cfg["fusion_top_k"]
        )
        self.method: str = cfg["method"]

        if self.rrf_k < 1:
            logger.warning(f"[Fusion] rrf_k={self.rrf_k} 非法，已修正为 60")
            self.rrf_k = 60
        if self.fusion_top_k < 1:
            logger.warning(f"[Fusion] fusion_top_k={self.fusion_top_k} 非法，已修正为 30")
            self.fusion_top_k = 30

    def merge(
        self,
        *,
        bm25_candidates: Optional[list[RetrievalCandidate]] = None,
        vector_candidates: Optional[list[RetrievalCandidate]] = None,
    ) -> list[RetrievalCandidate]:
        """把 BM25 和向量两路召回结果做去重、RRF 打分、截断。"""
        bm25_list: list[RetrievalCandidate] = list(bm25_candidates or [])
        vector_list: list[RetrievalCandidate] = list(vector_candidates or [])

        if not bm25_list and not vector_list:
            return []

        merged_pool: dict[str, dict[str, Optional[RetrievalCandidate]]] = {}

        for cand in vector_list:
            key = cand.get_dedup_key()
            bucket = merged_pool.setdefault(key, {"vector": None, "bm25": None})
            bucket["vector"] = cand

        for cand in bm25_list:
            key = cand.get_dedup_key()
            bucket = merged_pool.setdefault(key, {"vector": None, "bm25": None})
            bucket["bm25"] = cand

        logger.debug(
            f"[Fusion] 去重前：向量召回 {len(vector_list)} 条 + BM25 召回 {len(bm25_list)} 条，"
            f"去重后：{len(merged_pool)} 个唯一片段"
        )

        fused: list[RetrievalCandidate] = []
        for key, bucket in merged_pool.items():
            vc: Optional[RetrievalCandidate] = bucket["vector"]
            bc: Optional[RetrievalCandidate] = bucket["bm25"]

            base: RetrievalCandidate = vc or bc  # type: ignore[assignment]
            assert base is not None, "[Fusion] 发现两路都为空的 dedup bucket，逻辑异常"

            vector_rank = vc.rank if vc is not None else None
            bm25_rank = bc.rank if bc is not None else None

            score = 0.0
            if vector_rank is not None and vector_rank > 0:
                score += 1.0 / (self.rrf_k + vector_rank)
            if bm25_rank is not None and bm25_rank > 0:
                score += 1.0 / (self.rrf_k + bm25_rank)

            base.fusion_score = score
            base.source = "hybrid"
            fused.append(base)

        fused.sort(key=lambda c: c.fusion_score or 0.0, reverse=True)
        result = fused[: self.fusion_top_k]

        logger.info(
            f"[Fusion] RRF(k={self.rrf_k}) 完成，"
            f"送出 {len(result)}/{self.fusion_top_k} 条候选给 Rerank"
        )
        return result
