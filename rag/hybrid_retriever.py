"""混合检索编排器模块。"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from langchain_core.documents import Document

from rag.access_policy import AccessScope, filter_candidates_by_acl
from rag.bm25_retriever import BM25Retriever
from rag.context_compressor import AnswerabilityFilter
from rag.fusion import CandidateFusion
from rag.query_intent import QueryIntentResult, classify
from rag.query_rewriter import QueryRewriteResult, QueryRewriter
from rag.reranker import HttpRerankerClient, BaseRerankerClient
from rag.score_guard import ScoreGuard, ScoreGuardResult
from rag.types import RetrievalCandidate
from rag.vector_store import VectorStoreService
from utils.config_handler import rag_conf
from utils.logger_handler import logger


def _resolve_retrieval_config() -> dict:
    """从 rag_conf 中解析 retrieval 级配置。"""
    retrieval_cfg = rag_conf.get("retrieval", {}) or {}

    mode = str(retrieval_cfg.get("mode", "hybrid")).lower()
    allowed_modes = ("hybrid", "vector", "bm25")
    if mode not in allowed_modes:
        logger.warning(
            f"[HybridRetriever] 未知 retrieval.mode={mode}，已回退到 hybrid"
        )
        mode = "hybrid"

    vector_top_k = int(retrieval_cfg.get("vector_top_k", 20))
    bm25_top_k = int(retrieval_cfg.get("bm25_top_k", 20))

    if vector_top_k < 1:
        vector_top_k = 20
    if bm25_top_k < 1:
        bm25_top_k = 20

    # 并发开关：retrieval.concurrent_recall 控制是否并发执行双路召回
    concurrent_recall = bool(retrieval_cfg.get("concurrent_recall", True))
    # 多 query 召回：query_rewrite.enable 控制是否启用 query 改写后多路召回合并
    multi_query_recall = bool(
        (rag_conf.get("query_rewrite", {}) or {}).get("enable_multi_query_recall", True)
    )
    # 二次检索开关：score_guard 过滤后不足 min_candidates 时自动扩大 top_k
    enable_second_pass = bool(retrieval_cfg.get("enable_second_pass", True))
    # 二检允许的最大 top_k 放大倍数（避免二检放大到离谱）
    second_pass_max_expansion = int(retrieval_cfg.get("second_pass_max_expansion", 4))
    if second_pass_max_expansion < 1:
        second_pass_max_expansion = 4

    return {
        "mode": mode,
        "vector_top_k": vector_top_k,
        "bm25_top_k": bm25_top_k,
        "concurrent_recall": concurrent_recall,
        "multi_query_recall": multi_query_recall,
        "enable_second_pass": enable_second_pass,
        "second_pass_max_expansion": second_pass_max_expansion,
    }


class HybridRetriever:
    """混合检索编排器。"""

    def __init__(
        self,
        *,
        vector_store_service: Optional[VectorStoreService] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        fusion: Optional[CandidateFusion] = None,
        reranker: Optional[BaseRerankerClient] = None,
        score_guard: Optional[ScoreGuard] = None,
        answerability_filter: Optional[AnswerabilityFilter] = None,
        query_rewriter: Optional[QueryRewriter] = None,
    ):
        """初始化 HybridRetriever。"""
        self._retrieval_cfg = _resolve_retrieval_config()
        self.mode: str = self._retrieval_cfg["mode"]
        self.vector_top_k: int = self._retrieval_cfg["vector_top_k"]
        self.bm25_top_k: int = self._retrieval_cfg["bm25_top_k"]
        self.concurrent_recall: bool = self._retrieval_cfg["concurrent_recall"]
        self.multi_query_recall: bool = self._retrieval_cfg["multi_query_recall"]
        self.enable_second_pass: bool = self._retrieval_cfg["enable_second_pass"]
        self.second_pass_max_expansion: int = self._retrieval_cfg["second_pass_max_expansion"]

        self.vector_store = vector_store_service or VectorStoreService()
        self.bm25 = bm25_retriever or BM25Retriever(vector_store_service=self.vector_store)
        self.fusion = fusion or CandidateFusion()
        self.reranker: BaseRerankerClient = reranker or HttpRerankerClient()
        self.score_guard = score_guard or ScoreGuard()
        # AnswerabilityFilter：对 Rerank 输出的候选做「是否真能回答该 query」的规则检查
        # 答非所问的高分候选会被打 penalty（默认 x0.88），再交给 ScoreGuard 做 min_score 过滤
        self.answerability = answerability_filter or AnswerabilityFilter()
        self.query_rewriter = query_rewriter or QueryRewriter()

        logger.info(
            f"[HybridRetriever] 初始化完成：mode={self.mode}，"
            f"vector_top_k={self.vector_top_k}，bm25_top_k={self.bm25_top_k}，"
            f"concurrent_recall={self.concurrent_recall}，"
            f"multi_query_recall={self.multi_query_recall}，"
            f"second_pass={self.enable_second_pass}(x≤{self.second_pass_max_expansion})"
        )

    def _vector_retrieve(
        self,
        query: str,
        *,
        vector_top_k_override: Optional[int] = None,
    ) -> list[RetrievalCandidate]:
        """调用 Chroma 向量相似度搜索，并包装成 RetrievalCandidate 列表。"""
        if self.mode not in ("hybrid", "vector"):
            return []
        top_k = int(vector_top_k_override) if vector_top_k_override else self.vector_top_k
        try:
            raw_docs = self.vector_store.vector_store.similarity_search(
                query=query,
                k=top_k,
                filter={"chunk_level": {"$eq": "fine"}},
            )
        except Exception as exc:
            logger.error(
                f"[HybridRetriever] 向量召回异常：query='{query[:80]}...' 错误={exc}",
                exc_info=True,
            )
            return []

        candidates: list[RetrievalCandidate] = []
        for idx, doc in enumerate(raw_docs):
            rank = idx + 1
            candidates.append(
                RetrievalCandidate(
                    doc=doc,
                    source="vector",
                    rank=rank,
                    score=None,
                )
            )

        logger.debug(
            f"[HybridRetriever] 向量召回完成：query='{query[:50]}...' → {len(candidates)}/{top_k} 条"
        )
        return candidates

    def _bm25_retrieve(
        self,
        query: str,
        *,
        bm25_top_k_override: Optional[int] = None,
        access_scope: Optional[AccessScope] = None,
    ) -> list[RetrievalCandidate]:
        """调用 BM25Retriever 做关键词召回，外包一层异常防御。"""
        if self.mode not in ("hybrid", "bm25"):
            return []
        top_k = int(bm25_top_k_override) if bm25_top_k_override else self.bm25_top_k
        try:
            result = self.bm25.retrieve(query, top_k=top_k, access_scope=access_scope)
        except Exception as exc:
            logger.error(
                f"[HybridRetriever] BM25 召回异常：query='{query[:80]}...' 错误={exc}",
                exc_info=True,
            )
            return []
        return result

    @staticmethod
    def _salary_month_from_query(query: str) -> int | None:
        match = re.search(r"2026年\s*(\d{1,2})月|(\d{1,2})月", query or "")
        if not match:
            return None
        raw = match.group(1) or match.group(2)
        try:
            month = int(raw)
        except (TypeError, ValueError):
            return None
        return month if 1 <= month <= 12 else None

    @staticmethod
    def _salary_field_hits(query: str, content: str) -> int:
        fields = (
            "实发工资",
            "应发工资",
            "基本工资",
            "绩效工资",
            "岗位津贴",
            "社保",
            "公积金",
            "个税",
            "工资",
        )
        return sum(1 for field in fields if field in query and field in content)

    def _structured_salary_retrieve(
        self,
        query: str,
        *,
        access_scope: Optional[AccessScope] = None,
        top_k: int = 20,
    ) -> list[RetrievalCandidate]:
        """结构化工资召回：按 metadata 精确匹配月份/姓名，补足向量与 BM25 的短词弱点。"""
        if "工资" not in query and "社保" not in query and "公积金" not in query and "个税" not in query:
            return []

        month = self._salary_month_from_query(query)
        identity_name = (access_scope.user_name if access_scope is not None else "") or ""
        try:
            raw = self.vector_store.vector_store.get(
                include=["documents", "metadatas"],
                where={"doc_type": {"$eq": "salary"}},
                limit=5000,
            )
        except Exception as exc:
            logger.warning(f"[HybridRetriever] 工资结构化召回失败：{exc}")
            return []

        docs_text: list[str] = raw.get("documents") or []
        metas_list: list[dict] = raw.get("metadatas") or []
        candidates_with_score: list[tuple[float, RetrievalCandidate]] = []

        for idx, text in enumerate(docs_text):
            meta = dict(metas_list[idx] or {})
            if str(meta.get("chunk_level") or "") != "fine":
                continue
            score = 0.0

            doc_month = meta.get("data_month")
            if month is not None:
                if doc_month != month:
                    continue
                score += 20.0

            employee_name = str(meta.get("employee_name") or "").strip()
            is_total_row = not employee_name and ("合计" in (text or "") or "总额" in query)
            if employee_name and employee_name in query:
                score += 100.0
            elif identity_name and employee_name and ("我" in query or identity_name in query) and employee_name == identity_name:
                score += 100.0
            elif is_total_row and any(word in query for word in ("合计", "总额", "汇总", "所有", "全体", "本部门")):
                score += 50.0
            elif employee_name:
                # 问了工资但没有姓名匹配时，不让普通员工行淹没精确结果。
                score += 1.0

            score += self._salary_field_hits(query, text or "") * 10.0
            if score <= 0:
                continue

            doc = Document(page_content=text or "", metadata=meta)
            candidate = RetrievalCandidate(
                doc=doc,
                source="hybrid",
                rank=idx + 1,
                score=score,
                fusion_score=score,
                rerank_score=score,
            )
            candidates_with_score.append((score, candidate))

        candidates_with_score.sort(key=lambda item: item[0], reverse=True)
        candidates = [candidate for _score, candidate in candidates_with_score[:top_k]]

        if access_scope is not None:
            candidates, _blocked, _blocked_ids = filter_candidates_by_acl(access_scope, candidates)

        for rank, cand in enumerate(candidates, start=1):
            cand.rank = rank
            meta = dict(cand.doc.metadata or {})
            meta["_structured_recall"] = "salary"
            cand.doc.metadata = meta
        return candidates

    def _dual_recall(
        self,
        query: str,
        *,
        vector_top_k_override: Optional[int] = None,
        bm25_top_k_override: Optional[int] = None,
        access_scope: Optional[AccessScope] = None,
    ) -> tuple[list[RetrievalCandidate], list[RetrievalCandidate]]:
        """执行 vector + BM25 双路召回，返回 (vector_cands, bm25_cands)。

        concurrent_recall 开启时用 ThreadPoolExecutor(2) 并发执行，用户感知延迟约减半。
        """
        if not self.concurrent_recall or self.mode == "vector":
            vector_cands = self._vector_retrieve(
                query, vector_top_k_override=vector_top_k_override
            )
            bm25_cands = (
                []
                if self.mode == "vector"
                else self._bm25_retrieve(
                    query,
                    bm25_top_k_override=bm25_top_k_override,
                    access_scope=access_scope,
                )
            )
            return vector_cands, bm25_cands
        if self.mode == "bm25":
            return [], self._bm25_retrieve(
                query,
                bm25_top_k_override=bm25_top_k_override,
                access_scope=access_scope,
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_v = pool.submit(
                self._vector_retrieve,
                query,
                vector_top_k_override=vector_top_k_override,
            )
            fut_b = pool.submit(
                self._bm25_retrieve,
                query,
                bm25_top_k_override=bm25_top_k_override,
                access_scope=access_scope,
            )
            results: dict[str, list[RetrievalCandidate]] = {}
            for fut in as_completed([fut_v, fut_b]):
                if fut is fut_v:
                    results["v"] = fut.result()
                else:
                    results["b"] = fut.result()
            return results.get("v", []), results.get("b", [])

    def _multi_query_dual_recall(
        self,
        query: str,
        *,
        rewrite_result: QueryRewriteResult | None,
        vector_top_k_override: Optional[int] = None,
        bm25_top_k_override: Optional[int] = None,
        access_scope: Optional[AccessScope] = None,
    ) -> tuple[list[RetrievalCandidate], list[RetrievalCandidate]]:
        """多 query 并行双路召回：每个 query 都跑 vector+BM25，再合并结果。

        改写 query 的结果 fusion_score 会乘以一个系数（默认 0.75），保证原 query 优先。
        """
        if (
            rewrite_result is None
            or not rewrite_result.rewrites
            or not self.multi_query_recall
        ):
            vector_cands, bm25_cands = self._dual_recall(
                query,
                vector_top_k_override=vector_top_k_override,
                bm25_top_k_override=bm25_top_k_override,
                access_scope=access_scope,
            )
            structured = self._structured_salary_retrieve(
                query,
                access_scope=access_scope,
                top_k=bm25_top_k_override or self.bm25_top_k,
            )
            return vector_cands, self._merge_candidates_by_key(structured, bm25_cands)

        all_queries: list[tuple[str, float]] = [
            (rewrite_result.original, 1.0)
        ] + [(q, 0.75) for q in rewrite_result.rewrites]

        vector_all: list[RetrievalCandidate] = []
        bm25_all: list[RetrievalCandidate] = []
        seen_dedup_v: set[str] = set()
        seen_dedup_b: set[str] = set()

        def _one_q_recall(q_entry: tuple[str, float]) -> tuple[str, float, list[RetrievalCandidate], list[RetrievalCandidate]]:
            q, weight = q_entry
            vc, bc = self._dual_recall(
                q,
                vector_top_k_override=vector_top_k_override,
                bm25_top_k_override=bm25_top_k_override,
                access_scope=access_scope,
            )
            return q, weight, vc, bc

        with ThreadPoolExecutor(max_workers=min(4, max(1, len(all_queries)))) as pool:
            rows = list(pool.map(_one_q_recall, all_queries))

        for _q, weight, v_cands, b_cands in rows:
            for cand in v_cands:
                k = cand.get_dedup_key()
                if k in seen_dedup_v:
                    continue
                seen_dedup_v.add(k)
                if cand.fusion_score is None:
                    cand.fusion_score = 1.0 / (60 + cand.rank)
                cand.fusion_score = cand.fusion_score * weight
                vector_all.append(cand)
            for cand in b_cands:
                k = cand.get_dedup_key()
                if k in seen_dedup_b:
                    continue
                seen_dedup_b.add(k)
                if cand.fusion_score is None:
                    cand.fusion_score = (cand.score or 0.0) or (1.0 / (60 + cand.rank))
                cand.fusion_score = cand.fusion_score * weight
                bm25_all.append(cand)

        vector_all.sort(
            key=lambda c: (c.fusion_score if c.fusion_score is not None else -1),
            reverse=True,
        )
        bm25_all.sort(
            key=lambda c: (c.fusion_score if c.fusion_score is not None else -1),
            reverse=True,
        )
        return vector_all, bm25_all

    @staticmethod
    def _merge_candidates_by_key(
        primary: list[RetrievalCandidate],
        secondary: list[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:
        merged: list[RetrievalCandidate] = []
        seen: set[str] = set()
        for cand in [*primary, *secondary]:
            key = cand.get_dedup_key()
            if key in seen:
                continue
            seen.add(key)
            merged.append(cand)
        return merged

    def _fuse_and_rank(
        self,
        query: str,
        vector_cands: list[RetrievalCandidate],
        bm25_cands: list[RetrievalCandidate],
        intent_result: QueryIntentResult,
    ) -> tuple[list[RetrievalCandidate], ScoreGuardResult, list[RetrievalCandidate]]:
        """执行融合 → Rerank → Answerability → ScoreGuard，返回最终列表。"""
        if self.mode == "vector":
            fused = list(vector_cands)
            for idx, c in enumerate(fused):
                if c.fusion_score is None:
                    c.fusion_score = 1.0 / (60 + idx + 1)
        elif self.mode == "bm25":
            fused = list(bm25_cands)
            for idx, c in enumerate(fused):
                if c.fusion_score is None:
                    c.fusion_score = (c.score or 0.0) or (1.0 / (60 + idx + 1))
        else:
            fused = self.fusion.merge(
                bm25_candidates=bm25_cands,
                vector_candidates=vector_cands,
            )

        if not fused:
            empty_guard = ScoreGuardResult(
                candidates=[],
                filtered_count=0,
                used_fallback=False,
                fallback_reason="两路召回 + 融合后没有候选（知识库可能为空）",
            )
            return [], empty_guard, []

        reranker_fail_open: bool = bool(getattr(self.reranker, "fail_open", True))
        try:
            reranked = self.reranker.rerank(query, fused)
        except Exception as exc:
            if reranker_fail_open:
                logger.error(
                    f"[HybridRetriever] Rerank 阶段异常（fail_open=true → 回退 RRF 顺序）：{exc}",
                    exc_info=True,
                )
                reranked = fused
            else:
                logger.error(
                    f"[HybridRetriever] Rerank 阶段异常（fail_open=false → 继续抛出，不兜底）：{exc}",
                    exc_info=True,
                )
                raise

        logger.info(
            f"[HybridRetriever] 步骤3-Rerank："
            f"输入 {len(fused)} 条，输出 {len(reranked)} 条"
        )

        # ── 步骤3.5：AnswerabilityFilter（答非所问的高分候选，打 penalty 降权）──
        scored = self.answerability.filter(query, reranked, intent_result=intent_result)

        for cand in scored:
            meta = dict(cand.doc.metadata or {})
            meta.setdefault("_query_intent", intent_result.intent)
            cand.doc.metadata = meta

        guard_result: ScoreGuardResult = self.score_guard.filter(scored)
        return guard_result.candidates, guard_result, scored

    def retrieve(
        self,
        query: str,
        *,
        access_scope: Optional[AccessScope] = None,
    ) -> tuple[list[RetrievalCandidate], ScoreGuardResult]:
        """执行完整的 Hybrid Recall + Rerank 链路。

        返回：
          - final_candidates : 经过所有过滤后的最终候选
          - guard_result     : ScoreGuard 结果（含过滤计数 / 兜底原因）
        """
        if not query or not query.strip():
            empty_result = ScoreGuardResult(
                candidates=[],
                filtered_count=0,
                used_fallback=False,
                fallback_reason="query 为空",
            )
            return [], empty_result

        scope_blocked_total: int = 0
        scope_blocked_ids: list[str] = []

        # 先 classify 一次，后面 AnswerabilityFilter 和 ContextCompressor 都能复用这个 intent，
        # 避免整个链路重复 classify N 次。
        intent_result: QueryIntentResult = classify(query)

        # QueryRewrite：只有 multi_query_recall 或用户显式要改写时跑
        rewrite_result: QueryRewriteResult | None = None
        if self.multi_query_recall and self.query_rewriter.enabled:
            rewrite_result = self.query_rewriter.rewrite(query)
            if rewrite_result.rewrites:
                logger.info(
                    f"[HybridRetriever] Query改写：原文={query[:60]} → "
                    f"{len(rewrite_result.rewrites)} 条改写：{rewrite_result.rewrites}"
                )
            else:
                rewrite_result = None

        # 第一轮：正常 top_k 双路召回
        vector_cands, bm25_cands = self._multi_query_dual_recall(
            query,
            rewrite_result=rewrite_result,
            access_scope=access_scope,
        )
        logger.info(
            f"[HybridRetriever] 步骤1-双路召回："
            f"向量 {len(vector_cands)} 条，BM25 {len(bm25_cands)} 条"
            f"，intent={intent_result.intent}"
            + (
                f"，multi_query 改写启用，query总数={len(rewrite_result.all_queries)}"
                if rewrite_result
                else ""
            )
        )

        # 步骤2：双路召回后、融合 Rerank 之前做 ACL 过滤
        if access_scope is not None:
            vector_cands, v_blocked, v_blocked_ids = filter_candidates_by_acl(
                access_scope, vector_cands
            )
            bm25_cands, b_blocked, b_blocked_ids = filter_candidates_by_acl(
                access_scope, bm25_cands
            )
            scope_blocked_total += v_blocked + b_blocked
            scope_blocked_ids.extend(v_blocked_ids)
            scope_blocked_ids.extend(b_blocked_ids)
            logger.info(
                f"[HybridRetriever] 步骤2-ACL过滤："
                f"向量路过滤{v_blocked}条，BM25路过滤{b_blocked}条，"
                f"过滤后向量 {len(vector_cands)} 条，BM25 {len(bm25_cands)} 条"
            )
        else:
            logger.debug(
                "[HybridRetriever] access_scope=None，跳过 ACL 过滤（双路召回后）"
            )

        final_cands, guard_result, pre_guard_scored = self._fuse_and_rank(
            query, vector_cands, bm25_cands, intent_result
        )

        # 第二轮：如果 ScoreGuard 过滤后数量不够，再跑一次（扩大 top_k + 放宽 min_score）
        kept_after_filter = (
            guard_result.filtered_count
        )  # pre_guard 里被 min_score 过滤前留下的数量 = total - filtered_count
        pre_guard_total = len(pre_guard_scored)
        kept_after_filter_count = pre_guard_total - guard_result.filtered_count
        if (
            self.enable_second_pass
            and self.score_guard.needs_second_pass(kept_after_filter_count)
            and guard_result.candidates
            is not None  # safety：空列表也可以触发二检
        ):
            expansion = min(
                self.second_pass_max_expansion,
                max(2, int(self.score_guard.second_pass_expansion_factor)),
            )
            new_vector_k = int(self.vector_top_k) * expansion
            new_bm25_k = int(self.bm25_top_k) * expansion
            relaxed_min_score = (
                self.score_guard.min_score
                * self.score_guard.second_pass_relax_min_score_factor
            )
            logger.warning(
                f"[HybridRetriever] 一检只保留了 {kept_after_filter_count} 条，"
                f"低于 min_candidates_after_filter={self.score_guard.min_candidates_after_filter}。"
                f"二检扩大检索：vector_top_k×{expansion}={new_vector_k}，"
                f"bm25_top_k×{expansion}={new_bm25_k}，"
                f"min_score×{self.score_guard.second_pass_relax_min_score_factor:.2f}={relaxed_min_score:.3f}"
            )
            # 二检：临时构建一个放松的 ScoreGuard 实例做过滤，最终结果还是写回到原来的 guard_result
            vector_cands_2, bm25_cands_2 = self._multi_query_dual_recall(
                query,
                rewrite_result=rewrite_result,
                vector_top_k_override=new_vector_k,
                bm25_top_k_override=new_bm25_k,
                access_scope=access_scope,
            )

            # 二检：双路召回后也做 ACL 过滤
            if access_scope is not None:
                vector_cands_2, v2_blocked, v2_blocked_ids = filter_candidates_by_acl(
                    access_scope, vector_cands_2
                )
                bm25_cands_2, b2_blocked, b2_blocked_ids = filter_candidates_by_acl(
                    access_scope, bm25_cands_2
                )
                scope_blocked_total += v2_blocked + b2_blocked
                scope_blocked_ids.extend(v2_blocked_ids)
                scope_blocked_ids.extend(b2_blocked_ids)
                logger.info(
                    f"[HybridRetriever] 二检-ACL过滤："
                    f"向量路过滤{v2_blocked}条，BM25路过滤{b2_blocked}条"
                )
            else:
                logger.debug(
                    "[HybridRetriever] access_scope=None，二检跳过 ACL 过滤"
                )

            # 合并一检+二检候选（按 dedup_key 去重）
            merged_vec_keys: set[str] = set()
            merged_vec: list[RetrievalCandidate] = []
            for c in list(vector_cands) + list(vector_cands_2):
                k = c.get_dedup_key()
                if k in merged_vec_keys:
                    continue
                merged_vec_keys.add(k)
                merged_vec.append(c)
            merged_bm_keys: set[str] = set()
            merged_bm: list[RetrievalCandidate] = []
            for c in list(bm25_cands) + list(bm25_cands_2):
                k = c.get_dedup_key()
                if k in merged_bm_keys:
                    continue
                merged_bm_keys.add(k)
                merged_bm.append(c)

            temp_guard = ScoreGuard(
                min_score=relaxed_min_score,
                final_top_k=self.score_guard.final_top_k,
                strategy=self.score_guard.strategy,
                fallback_count=self.score_guard.fallback_count,
                # 二检不再递归触发三检
                min_candidates_after_filter=0,
            )
            try:
                # 二检走完整融合+Rerank+Answerability（保证排序质量）
                fused2 = self.fusion.merge(
                    bm25_candidates=merged_bm,
                    vector_candidates=merged_vec,
                ) if self.mode == "hybrid" else (
                    list(merged_vec) if self.mode == "vector" else list(merged_bm)
                )
                if self.mode == "hybrid":
                    pass  # 上面已经处理
                else:
                    for idx, c in enumerate(fused2):
                        if c.fusion_score is None:
                            c.fusion_score = 1.0 / (60 + idx + 1)
                if fused2:
                    try:
                        reranked2 = self.reranker.rerank(query, fused2)
                    except Exception:
                        reranked2 = fused2
                    scored2 = self.answerability.filter(
                        query, reranked2, intent_result=intent_result
                    )
                    for cand in scored2:
                        meta = dict(cand.doc.metadata or {})
                        meta.setdefault("_query_intent", intent_result.intent)
                        cand.doc.metadata = meta
                    # 把这次的二检标记写入 metadata，方便后续日志追溯
                    for cand in scored2:
                        m2 = dict(cand.doc.metadata or {})
                        m2["_second_pass_applied"] = True
                        m2["_second_pass_vector_k"] = new_vector_k
                        m2["_second_pass_bm25_k"] = new_bm25_k
                        m2["_second_pass_min_score"] = relaxed_min_score
                        cand.doc.metadata = m2
                    guard2 = temp_guard.filter(scored2)
                else:
                    guard2 = ScoreGuardResult(candidates=[], filtered_count=0)
            except Exception as exc2:
                logger.error(
                    f"[HybridRetriever] 二检阶段异常，继续使用一检结果：{exc2}",
                    exc_info=True,
                )
                guard2 = guard_result

            # 二检的候选覆盖一检（只有二检输出更多有价值内容时才替换）
            if guard2.candidates and len(guard2.candidates) >= max(
                1, len(guard_result.candidates)
            ):
                guard2.fallback_reason = (
                    f"[SECOND-PASS EXPANSION ×{expansion}] "
                    + (guard2.fallback_reason or guard_result.fallback_reason or "")
                )
                guard2.filtered_count = (
                    guard_result.filtered_count + guard2.filtered_count
                )
                guard_result = guard2
                final_cands = guard2.candidates

        # 把 ACL 过滤统计写回每个最终 Document 的 metadata
        if scope_blocked_total > 0 or scope_blocked_ids:
            blocked_ids_str = ",".join(scope_blocked_ids)
            for cand in guard_result.candidates:
                meta = dict(cand.doc.metadata or {})
                meta["_acl_blocked_count"] = scope_blocked_total
                meta["_acl_blocked_ids"] = blocked_ids_str
                cand.doc.metadata = meta
            # 同时追加到 guard_result.fallback_reason 前缀（若有）
            if guard_result.fallback_reason:
                guard_result.fallback_reason = (
                    f"[ACL filtered={scope_blocked_total}] " + guard_result.fallback_reason
                )
            else:
                guard_result.fallback_reason = f"[ACL filtered={scope_blocked_total}]"

        logger.info(
            f"[HybridRetriever] 全链路完成："
            f"最终输出 {len(guard_result.candidates)} 条候选；{guard_result.to_log_line()}"
            + (
                f"；ACL过滤{scope_blocked_total}条"
                if scope_blocked_total
                else ""
            )
        )
        return final_cands, guard_result

    def retrieve_with_intent(
        self,
        query: str,
        *,
        access_scope: Optional[AccessScope] = None,
    ) -> tuple[list[Document], QueryIntentResult]:
        """对外暴露：返回最终 LangChain Document + 本次 query 的意图分类结果。

        供上层 ContextCompressor 复用同一个 intent，避免重复 classify。
        """
        final_candidates, _ = self.retrieve(query, access_scope=access_scope)
        if final_candidates:
            first_meta = final_candidates[0].doc.metadata or {}
            cached_intent = first_meta.get("_query_intent")
        else:
            cached_intent = None
        intent_result = (
            classify(query)
            if not cached_intent
            else QueryIntentResult(intent=str(cached_intent), scores={})
        )
        return [c.as_document() for c in final_candidates], intent_result

    def retrieve_documents(
        self,
        query: str,
        *,
        access_scope: Optional[AccessScope] = None,
    ) -> list[Document]:
        """对外暴露的最终接口，直接返回 LangChain Document。"""
        docs, _ = self.retrieve_with_intent(query, access_scope=access_scope)
        return docs
