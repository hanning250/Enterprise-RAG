"""RAG 查询编排服务模块。

统一链路：HybridRetriever → ContextExpander → ContextCompressor → evidence
同时用于 API 返回与 LLM 总结，并与评测共用同一编排。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.documents import Document

from rag.access_policy import AccessScope
from rag.context_budget import build_context_text
from rag.context_compressor import ContextCompressor
from rag.context_expander import ContextExpander
from rag.hybrid_retriever import HybridRetriever
from rag.query_intent import QueryIntentResult
from utils.config_handler import rag_conf
from utils.logger_handler import logger


def _filter_salary_evidence_by_query_month(
    query: str, docs: list[Document]
) -> list[Document]:
    """查询含明确月份时，丢掉其它月份的工资证据（避免「问3月却带回1月」）。"""
    month = HybridRetriever._salary_month_from_query(query)
    if month is None or not docs:
        return docs

    salary_with_month = [
        d
        for d in docs
        if str((d.metadata or {}).get("doc_type") or "").lower() == "salary"
        and (d.metadata or {}).get("data_month") is not None
    ]
    if not salary_with_month:
        return docs

    matched = [
        d
        for d in salary_with_month
        if int((d.metadata or {}).get("data_month") or -1) == month
    ]
    if not matched:
        return docs

    kept: list[Document] = []
    dropped = 0
    for d in docs:
        meta = d.metadata or {}
        if str(meta.get("doc_type") or "").lower() != "salary":
            kept.append(d)
            continue
        dm = meta.get("data_month")
        if dm is None or int(dm) == month:
            kept.append(d)
        else:
            dropped += 1
    if dropped:
        logger.info(
            f"[EnterpriseQueryService] 按查询月份={month} 过滤工资证据：丢弃 {dropped} 条其它月份"
        )
    return kept if kept else docs


@dataclass
class RetrievalTrace:
    """检索链路的追踪埋点，用于接口响应的 trace 字段与审计日志。"""

    retriever_mode: str = "hybrid"
    rerank_used: bool = False
    vector_hit_count: int = 0
    bm25_hit_count: int = 0
    fusion_candidate_count: int = 0
    acl_filtered_out_count: int = 0
    answerability_failed_count: int = 0
    second_pass_applied: bool = False
    context_tokens_used: int = 0
    docs_included_in_context: int = 0
    retrieval_total_ms: Optional[int] = None
    answer_generation_ms: Optional[int] = None
    fallback_reason: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """统一结果：evidence_docs 与 LLM 用的 context 是同源的同一份。"""

    query: str
    answer: str = ""
    evidence_docs: list[Document] = field(default_factory=list)
    intent: Optional[QueryIntentResult] = None
    trace: RetrievalTrace = field(default_factory=RetrievalTrace)
    acl_blocked_docs: list[str] = field(default_factory=list)


class EnterpriseQueryService:
    """企业级 RAG 查询编排服务。

    纯编排层，不直接初始化模型/Embedding，链路由外部注入。
    目标：一次检索（含 expand/compress）→ evidence 同时用于返回 + 给 LLM 总结。
    """

    def __init__(
        self,
        hybrid_retriever: Optional[HybridRetriever] = None,
        *,
        context_expander: Optional[ContextExpander] = None,
        context_compressor: Optional[ContextCompressor] = None,
    ):
        """初始化编排服务。

        Parameters
        ----------
        hybrid_retriever : HybridRetriever, optional
            外部传入已初始化的 HybridRetriever。若不传则新建一个。
        context_expander / context_compressor :
            可注入以便测试；默认基于 hybrid 的 vector store 创建。
        """
        self.hybrid = hybrid_retriever or HybridRetriever()
        if context_expander is not None:
            self.context_expander = context_expander
        else:
            self.context_expander = ContextExpander(self.hybrid.vector_store.vector_store)
        self.context_compressor = context_compressor or ContextCompressor()
        logger.info(
            f"[EnterpriseQueryService] 初始化完成：hybrid.mode={self.hybrid.mode}"
        )

    def retrieve_only(
        self,
        query: str,
        *,
        top_k_override: Optional[int] = None,
        access_scope: Optional[AccessScope] = None,
    ) -> QueryResult:
        """纯检索阶段：Hybrid → Expand → Compress，返回 evidence_docs + trace。"""
        t0 = time.perf_counter()
        trace = RetrievalTrace(retriever_mode=self.hybrid.mode)

        original_final_top_k = getattr(self.hybrid.score_guard, "final_top_k", None)
        try:
            if top_k_override is not None and int(top_k_override) > 0:
                logger.info(
                    f"[EnterpriseQueryService] 临时覆盖 final_top_k："
                    f"{original_final_top_k} → {top_k_override}"
                )
                self.hybrid.score_guard.final_top_k = int(top_k_override)

            fine_docs, intent_result = self.hybrid.retrieve_with_intent(
                query,
                access_scope=access_scope,
            )
        finally:
            if original_final_top_k is not None:
                self.hybrid.score_guard.final_top_k = original_final_top_k

        expanded_docs = self.context_expander.expand(fine_docs, query=query)
        evidence_docs = self.context_compressor.compress(
            query, expanded_docs, intent_result=intent_result
        )
        evidence_docs = _filter_salary_evidence_by_query_month(query, evidence_docs)

        trace.fusion_candidate_count = len(fine_docs)
        trace.extra["fine_docs_count"] = len(fine_docs)
        trace.extra["expanded_docs_count"] = len(expanded_docs)
        trace.extra["evidence_docs_count"] = len(evidence_docs)

        for doc in fine_docs:
            meta = doc.metadata or {}
            src = meta.get("_retrieval_source")
            if src == "vector":
                trace.vector_hit_count += 1
            elif src == "bm25":
                trace.bm25_hit_count += 1
            if meta.get("_rerank_score") is not None:
                trace.rerank_used = True
            if meta.get("_answerability_passed") is False:
                trace.answerability_failed_count += 1
            if meta.get("_second_pass_applied") is True:
                trace.second_pass_applied = True

        fallback_reason = None
        if intent_result and hasattr(intent_result, "extra"):
            fallback_reason = intent_result.extra.get("fallback_reason")
        if fallback_reason is None and fine_docs:
            first_meta = fine_docs[0].metadata or {}
            fallback_reason = first_meta.get("_fallback_reason")
        trace.fallback_reason = fallback_reason

        t1 = time.perf_counter()
        trace.retrieval_total_ms = int((t1 - t0) * 1000)

        logger.info(
            f"[EnterpriseQueryService] 纯检索完成：query='{query[:60]}...'，"
            f"fine={len(fine_docs)} → expanded={len(expanded_docs)} → evidence={len(evidence_docs)}，"
            f"vector_hit={trace.vector_hit_count}，bm25_hit={trace.bm25_hit_count}，"
            f"answerability_failed={trace.answerability_failed_count}，"
            f"second_pass={trace.second_pass_applied}，"
            f"耗时={trace.retrieval_total_ms}ms"
        )

        return QueryResult(
            query=query,
            evidence_docs=evidence_docs,
            intent=intent_result,
            trace=trace,
        )

    def build_context_from_docs(
        self, docs: list[Document], *, rag_config: Optional[dict] = None
    ) -> tuple[str, int, int]:
        """把 docs 按 token 预算拼接成 context 文本。

        直接复用 rag.context_budget.build_context_text，max_context_tokens
        从 rag_config["max_context_tokens"] 读取，若未传 rag_config 则回退到全局 rag_conf。

        Parameters
        ----------
        docs : list[Document]
            已检索到的证据文档列表。
        rag_config : dict, optional
            接口层传入的 rag 级配置，支持覆盖 max_context_tokens。
            若为 None 则使用全局 rag_conf.get("rag", {})。

        Returns
        -------
        tuple[str, int, int]
            (context_text, tokens_used, included_count)
        """
        effective_cfg = rag_config if rag_config is not None else (rag_conf.get("rag") or {})
        max_context_tokens = effective_cfg.get("max_context_tokens")
        budget = (
            int(max_context_tokens)
            if max_context_tokens is not None and int(max_context_tokens) > 0
            else None
        )
        context_text, tokens_used, included_count = build_context_text(
            docs=docs,
            max_context_tokens=budget,
        )
        return context_text, tokens_used, included_count

    def summarize_with_context(
        self,
        query: str,
        context_docs: list[Document],
        *,
        rag_config: Optional[dict] = None,
        rag_service_chain: Any = None,
    ) -> str:
        """只跑 LLM 总结阶段，不再重复检索。

        Parameters
        ----------
        query : str
            用户查询问题。
        context_docs : list[Document]
            已检索到的证据文档，直接用于构建 context 给 LLM。
        rag_config : dict, optional
            rag 级配置，透传给 build_context_from_docs。
        rag_service_chain : Any, optional
            外部传入的已初始化好的 LCEL 链
            （PromptTemplate | model | StrOutputParser）。
            若未传入则返回空串并打印 warning，避免此模块直接依赖 model.factory。

        Returns
        -------
        str
            LLM 生成的回答。资料不足或缺少 chain 时返回空串。
        """
        if rag_service_chain is None:
            logger.warning(
                "[EnterpriseQueryService] summarize_with_context 未传入 rag_service_chain，"
                "跳过 LLM 总结，返回空串。请在外部初始化 rag_service.chain 后注入。"
            )
            return ""

        context_text, tokens_used, included_count = self.build_context_from_docs(
            context_docs, rag_config=rag_config
        )

        if not context_text.strip() or included_count == 0:
            logger.info(
                f"[EnterpriseQueryService] 资料不足，不调用LLM："
                f"included_count={included_count}，context_empty={not bool(context_text.strip())}"
            )
            return ""

        intent_name = (
            context_docs[0].metadata or {}
        ).get("_query_intent", "unknown") if context_docs else "unknown"
        budget_cfg = rag_config if rag_config is not None else (rag_conf.get("rag") or {})
        budget = budget_cfg.get("max_context_tokens")
        logger.info(
            f"[EnterpriseQueryService] 调用LLM总结：intent={intent_name}, "
            f"evidence={len(context_docs)} 条，实际纳入 {included_count} 条，"
            f"估算占用 {tokens_used}/{budget or '∞'} tokens"
        )

        try:
            answer = rag_service_chain.invoke(
                {"input": query, "context": context_text}
            )
            return answer or ""
        except Exception as exc:
            logger.error(
                f"[EnterpriseQueryService] LLM 总结阶段异常：{exc}",
                exc_info=True,
            )
            return ""

    def query_full(
        self,
        query: str,
        *,
        summarize: bool = True,
        top_k_override: Optional[int] = None,
        rag_service_chain: Any = None,
        rag_config: Optional[dict] = None,
        access_scope: Optional[AccessScope] = None,
    ) -> QueryResult:
        """完整链路：一次检索 → evidence 同时用于返回 + 给 LLM 总结。

        这是给 /api/rag/query 接口用的主入口。保证返回的 evidence_docs 与
        LLM 生成 answer 时使用的 context 来源于同一次检索结果。

        Parameters
        ----------
        query : str
            用户查询问题。
        summarize : bool, default True
            是否执行 LLM 总结。若为 False 则只检索，answer 保持空串。
        top_k_override : int, optional
            临时覆盖 final_top_k，透传给 retrieve_only。
        rag_service_chain : Any, optional
            外部注入的 LCEL 链，透传给 summarize_with_context。
        rag_config : dict, optional
            rag 级配置，透传给 build_context_from_docs。

        Returns
        -------
        QueryResult
            填充完整的 query 结果：evidence_docs + answer + intent + trace。
        """
        result = self.retrieve_only(
            query,
            top_k_override=top_k_override,
            access_scope=access_scope,
        )

        if not summarize:
            logger.info(
                f"[EnterpriseQueryService] query_full 仅检索（summarize=False）："
                f"query='{query[:60]}...'，evidence={len(result.evidence_docs)} 条"
            )
            return result

        t_ans0 = time.perf_counter()
        answer_text = self.summarize_with_context(
            query,
            result.evidence_docs,
            rag_config=rag_config,
            rag_service_chain=rag_service_chain,
        )
        t_ans1 = time.perf_counter()

        result.answer = answer_text or ""
        result.trace.answer_generation_ms = int((t_ans1 - t_ans0) * 1000)

        context_text, tokens_used, included_count = self.build_context_from_docs(
            result.evidence_docs, rag_config=rag_config
        )
        result.trace.context_tokens_used = tokens_used
        result.trace.docs_included_in_context = included_count

        logger.info(
            f"[EnterpriseQueryService] query_full 完成：query='{query[:60]}...'，"
            f"evidence={len(result.evidence_docs)} 条，"
            f"answer_len={len(result.answer)} 字符，"
            f"总结耗时={result.trace.answer_generation_ms}ms"
        )

        return result
