"""韩宁科技有限公司 - 内部知识库 RAG 服务入口（FastAPI）。

Phase1：接口边界 —— 身份校验、CORS、审计日志、V2 显式身份模型。
Phase2：单次查询链路 —— 统一走 EnterpriseQueryService，一次检索拿 evidence，
        同一份 evidence 同时用于 API 返回 + 传给 summarize，彻底修复
        「先 retriever_docs() 再 rag_summarize() 又重新检索一次」的问题。

保留的 V1 公开 API 路径（旧调用方兼容）：
  - POST /api/rag/query        完整 RAG 问答（检索 + 上下文增强 + LLM 总结）
  - POST /api/rag/retrieve     纯检索（只返回相关证据文档，不调 LLM 总结）
  - POST /api/rag/build        扫描 data/ 目录并（增量）构建 / 重建向量知识库
  - GET  /api/health           健康检查 + 配置校验报告
  - GET  /api/config/validate  详细配置校验结果（同 utils/config_validator）

新增的 V2 API 路径（显式携带 identity）：
  - POST /api/v2/rag/query     QueryRequestV2 → QueryResponseV2
  - POST /api/v2/rag/retrieve  RetrieveRequestV2 → RetrieveResponseV2
  - POST /api/v2/rag/build     BuildRequestV2 → BuildResponseV2
"""

from __future__ import annotations

import sys
from pathlib import Path

# 把项目根目录加到 sys.path：这样 `from rag.xxx import ...` 的导入语句才能找到包
# （如果你把项目安装成了 pip 包，这段就不需要了，可以删掉）
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from contextlib import asynccontextmanager
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from utils.config_validator import validate_all_configs
from utils.logger_handler import logger
from utils.config_handler import rag_conf

# ──────────────────────────────────────────────────────────────────────
# Phase1 新增：身份 / ACL / V2 Schema / QueryService imports
# ──────────────────────────────────────────────────────────────────────
from app.auth import (
    load_identity_config,
    parse_identity_from_headers,
    require_request_context,
    is_authorized_for_build,
    is_trusted_identity_request,
    RequestContext,
)
from app.schemas import (
    IdentityInfo,
    QueryRequestV2,
    RetrieveRequestV2,
    BuildRequestV2,
    RetrievedDoc as RetrievedDocV2,
    QueryTrace,
    QueryResponseV2,
    RetrieveResponseV2,
    BuildResponseV2,
)
from rag.query_service import EnterpriseQueryService
from rag.access_policy import AccessScope, filter_documents_by_acl
from rag.answer_policy import post_process_answer, refusal_or_none
from langchain_core.documents import Document


# ──────────────────────────────────────────────────────────────────────
# 全局单例：延迟加载，避免 import 时就卡网络 / 锁 GPU
# ──────────────────────────────────────────────────────────────────────
_rag_service: Any = None
_identity_cfg: Any = None
_query_svc: EnterpriseQueryService | None = None
_PUBLIC_PATHS = {"/api/health"}


def get_rag_service():
    """懒加载 RagSummarizeService 单例（与 QueryService 共享 HybridRetriever）。"""
    global _rag_service
    if _rag_service is None:
        from rag.rag_service import RagSummarizeService

        _rag_service = RagSummarizeService(query_service=get_query_svc())
    return _rag_service


def get_identity_cfg():
    """懒加载 IdentityConfig 单例。（Phase1）"""
    global _identity_cfg
    if _identity_cfg is None:
        _identity_cfg = load_identity_config()
    return _identity_cfg


def get_query_svc() -> EnterpriseQueryService:
    """懒加载 EnterpriseQueryService 单例（统一检索编排）。"""
    global _query_svc
    if _query_svc is None:
        _query_svc = EnterpriseQueryService()
    return _query_svc


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时做一次配置校验 + 身份配置日志；关闭时释放。"""
    validation = validate_all_configs()
    logger.info(validation.report())
    identity_cfg = get_identity_cfg()
    logger.info(f"[配置] 身份校验 require_auth={identity_cfg.require_auth}")
    yield
    logger.info("韩宁科技 RAG 服务已停止。")


app = FastAPI(
    title="韩宁科技有限公司 - 内部知识库 RAG 服务",
    description="企业内部 RAG 知识库服务：Hybrid 检索 + 身份 ACL + 审计日志 + 可观测性 Trace。"
                " 支持 V1（从请求头取身份）和 V2（body 显式携带 identity）两套接口。",
    version="2.0.0",
    lifespan=lifespan,
)


# ──────────────────────────────────────────────────────────────────────
# Phase1：CORS 改造 —— 用 identity_cfg.cors_allow_origins，空才回退 *
# ──────────────────────────────────────────────────────────────────────
def _configure_cors(app: FastAPI) -> None:
    identity_cfg = get_identity_cfg()
    origins = list(identity_cfg.cors_allow_origins or [])
    if not origins:
        logger.warning(
            "[CORS] identity_cfg.cors_allow_origins 为空，"
            "回退 allow_origins=[\"*\"]（生产环境建议显式配置白名单）"
        )
        origins_use = ["*"]
    else:
        origins_use = origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins_use,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


_configure_cors(app)


# ──────────────────────────────────────────────────────────────────────
# Phase1：身份上下文提取中间件
#   1) 每个请求先从 request.headers parse 出 RequestContext
#   2) require_auth=True 且 ctx is None → 401
#   3) 否则写入 request.state.ctx 供下游 Depends 使用
# ──────────────────────────────────────────────────────────────────────
@app.middleware("http")
async def identity_context_middleware(request: Request, call_next):
    """身份上下文提取中间件。（Phase1）

    V2 接口身份在 body，由 handler 校验 X-Internal-Auth + identity；
    此处不强制要求 X-User-Id 请求头。
    """
    path = request.url.path
    if path in _PUBLIC_PATHS or path.startswith("/api/v2/"):
        request.state.ctx = None
        return await call_next(request)

    headers_dict = dict(request.headers)
    ctx = parse_identity_from_headers(headers_dict)
    identity_cfg = get_identity_cfg()

    if ctx is None and identity_cfg.require_auth:
        return JSONResponse(
            status_code=401,
            content={
                "detail": "缺少身份上下文，请配置 X-User-Id 等请求头"
            },
        )

    request.state.ctx = ctx
    response = await call_next(request)
    return response


# ──────────────────────────────────────────────────────────────────────
# 辅助：从 Request.state.ctx 取 ctx，不存在就抛 401（给 V1 接口用）
# ──────────────────────────────────────────────────────────────────────
def _get_ctx_from_request(request: Request) -> RequestContext:
    ctx = getattr(request.state, "ctx", None)
    try:
        return require_request_context(ctx)
    except PermissionError as exc:
        raise HTTPException(
            status_code=401,
            detail=f"缺少身份上下文：{exc}",
        ) from exc


def _require_trusted_identity_request(request: Request) -> None:
    """V2 body identity 也必须由可信内部调用方提交。"""
    identity_cfg = get_identity_cfg()
    if identity_cfg.require_auth and not is_trusted_identity_request(
        dict(request.headers), identity_cfg
    ):
        raise HTTPException(
            status_code=401,
            detail="身份上下文必须由可信内部网关注入，请配置 X-Internal-Auth",
        )


def _summarize_with_answer_policy(
    *,
    query: str,
    scope: AccessScope,
    filtered_docs: list[Document],
    acl_blocked: int,
    qsvc: EnterpriseQueryService,
    rag_svc: Any,
) -> str:
    """拒答策略优先；通过后才 LLM 总结，并做免责声明等后处理。"""
    refusal = refusal_or_none(
        scope,
        filtered_docs,
        acl_blocked_count=acl_blocked,
    )
    if refusal:
        return refusal

    answer = qsvc.summarize_with_context(
        query,
        filtered_docs,
        rag_config=rag_conf.get("rag"),
        rag_service_chain=rag_svc.get_chain(),
    )
    return post_process_answer(answer, docs=filtered_docs)


# ──────────────────────────────────────────────────────────────────────
# V1 请求 / 响应 Schema（保留，旧调用方兼容）
# ──────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户问题")
    summarize: bool = Field(True, description="是否用 LLM 做总结；False 时等价于纯检索")


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户问题")
    top_k: int | None = Field(None, description="可选，覆盖 config 中的 final_top_k")


class BuildRequest(BaseModel):
    force: bool = Field(False, description="是否强制重建（忽略已有的增量缓存）")


class RetrievedDoc(BaseModel):
    page_content: str
    metadata: dict[str, Any]


class QueryResponse(BaseModel):
    answer: str
    evidence_docs: list[RetrievedDoc]


class RetrieveResponse(BaseModel):
    docs: list[RetrievedDoc]
    count: int


class BuildResponse(BaseModel):
    ok: bool
    message: str


# ──────────────────────────────────────────────────────────────────────
# 通用工具：把 RetrievalTrace → QueryTrace（字段一一对应）
# ──────────────────────────────────────────────────────────────────────
def _trace_to_v2(retrieval_trace: Any, *, cache_hit: bool = False) -> QueryTrace:
    """把 rag.query_service.RetrievalTrace 转成 app.schemas.QueryTrace。"""
    retriever_mode = str(getattr(retrieval_trace, "retriever_mode", "hybrid")) or "hybrid"
    if retriever_mode not in ("hybrid", "vector", "bm25"):
        retriever_mode = "hybrid"
    return QueryTrace(
        retriever_used=retriever_mode,
        rerank_used=bool(getattr(retrieval_trace, "rerank_used", False)),
        acl_filtered_count=int(getattr(retrieval_trace, "acl_filtered_out_count", 0) or 0),
        context_used_tokens=int(getattr(retrieval_trace, "context_tokens_used", 0) or 0),
        docs_included_count=int(getattr(retrieval_trace, "docs_included_in_context", 0) or 0),
        retrieval_ms=getattr(retrieval_trace, "retrieval_total_ms", None),
        answer_ms=getattr(retrieval_trace, "answer_generation_ms", None),
        answerability_failed_count=int(getattr(retrieval_trace, "answerability_failed_count", 0) or 0),
        second_pass_applied=bool(getattr(retrieval_trace, "second_pass_applied", False)),
        fallback_reason=getattr(retrieval_trace, "fallback_reason", None),
        cache_hit=cache_hit,
    )


def _docs_to_v1(docs: list) -> list[RetrievedDoc]:
    """把 LangChain Document 列表转成 V1 RetrievedDoc 列表。"""
    return [
        RetrievedDoc(page_content=d.page_content, metadata=dict(d.metadata or {}))
        for d in docs
    ]


def _docs_to_v2(docs: list) -> list[RetrievedDocV2]:
    """把 LangChain Document 列表转成 V2 RetrievedDocV2 列表。"""
    return [
        RetrievedDocV2(page_content=d.page_content, metadata=dict(d.metadata or {}))
        for d in docs
    ]


def _ctx_to_access_scope(ctx: RequestContext) -> AccessScope:
    """把 RequestContext 转成 AccessScope。"""
    return AccessScope(
        user_id=ctx.user_id,
        user_name=ctx.user_name or "",
        department=ctx.department or "",
        roles=list(ctx.roles or []),
        data_scope=str(ctx.data_scope or "self"),
    )


def _identity_to_access_scope(identity: IdentityInfo) -> AccessScope:
    """把 V2 body 里的 IdentityInfo 转成 AccessScope。"""
    return AccessScope(
        user_id=identity.user_id,
        user_name=identity.user_name or "",
        department=identity.department or "",
        roles=list(identity.roles or []),
        data_scope=str(identity.data_scope or "self"),
    )


def _identity_to_ctx_like(identity: IdentityInfo) -> RequestContext:
    """把 V2 IdentityInfo 构造成 RequestContext 风格对象（用于 build 权限判断 / 审计）。"""
    return RequestContext(
        user_id=identity.user_id,
        user_name=identity.user_name or "",
        department=identity.department or "",
        roles=list(identity.roles or []),
        data_scope=str(identity.data_scope or "self"),  # type: ignore[arg-type]
        request_id=identity.request_id or "",
        client_ip=identity.client_ip or "",
    )


# ──────────────────────────────────────────────────────────────────────
# 构建知识库：force=True 真正重建（先删旧 chunk + 标记 pending，再增量）
# ──────────────────────────────────────────────────────────────────────
def _do_build(force: bool) -> tuple[int, int]:
    """执行知识库构建，返回 (rebuild_count, deleted_chunk_count)。

    force=True 时：
      1) 遍历 registry 里所有记录，把每个 document_id 对应的 chroma chunk 全部删掉；
      2) 把 registry 中对应记录 status 置为 rebuild_pending，让后续 load_document()
         重新处理（_can_skip 会因 status != active 而不跳过）。

    无论 force 与否：load_document 全量扫描后会热删除磁盘上已不存在的源文件对应索引，
    并刷新进程内 BM25 缓存。
    """
    from rag.vector_store import VectorStoreService

    vs = VectorStoreService()
    rebuild_count = 0
    deleted_chunk_count = 0

    if force:
        all_records = vs.registry.list_all()
        pending_doc_ids: list[str] = []
        for rec in all_records:
            doc_id = str(rec.get("document_id") or "")
            if not doc_id:
                continue
            try:
                deleted_chunk_count += vs._delete_chunks_by_document_id(doc_id)
            except Exception as exc:
                logger.warning(
                    f"[Build/Force] 删除 document_id={doc_id} 的旧 chunk 失败：{exc}"
                )
            pending_doc_ids.append(doc_id)

        for doc_id in pending_doc_ids:
            vs.registry.set_status(doc_id, "rebuild_pending")

        logger.info(
            f"[Build/Force] 共标记 {len(pending_doc_ids)} 个文档为 rebuild_pending，"
            f"删除旧 chunk {deleted_chunk_count} 条，开始重新入库..."
        )

    vs.load_document()

    if force:
        after_active = vs.registry.list_all(status="active")
        rebuild_count = len(after_active)
    else:
        after_active = vs.registry.list_all(status="active")
        rebuild_count = len(after_active)

    # 刷新线上单例的 BM25，避免热删除后仍命中旧关键词索引
    global _query_svc
    if _query_svc is not None:
        try:
            rebuild_bm25 = bool(
                ((rag_conf.get("cache") or {}).get("bm25") or {}).get("rebuild_on_ingest", True)
            )
            if rebuild_bm25:
                _query_svc.hybrid.bm25.rebuild_index()
                logger.info("[Build] 已刷新进程内 BM25 索引缓存")
        except Exception as exc:
            logger.warning(f"[Build] 刷新 BM25 失败（下次 retrieve 可能仍用旧索引）：{exc}")

    # 知识变更后清空 QA 缓存，避免答旧内容
    try:
        from rag.qa_cache import get_qa_cache

        cleared = get_qa_cache().clear()
        if cleared:
            logger.info(f"[Build] 已清空 QA 缓存 {cleared} 条")
    except Exception as exc:
        logger.warning(f"[Build] 清空 QA 缓存失败：{exc}")

    return rebuild_count, deleted_chunk_count


# ──────────────────────────────────────────────────────────────────────
# Phase2：V2 查询接口 —— 显式 identity，一次检索 + ACL 最后防线 + 审计
# ──────────────────────────────────────────────────────────────────────

@app.post("/api/v2/rag/query", response_model=QueryResponseV2)
def api_v2_rag_query(req: QueryRequestV2, request: Request):
    """V2 完整 RAG：一次检索 → evidence 同源 → LLM 总结 + ACL + 审计。（Phase1+Phase2）"""
    status = "ok"
    evidence_count = 0
    acl_filtered_count = 0
    ctx_like = _identity_to_ctx_like(req.identity)
    try:
        _require_trusted_identity_request(request)
        scope = _identity_to_access_scope(req.identity)

        from rag.qa_cache import QaCache, QaCacheEntry, get_qa_cache

        qa_cache = get_qa_cache()
        cache_key = QaCache.make_key(
            req.query,
            user_id=req.identity.user_id,
            user_name=req.identity.user_name,
            department=req.identity.department,
            roles=list(req.identity.roles or []),
            data_scope=req.identity.data_scope,
            summarize=bool(req.summarize),
        )
        if req.summarize:
            cached = qa_cache.get(cache_key)
            if cached is not None:
                logger.info(
                    f"[V2/query] QA 缓存命中 key={cache_key[:12]}… "
                    f"user={req.identity.user_id}"
                )
                evidence_count = len(cached.evidence_docs)
                acl_filtered_count = int(cached.trace.get("acl_filtered_count") or 0)
                trace = QueryTrace(
                    retriever_used="hybrid",
                    rerank_used=False,
                    acl_filtered_count=acl_filtered_count,
                    context_used_tokens=int(cached.trace.get("context_used_tokens") or 0),
                    docs_included_count=int(cached.trace.get("docs_included_count") or 0),
                    retrieval_ms=0,
                    answer_ms=0,
                    answerability_failed_count=0,
                    second_pass_applied=False,
                    fallback_reason=None,
                    cache_hit=True,
                )
                return QueryResponseV2(
                    answer=cached.answer,
                    evidence_docs=[
                        RetrievedDocV2(
                            page_content=str(d.get("page_content") or ""),
                            metadata=dict(d.get("metadata") or {}),
                        )
                        for d in cached.evidence_docs
                    ],
                    trace=trace,
                    identity_snapshot=req.identity,
                )

        qsvc = get_query_svc()
        rag_svc = get_rag_service()

        result = qsvc.retrieve_only(
            req.query,
            top_k_override=None,
            access_scope=scope,
        )

        before_acl = len(result.evidence_docs)
        filtered_docs, acl_blocked, _blocked_ids = filter_documents_by_acl(
            scope, result.evidence_docs
        )
        acl_filtered_count = acl_blocked
        result.evidence_docs = filtered_docs

        if req.summarize:
            if acl_blocked > 0 and len(filtered_docs) != before_acl:
                logger.info(
                    f"[V2/query] ACL 过滤后 evidence {before_acl} → {len(filtered_docs)}，"
                    f"仅使用授权 evidence summarize"
                )
            result.answer = _summarize_with_answer_policy(
                query=req.query,
                scope=scope,
                filtered_docs=filtered_docs,
                acl_blocked=acl_blocked,
                qsvc=qsvc,
                rag_svc=rag_svc,
            )
            if hasattr(qsvc, "build_context_from_docs"):
                _context_text, tokens_used, included_count = qsvc.build_context_from_docs(
                    filtered_docs,
                    rag_config=rag_conf.get("rag"),
                )
                result.trace.context_tokens_used = tokens_used
                result.trace.docs_included_in_context = included_count
                # 仅返回实际进入上下文的文档，保证【N】与 evidence_docs[N-1] 对齐
                if included_count > 0:
                    result.evidence_docs = filtered_docs[:included_count]
        else:
            logger.info(
                f"[V2/query] summarize=False，仅返回授权 evidence：{len(filtered_docs)} 条"
            )

        evidence_count = len(result.evidence_docs)
        v2_docs = _docs_to_v2(result.evidence_docs)
        v2_trace = _trace_to_v2(result.trace)
        v2_trace.acl_filtered_count = acl_blocked
        v2_trace.cache_hit = False

        if req.summarize and (result.answer or "").strip():
            qa_cache.put(
                cache_key,
                QaCacheEntry(
                    answer=result.answer or "",
                    evidence_docs=[
                        {
                            "page_content": d.page_content,
                            "metadata": dict(d.metadata or {}),
                        }
                        for d in result.evidence_docs
                    ],
                    trace={
                        "acl_filtered_count": acl_blocked,
                        "context_used_tokens": v2_trace.context_used_tokens,
                        "docs_included_count": v2_trace.docs_included_count,
                    },
                    created_at=time.monotonic(),
                ),
            )

        return QueryResponseV2(
            answer=result.answer or "",
            evidence_docs=v2_docs,
            trace=v2_trace,
            identity_snapshot=req.identity,
        )
    except HTTPException:
        status = "err"
        raise
    except Exception as exc:  # pragma: no cover - 防御性
        status = "err"
        logger.exception("[V2/RAG/query] 调用失败")
        raise HTTPException(status_code=500, detail=f"RAG 服务内部错误：{exc}") from exc
    finally:
        req_id = ctx_like.request_id or "-"
        logger.info(
            f"[审计] {req_id} user={ctx_like.user_id}({ctx_like.user_name}) "
            f"dept={ctx_like.department} roles={ctx_like.roles} "
            f"scope={ctx_like.data_scope} "
            f"query={req.query[:80]} evidence={evidence_count} "
            f"acl_filtered={acl_filtered_count} status={status}"
        )


@app.post("/api/v2/rag/retrieve", response_model=RetrieveResponseV2)
def api_v2_rag_retrieve(req: RetrieveRequestV2, request: Request):
    """V2 纯检索：top_k 真正生效 + ACL 最后防线 + 审计。（Phase1+Phase2）"""
    status = "ok"
    evidence_count = 0
    acl_filtered_count = 0
    ctx_like = _identity_to_ctx_like(req.identity)
    try:
        _require_trusted_identity_request(request)
        scope = _identity_to_access_scope(req.identity)
        qsvc = get_query_svc()

        result = qsvc.retrieve_only(
            req.query,
            top_k_override=req.top_k,
            access_scope=scope,
        )

        before_acl = len(result.evidence_docs)
        filtered_docs, acl_blocked, _blocked_ids = filter_documents_by_acl(
            scope, result.evidence_docs
        )
        acl_filtered_count = acl_blocked
        evidence_count = len(filtered_docs)

        v2_docs = _docs_to_v2(filtered_docs)
        return RetrieveResponseV2(
            docs=v2_docs,
            count=len(v2_docs),
            identity_snapshot=req.identity,
        )
    except HTTPException:
        status = "err"
        raise
    except Exception as exc:  # pragma: no cover
        status = "err"
        logger.exception("[V2/RAG/retrieve] 调用失败")
        raise HTTPException(status_code=500, detail=f"检索服务内部错误：{exc}") from exc
    finally:
        req_id = ctx_like.request_id or "-"
        logger.info(
            f"[审计] {req_id} user={ctx_like.user_id}({ctx_like.user_name}) "
            f"dept={ctx_like.department} roles={ctx_like.roles} "
            f"scope={ctx_like.data_scope} "
            f"query={req.query[:80]} evidence={evidence_count} "
            f"acl_filtered={acl_filtered_count} status={status}"
        )


@app.post("/api/v2/rag/build", response_model=BuildResponseV2)
def api_v2_rag_build(req: BuildRequestV2, request: Request):
    """V2 构建：先鉴权 build_endpoint_roles，force=True 真正重建。（Phase1+Phase2）"""
    status = "ok"
    rebuild_count = 0
    deleted_chunk_count = 0
    ctx_like = _identity_to_ctx_like(req.identity)
    try:
        _require_trusted_identity_request(request)
        identity_cfg = get_identity_cfg()
        if not is_authorized_for_build(ctx_like, identity_cfg):
            raise HTTPException(
                status_code=403,
                detail=(
                    "无权限触发知识库重建，需要 build_endpoint_roles 中任一角色："
                    + ", ".join(identity_cfg.build_endpoint_roles)
                ),
            )

        rebuild_count, deleted_chunk_count = _do_build(force=req.force)

        if req.force:
            message = (
                "向量知识库构建完成（force=True 强制重建完成）。"
                f" 重建文档 {rebuild_count} 个，删除旧 chunk {deleted_chunk_count} 条。"
            )
        else:
            message = (
                "向量知识库构建完成（增量，含磁盘已删文件的热删除）。"
                f" 当前 active 文档约 {rebuild_count} 个。"
            )
        return BuildResponseV2(
            ok=True,
            message=message,
            rebuild_count=rebuild_count,
            deleted_chunk_count=deleted_chunk_count,
            identity_snapshot=req.identity,
        )
    except HTTPException:
        status = "err"
        raise
    except Exception as exc:  # pragma: no cover
        status = "err"
        logger.exception("[V2/RAG/build] 构建失败")
        raise HTTPException(status_code=500, detail=f"构建知识库失败：{exc}") from exc
    finally:
        req_id = ctx_like.request_id or "-"
        logger.info(
            f"[审计] {req_id} user={ctx_like.user_id}({ctx_like.user_name}) "
            f"dept={ctx_like.department} roles={ctx_like.roles} "
            f"scope={ctx_like.data_scope} "
            f"query=<BUILD force={req.force}> evidence={rebuild_count} "
            f"acl_filtered={deleted_chunk_count} status={status}"
        )


# ──────────────────────────────────────────────────────────────────────
# Phase1+Phase2：V1 旧接口 —— 兼容旧路径；内部走 QueryService + ACL + 审计
# ──────────────────────────────────────────────────────────────────────

@app.post("/api/rag/query", response_model=QueryResponse)
def api_rag_query(req: QueryRequest, request: Request):
    """V1 完整 RAG（兼容旧调用方）：从 request.state.ctx 取身份，内部走 QueryService。（Phase1+Phase2）"""
    status = "ok"
    evidence_count = 0
    acl_filtered_count = 0
    ctx = _get_ctx_from_request(request)
    try:
        scope = _ctx_to_access_scope(ctx)

        from rag.qa_cache import QaCache, QaCacheEntry, get_qa_cache

        qa_cache = get_qa_cache()
        cache_key = QaCache.make_key(
            req.query,
            user_id=ctx.user_id,
            user_name=ctx.user_name,
            department=ctx.department,
            roles=list(ctx.roles or []),
            data_scope=ctx.data_scope,
            summarize=bool(req.summarize),
        )
        if req.summarize:
            cached = qa_cache.get(cache_key)
            if cached is not None:
                logger.info(
                    f"[RAG/query] QA 缓存命中 key={cache_key[:12]}… user={ctx.user_id}"
                )
                evidence_count = len(cached.evidence_docs)
                acl_filtered_count = int(cached.trace.get("acl_filtered_count") or 0)
                return QueryResponse(
                    answer=cached.answer,
                    evidence_docs=[
                        RetrievedDoc(
                            page_content=str(d.get("page_content") or ""),
                            metadata=dict(d.get("metadata") or {}),
                        )
                        for d in cached.evidence_docs
                    ],
                )

        qsvc = get_query_svc()
        rag_svc = get_rag_service()

        result = qsvc.retrieve_only(
            req.query,
            top_k_override=None,
            access_scope=scope,
        )

        before_acl = len(result.evidence_docs)
        filtered_docs, acl_blocked, _blocked_ids = filter_documents_by_acl(
            scope, result.evidence_docs
        )
        acl_filtered_count = acl_blocked
        result.evidence_docs = filtered_docs

        if req.summarize:
            if acl_blocked > 0 and len(filtered_docs) != before_acl:
                logger.info(
                    f"[RAG/query] ACL 过滤后 evidence {before_acl} → {len(filtered_docs)}，"
                    f"仅使用授权 evidence summarize"
                )
            result.answer = _summarize_with_answer_policy(
                query=req.query,
                scope=scope,
                filtered_docs=filtered_docs,
                acl_blocked=acl_blocked,
                qsvc=qsvc,
                rag_svc=rag_svc,
            )
            if hasattr(qsvc, "build_context_from_docs"):
                _context_text, tokens_used, included_count = qsvc.build_context_from_docs(
                    filtered_docs,
                    rag_config=rag_conf.get("rag"),
                )
                result.trace.context_tokens_used = tokens_used
                result.trace.docs_included_in_context = included_count
                if included_count > 0:
                    result.evidence_docs = filtered_docs[:included_count]

        evidence_count = len(result.evidence_docs)

        if not req.summarize:
            return QueryResponse(
                answer="",
                evidence_docs=_docs_to_v1(result.evidence_docs),
            )

        if (result.answer or "").strip():
            qa_cache.put(
                cache_key,
                QaCacheEntry(
                    answer=result.answer or "",
                    evidence_docs=[
                        {
                            "page_content": d.page_content,
                            "metadata": dict(d.metadata or {}),
                        }
                        for d in result.evidence_docs
                    ],
                    trace={
                        "acl_filtered_count": acl_blocked,
                        "context_used_tokens": int(
                            getattr(result.trace, "context_tokens_used", 0) or 0
                        ),
                        "docs_included_count": int(
                            getattr(result.trace, "docs_included_in_context", 0) or 0
                        ),
                    },
                    created_at=time.monotonic(),
                ),
            )

        return QueryResponse(
            answer=result.answer or "",
            evidence_docs=_docs_to_v1(result.evidence_docs),
        )
    except HTTPException:
        status = "err"
        raise
    except Exception as exc:  # pragma: no cover - 防御性
        status = "err"
        logger.exception("[RAG/query] 调用失败")
        raise HTTPException(status_code=500, detail=f"RAG 服务内部错误：{exc}") from exc
    finally:
        req_id = ctx.request_id or "-"
        logger.info(
            f"[审计] {req_id} user={ctx.user_id}({ctx.user_name}) "
            f"dept={ctx.department} roles={ctx.roles} "
            f"scope={ctx.data_scope} "
            f"query={req.query[:80]} evidence={evidence_count} "
            f"acl_filtered={acl_filtered_count} status={status}"
        )


@app.post("/api/rag/retrieve", response_model=RetrieveResponse)
def api_rag_retrieve(req: RetrieveRequest, request: Request):
    """V1 纯检索（兼容旧调用方）：从 ctx 取身份，内部走 QueryService.retrieve_only。（Phase1+Phase2）"""
    status = "ok"
    evidence_count = 0
    acl_filtered_count = 0
    ctx = _get_ctx_from_request(request)
    try:
        scope = _ctx_to_access_scope(ctx)
        qsvc = get_query_svc()

        result = qsvc.retrieve_only(
            req.query,
            top_k_override=req.top_k,
            access_scope=scope,
        )

        filtered_docs, acl_blocked, _blocked_ids = filter_documents_by_acl(
            scope, result.evidence_docs
        )
        acl_filtered_count = acl_blocked
        evidence_count = len(filtered_docs)

        v1_docs = _docs_to_v1(filtered_docs)
        return RetrieveResponse(docs=v1_docs, count=len(v1_docs))
    except HTTPException:
        status = "err"
        raise
    except Exception as exc:  # pragma: no cover
        status = "err"
        logger.exception("[RAG/retrieve] 调用失败")
        raise HTTPException(status_code=500, detail=f"检索服务内部错误：{exc}") from exc
    finally:
        req_id = ctx.request_id or "-"
        logger.info(
            f"[审计] {req_id} user={ctx.user_id}({ctx.user_name}) "
            f"dept={ctx.department} roles={ctx.roles} "
            f"scope={ctx.data_scope} "
            f"query={req.query[:80]} evidence={evidence_count} "
            f"acl_filtered={acl_filtered_count} status={status}"
        )


@app.post("/api/rag/build", response_model=BuildResponse)
def api_rag_build(req: BuildRequest, request: Request):
    """V1 构建（兼容旧调用方）：从 ctx 取身份后鉴权，force=True 真正重建。（Phase1+Phase2）"""
    status = "ok"
    rebuild_count = 0
    deleted_chunk_count = 0
    ctx = _get_ctx_from_request(request)
    try:
        identity_cfg = get_identity_cfg()
        if not is_authorized_for_build(ctx, identity_cfg):
            raise HTTPException(
                status_code=403,
                detail=(
                    "无权限触发知识库重建，需要 build_endpoint_roles 中任一角色："
                    + ", ".join(identity_cfg.build_endpoint_roles)
                ),
            )

        rebuild_count, deleted_chunk_count = _do_build(force=req.force)

        if req.force:
            message = (
                "向量知识库构建完成（force=True 强制重建完成）。"
                f" 重建文档 {rebuild_count} 个，删除旧 chunk {deleted_chunk_count} 条。"
            )
        else:
            message = (
                "向量知识库构建完成（增量，含磁盘已删文件的热删除）。"
                f" 当前 active 文档约 {rebuild_count} 个。"
            )
        return BuildResponse(ok=True, message=message)
    except HTTPException:
        status = "err"
        raise
    except Exception as exc:  # pragma: no cover
        status = "err"
        logger.exception("[RAG/build] 构建失败")
        raise HTTPException(status_code=500, detail=f"构建知识库失败：{exc}") from exc
    finally:
        req_id = ctx.request_id or "-"
        logger.info(
            f"[审计] {req_id} user={ctx.user_id}({ctx.user_name}) "
            f"dept={ctx.department} roles={ctx.roles} "
            f"scope={ctx.data_scope} "
            f"query=<BUILD force={req.force}> evidence={rebuild_count} "
            f"acl_filtered={deleted_chunk_count} status={status}"
        )


# ──────────────────────────────────────────────────────────────────────
# 健康探针 & 配置校验：默认允许无身份访问（作为探针）
# ──────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def api_health():
    """健康检查 + 配置校验摘要。（Phase1：探针类默认放行）"""
    validation = validate_all_configs()
    return {
        "status": "ok" if validation.ok else "config_error",
        "config": validation.to_dict(),
    }


@app.get("/api/config/validate")
def api_config_validate():
    """返回完整的配置校验报告（JSON + 文本版）。（Phase1：探针类默认放行）"""
    validation = validate_all_configs()
    return {
        **validation.to_dict(),
        "report_text": validation.report(),
    }


# ──────────────────────────────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8765,
        reload=False,
    )
