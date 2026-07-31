"""FastAPI 请求 / 响应 Schema 定义。

集中管理 RAG API 的 Pydantic 模型，统一结构化身份字段与可观测性指标。
所有 V2 版本模型都显式携带 identity 信息，用于：
  - 写入审计日志
  - 传入检索链路做 ACL 过滤
  - 在响应中快照回显，便于端到端排障
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IdentityInfo(BaseModel):
    """调用方身份信息。

    用途：写入审计日志、传入检索链路做 ACL 过滤。
    """

    user_id: str = Field(..., description="用户唯一标识（必填）")
    user_name: str = Field("", description="用户显示名称")
    department: str = Field("", description="用户所属部门")
    roles: list[str] = Field(default_factory=list, description="用户角色列表，用于权限判定")
    data_scope: str = Field(
        "self",
        description="数据可见范围，默认 self（仅本人数据），可扩展为 department / all 等",
    )
    client_ip: str = Field("", description="调用方客户端 IP，用于审计")
    request_id: str = Field("", description="全链路追踪 ID，方便日志串联")


class QueryRequestV2(BaseModel):
    """完整 RAG 问答请求（V2）。

    替代原 QueryRequest，显式携带身份信息。
    summarize=False 时 answer 字段为空，等价于纯检索。
    """

    query: str = Field(..., min_length=1, description="用户问题（必填，至少 1 个字符）")
    summarize: bool = Field(
        True,
        description="是否调用 LLM 做总结；False 时等价于纯检索（只返回 evidence_docs）",
    )
    identity: IdentityInfo = Field(..., description="调用方身份信息，用于 ACL 与审计")


class RetrieveRequestV2(BaseModel):
    """纯检索请求（V2）。

    top_k 这次真正透传给检索器，覆盖配置文件 retrieval.final_top_k。
    """

    query: str = Field(..., min_length=1, description="用户问题（必填，至少 1 个字符）")
    top_k: int | None = Field(
        None,
        description="覆盖 config retrieval.final_top_k，真正生效；None 时使用配置值",
    )
    identity: IdentityInfo = Field(..., description="调用方身份信息，用于 ACL 与审计")


class BuildRequestV2(BaseModel):
    """向量知识库构建 / 重建请求（V2）。

    force=True 时调用 vector_store 的强制重建逻辑：
    先删除 registry 中该文档版本，再重新入库，而不是只清理失效 chunk。
    """

    force: bool = Field(
        False,
        description="是否强制重建：True 时先删 registry 中对应文档版本再重建，False 走增量",
    )
    identity: IdentityInfo = Field(..., description="调用方身份信息，用于审计")


class RetrievedDoc(BaseModel):
    """单条检索命中的文档片段。

    与 main.py 中定义保持一致：page_content + 自由 metadata。
    """

    page_content: str = Field(..., description="文档片段的正文内容")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="文档元数据（来源文件、页码、标题、版本号等）",
    )


class QueryTrace(BaseModel):
    """一次 query 的可观测性指标。

    全部数据来自检索链路，不伪造。用于排障、性能分析与效果回归。
    """

    retriever_used: str = Field(
        ...,
        description="实际使用的检索器：hybrid / vector / bm25",
        pattern=r"^(hybrid|vector|bm25)$",
    )
    rerank_used: bool = Field(..., description="是否启用了 reranker 重排")
    acl_filtered_count: int = Field(
        0,
        ge=0,
        description="ACL 在召回前过滤掉的候选文档数量",
    )
    context_used_tokens: int = Field(
        0,
        ge=0,
        description="实际送入 LLM prompt 的上下文 token 数",
    )
    docs_included_count: int = Field(
        0,
        ge=0,
        description="实际纳入 LLM context 的文档片段数量",
    )
    retrieval_ms: int | None = Field(
        None,
        ge=0,
        description="检索阶段耗时（毫秒），含召回 + 重排 + ACL",
    )
    answer_ms: int | None = Field(
        None,
        ge=0,
        description="LLM 生成答案耗时（毫秒）",
    )
    answerability_failed_count: int = Field(
        0,
        ge=0,
        description="可答性校验失败、被丢弃的候选 chunk 数",
    )
    second_pass_applied: bool = Field(
        False,
        description="是否触发了二次检索（上下文扩展 / 补充召回）",
    )
    fallback_reason: str | None = Field(
        None,
        description="降级 / 回退触发原因，例如：bm25_failover、rerank_timeout 等",
    )
    cache_hit: bool = Field(
        False,
        description="是否命中进程内 QA 缓存（跳过检索与 LLM）",
    )


class QueryResponseV2(BaseModel):
    """完整 RAG 问答响应（V2）。

    answer 为空时等价于纯检索。
    evidence_docs 必须与 LLM 真正用的 context 同源，不可包含未使用的冗余片段。
    """

    answer: str = Field(
        ...,
        description="LLM 总结后的回答；空字符串表示 summarize=False 或纯检索模式",
    )
    evidence_docs: list[RetrievedDoc] = Field(
        default_factory=list,
        description="LLM 生成时实际使用的证据文档（与 context 同源）",
    )
    trace: QueryTrace | None = Field(
        None,
        description="本次调用的可观测性指标；排障模式下返回",
    )
    identity_snapshot: IdentityInfo | None = Field(
        None,
        description="请求中携带的身份快照，便于端到端对账与排障",
    )


class RetrieveResponseV2(BaseModel):
    """纯检索响应（V2）。"""

    docs: list[RetrievedDoc] = Field(
        default_factory=list,
        description="按相关度排序的检索结果列表",
    )
    count: int = Field(
        ...,
        ge=0,
        description="实际返回的文档数量（等于 len(docs)）",
    )
    identity_snapshot: IdentityInfo | None = Field(
        None,
        description="请求中携带的身份快照",
    )


class BuildResponseV2(BaseModel):
    """向量知识库构建响应（V2）。"""

    ok: bool = Field(..., description="构建是否成功")
    message: str = Field(..., description="构建结果的可读描述")
    rebuild_count: int = Field(
        0,
        ge=0,
        description="本次实际重建（或重新入库）了多少个源文件",
    )
    deleted_chunk_count: int = Field(
        0,
        ge=0,
        description="本次强制重建时删除的旧版本 chunk 数量",
    )
    identity_snapshot: IdentityInfo | None = Field(
        None,
        description="请求中携带的身份快照",
    )
