"""RAG 检索候选文档类型模块。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from langchain_core.documents import Document


RetrievalSource = Literal["vector", "bm25", "hybrid"]


@dataclass
class RetrievalCandidate:
    """RAG 检索链路中的候选文档包装类。"""

    doc: Document
    source: RetrievalSource
    rank: int
    score: Optional[float] = None
    fusion_score: Optional[float] = None
    rerank_score: Optional[float] = None

    def get_dedup_key(self) -> str:
        """生成该候选文档的去重稳定键。"""
        meta = self.doc.metadata or {}

        doc_id = meta.get("document_id")
        chunk_idx = meta.get("chunk_index")
        if doc_id is not None and chunk_idx is not None:
            return f"doc:{doc_id}#chunk:{chunk_idx}"

        chunk_id = meta.get("chunk_id")
        if chunk_id:
            return f"chunk_id:{chunk_id}"

        import hashlib
        content_hash = hashlib.md5(
            (self.source + "|" + (self.doc.page_content or "")).encode("utf-8")
        ).hexdigest()
        return f"fallback:{content_hash}"

    def as_document(self) -> Document:
        """把候选文档还原成 LangChain 的 Document 对象。"""
        enriched_meta = dict(self.doc.metadata or {})

        enriched_meta["_retrieval_source"] = self.source
        enriched_meta["_retrieval_rank"] = self.rank
        if self.score is not None:
            enriched_meta["_retrieval_score"] = self.score
        if self.fusion_score is not None:
            enriched_meta["_fusion_score"] = self.fusion_score
        if self.rerank_score is not None:
            enriched_meta["_rerank_score"] = self.rerank_score

        return Document(
            page_content=self.doc.page_content,
            metadata=enriched_meta,
        )
