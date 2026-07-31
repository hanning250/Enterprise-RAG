"""BM25 关键词召回模块。"""

from __future__ import annotations

import re
import threading
from typing import Any, Callable, Optional

from langchain_core.documents import Document

from rag.access_policy import AccessScope, filter_candidates_by_acl
from rag.types import RetrievalCandidate
from rag.vector_store import VectorStoreService
from utils.config_handler import rag_conf
from utils.logger_handler import logger


def _jieba_tokenize(text: str) -> list[str]:
    """用 jieba 对中文文本做分词。"""
    import jieba  # type: ignore

    if not text:
        return []

    tokens: list[str] = []

    def add_token(value: str) -> None:
        token = value.strip().lower()
        if token and token not in tokens:
            tokens.append(token)

    for raw_token in jieba.lcut(text):
        token = raw_token.strip().lower()
        if len(token) >= 2 and any(ch.isalnum() or ord(ch) > 127 for ch in token):
            add_token(token)

    for match in re.findall(r"\d{4}年|\d{1,2}月|\d{4}年\d{1,2}月", text):
        add_token(match)

    for field in (
        "实发工资",
        "应发工资",
        "基本工资",
        "绩效工资",
        "岗位津贴",
        "社保",
        "公积金",
        "个税",
        "部门",
        "岗位",
    ):
        if field in text:
            add_token(field)

    for cjk_run in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        for i in range(0, len(cjk_run) - 1):
            add_token(cjk_run[i : i + 2])
    return tokens


class BM25Retriever:
    """BM25 关键词召回器。"""

    def __init__(
        self,
        *,
        tokenizer: Optional[Callable[[str], list[str]]] = None,
        bm25_config: Optional[dict[str, Any]] = None,
        vector_store_service: Optional[VectorStoreService] = None,
    ):
        """初始化 BM25 召回器。"""
        bm25_yaml_cfg = rag_conf.get("bm25", {}) or {}
        self._enabled: bool = bool(bm25_yaml_cfg.get("enabled", True))
        tokenizer_name = str(bm25_yaml_cfg.get("tokenizer", "jieba")).lower()

        if tokenizer is not None:
            self._tokenize = tokenizer
        elif tokenizer_name == "jieba":
            self._ensure_dependency("jieba")
            self._tokenize = _jieba_tokenize
        else:
            logger.warning(f"[BM25] 未知 tokenizer 名称: {tokenizer_name}，回退到 jieba")
            self._ensure_dependency("jieba")
            self._tokenize = _jieba_tokenize

        self._ensure_dependency("rank_bm25", pip_name="rank-bm25")

        self._bm25_extra_params: dict[str, Any] = dict(bm25_config or {})

        self._vector_store = vector_store_service or VectorStoreService()

        self._lock = threading.Lock()
        self._corpus: list[Document] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25_index: Any = None
        self._index_built = False

    @staticmethod
    def _ensure_dependency(module_name: str, pip_name: Optional[str] = None) -> None:
        """检查可选依赖是否已安装，未安装则打印提示。"""
        import importlib
        try:
            importlib.import_module(module_name)
        except ImportError:
            display = pip_name or module_name
            logger.warning(
                f"[BM25] 缺少可选依赖 `{display}`，BM25 召回将被禁用。\n"
                f"       请运行: pip install {display}"
            )

    def _build_index_if_needed(self) -> None:
        """懒加载构建 BM25 索引（线程安全）。"""
        if self._index_built:
            return
        with self._lock:
            if self._index_built:
                return

            if not self._enabled:
                logger.info("[BM25] 配置中 bm25.enabled=false，跳过 BM25 索引构建")
                self._index_built = True
                return

            try:
                raw = self._vector_store.vector_store.get(
                    include=["documents", "metadatas"],
                    where={"chunk_level": {"$eq": "fine"}},
                )
                ids: list[str] = raw.get("ids") or []
                docs_text: list[str] = raw.get("documents") or []
                metas_list: list[dict] = raw.get("metadatas") or []

                chunk_count = len(ids)
                logger.info(f"[BM25] 从 Chroma 拉取到 {chunk_count} 条 chunks，开始构建 BM25 索引")

                if chunk_count == 0:
                    logger.warning("[BM25] Chroma 知识库为空，BM25 索引将是空壳（返回空结果）")
                    self._corpus = []
                    self._tokenized_corpus = []
                    self._bm25_index = None
                    self._index_built = True
                    return

                corpus: list[Document] = []
                for i in range(chunk_count):
                    meta = dict(metas_list[i] or {})
                    if "chunk_id" not in meta and i < len(ids):
                        meta["chunk_id"] = ids[i]
                    corpus.append(
                        Document(
                            page_content=docs_text[i] if i < len(docs_text) else "",
                            metadata=meta,
                        )
                    )

                tokenized_corpus: list[list[str]] = []
                for doc in corpus:
                    tokens = self._tokenize(doc.page_content or "")
                    tokenized_corpus.append(tokens)

                from rank_bm25 import BM25Okapi  # type: ignore
                bm25_index = BM25Okapi(tokenized_corpus, **self._bm25_extra_params)

                self._corpus = corpus
                self._tokenized_corpus = tokenized_corpus
                self._bm25_index = bm25_index
                self._index_built = True
                logger.info(f"[BM25] 索引构建完成，共 {chunk_count} 条 chunks")

            except Exception as exc:
                logger.error(f"[BM25] 索引构建失败：{exc}", exc_info=True)

    def rebuild_index(self) -> None:
        """强制重建 BM25 索引。"""
        with self._lock:
            self._corpus = []
            self._tokenized_corpus = []
            self._bm25_index = None
            self._index_built = False
        logger.info("[BM25] 已清空索引缓存，下次 retrieve 时将重新构建")

    def retrieve(
        self,
        query: str,
        top_k: int,
        *,
        access_scope: Optional[AccessScope] = None,
    ) -> list[RetrievalCandidate]:
        """执行 BM25 关键词召回。"""
        if top_k is None or top_k <= 0:
            return []

        self._build_index_if_needed()

        if not self._enabled or self._bm25_index is None or not self._corpus:
            return []

        try:
            query_tokens = self._tokenize(query or "")
            if not query_tokens:
                return []

            scores = self._bm25_index.get_scores(query_tokens)

            indexed_scores: list[tuple[int, float]] = [
                (i, float(scores[i])) for i in range(len(scores))
            ]
            indexed_scores.sort(key=lambda x: x[1], reverse=True)

            candidates: list[RetrievalCandidate] = []
            rank = 0
            for idx, bm25_score in indexed_scores:
                if bm25_score <= 0:
                    break
                rank += 1
                if rank > top_k:
                    break
                candidates.append(
                    RetrievalCandidate(
                        doc=self._corpus[idx],
                        source="bm25",
                        rank=rank,
                        score=bm25_score,
                    )
                )

            logger.debug(
                f"[BM25] query='{query[:50]}...' 分词={query_tokens}，召回 {len(candidates)}/{top_k} 条"
            )

            if access_scope is not None:
                passed, blocked_count, _blocked_ids = filter_candidates_by_acl(
                    access_scope, candidates
                )
                if blocked_count > 0:
                    for cand in passed:
                        meta = dict(cand.doc.metadata or {})
                        meta["_bm25_acl_blocked_count"] = blocked_count
                        cand.doc.metadata = meta
                    logger.info(
                        f"[BM25] ACL过滤：{len(candidates)} → {len(passed)} 条，过滤掉{blocked_count}条"
                    )
                return passed

            return candidates

        except Exception as exc:
            logger.error(f"[BM25] 召回异常：query='{query[:80]}...' 错误={exc}", exc_info=True)
            return []
