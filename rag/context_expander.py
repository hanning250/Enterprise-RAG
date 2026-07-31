"""检索后上下文补全模块。"""

from __future__ import annotations

import re

from langchain_core.documents import Document

from rag.context_budget import estimate_tokens
from utils.config_handler import rag_conf
from utils.logger_handler import logger


def _resolve_context_expansion_config() -> dict:
    cfg = rag_conf.get("context_expansion", {}) or {}
    proc_keywords_raw = cfg.get("procedural_keywords") or [
        "怎么", "如何", "步骤", "预约", "流程", "操作", "做法", "指南",
        "how to", "steps",
    ]
    if isinstance(proc_keywords_raw, str):
        proc_keywords = [proc_keywords_raw.strip()]
    else:
        proc_keywords = [str(x).strip() for x in proc_keywords_raw if str(x).strip()]
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "include_parent": bool(cfg.get("include_parent", True)),
        "max_parent_tokens": int(cfg.get("max_parent_tokens", 900)),
        "max_parent_chars": (
            int(cfg["max_parent_chars"])
            if isinstance(cfg.get("max_parent_chars"), int) and int(cfg["max_parent_chars"]) > 0
            else None
        ),
        "procedural_extra_enable": bool(cfg.get("procedural_extra_enable", False)),
        "procedural_keywords": proc_keywords,
        "procedural_extra_max_sections": int(cfg.get("procedural_extra_max_sections", 2)),
        "procedural_extra_max_tokens": int(cfg.get("procedural_extra_max_tokens", 900)),
    }


_PROCEDURAL_QUERY_CACHE: dict[str, bool] = {}


def _is_procedural_query(query: str, keywords: list[str]) -> bool:
    """关键词判步骤型问题（结果按字符串做全局缓存，省得每次都 re）。"""
    if not query:
        return False
    if query in _PROCEDURAL_QUERY_CACHE:
        return _PROCEDURAL_QUERY_CACHE[query]
    q = query.strip().lower()
    hit = False
    for kw in keywords:
        if not kw:
            continue
        if re.search(re.escape(kw.lower()), q):
            hit = True
            break
    _PROCEDURAL_QUERY_CACHE[query] = hit
    return hit


class ContextExpander:
    """根据 chunk 元数据补全父级上下文；步骤型问题额外补相邻 section。"""

    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.cfg = _resolve_context_expansion_config()

    def expand(self, docs: list[Document], query: str = "") -> list[Document]:
        """把命中的 fine chunk 替换/补充为 parent section。

        参数 query 用于步骤型问题识别；为空时回退为纯 parent 补全（兼容旧调用方）。
        """
        if not self.cfg["enabled"] or not self.cfg["include_parent"]:
            return docs

        expanded: list[Document] = []
        seen_keys: set[str] = set()
        parent_cache: dict[str, Document | None] = {}

        for doc in docs:
            meta = doc.metadata or {}
            # 工资 fine chunk 已是完整行记录；xlsx 常把 parent 误绑到文件首条 section，expand 会换人。
            if str(meta.get("doc_type") or "").strip().lower() == "salary":
                self._append_once(expanded, seen_keys, doc)
                continue

            parent_id = str(meta.get("parent_chunk_id") or "")
            if parent_id:
                parent_doc = parent_cache.get(parent_id)
                if parent_id not in parent_cache:
                    parent_doc = self._get_parent(parent_id)
                    parent_cache[parent_id] = parent_doc
                if parent_doc is not None and self._parent_compatible(doc, parent_doc):
                    self._append_once(expanded, seen_keys, self._mark_expanded_parent(parent_doc, doc))
                    continue
            self._append_once(expanded, seen_keys, doc)

        if len(expanded) != len(docs):
            logger.info(
                f"[ContextExpander] 上下文补全完成：输入 {len(docs)} 条，输出 {len(expanded)} 条"
            )

        # 步骤型问题：额外追加同一 source 的相邻 section（比如实用提示里的操作步骤）
        if self.cfg["procedural_extra_enable"] and _is_procedural_query(
            query, self.cfg["procedural_keywords"]
        ):
            extras = self._fetch_procedural_extra_sections(expanded, seen_keys)
            if extras:
                logger.info(
                    f"[ContextExpander] 步骤型问题额外补全：query={query[:30]!r} 新增 {len(extras)} 条相邻 section"
                )
                expanded.extend(extras)

        return expanded

    @staticmethod
    def _parent_compatible(source_doc: Document, parent_doc: Document) -> bool:
        """父节与子 chunk 必须身份一致且内容相关，否则保留 fine。"""
        if not ContextExpander._parent_identity_compatible(source_doc, parent_doc):
            return False
        return ContextExpander._parent_content_compatible(source_doc, parent_doc)

    @staticmethod
    def _parent_identity_compatible(source_doc: Document, parent_doc: Document) -> bool:
        """父节与子 chunk 身份冲突时拒绝替换（防工资表等误绑 parent）。"""
        src = source_doc.metadata or {}
        par = parent_doc.metadata or {}
        src_emp = str(src.get("employee_name") or "").strip()
        par_emp = str(par.get("employee_name") or "").strip()
        if src_emp and par_emp and src_emp != par_emp:
            logger.warning(
                f"[ContextExpander] parent 员工不一致，保留 fine："
                f"fine={src_emp} parent={par_emp} parent_id={par.get('chunk_id')}"
            )
            return False
        return True

    @staticmethod
    def _parent_content_compatible(source_doc: Document, parent_doc: Document) -> bool:
        """parent 正文几乎不含 fine 时拒绝替换（防长节被切段后一律绑到 ::0）。"""
        fine = (source_doc.page_content or "").strip()
        parent = (parent_doc.page_content or "").strip()
        if not fine or not parent:
            return True
        if fine in parent:
            return True
        window = 24 if len(fine) >= 24 else max(8, len(fine) // 2 or 1)
        probes: list[str] = []
        if len(fine) <= window:
            probes = [fine]
        else:
            mid = max(0, len(fine) // 2 - window // 2)
            probes = [fine[:window], fine[mid : mid + window], fine[-window:]]
        if any(p and p in parent for p in probes):
            return True
        logger.warning(
            f"[ContextExpander] parent 与 fine 内容不重叠，保留 fine："
            f"fine_chunk={ (source_doc.metadata or {}).get('chunk_id') } "
            f"parent_id={ (parent_doc.metadata or {}).get('chunk_id') }"
        )
        return False

    def _get_parent(self, parent_chunk_id: str) -> Document | None:
        try:
            result = self.vector_store.get(ids=[parent_chunk_id], include=["documents", "metadatas"])
        except Exception as exc:
            logger.warning(f"[ContextExpander] 读取 parent_chunk_id={parent_chunk_id} 失败：{exc}")
            return None

        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        if not documents:
            return None

        text = str(documents[0] or "")
        if estimate_tokens(text) > self.cfg["max_parent_tokens"]:
            return None
        max_chars = self.cfg.get("max_parent_chars")
        if isinstance(max_chars, int) and max_chars > 0 and len(text) > max_chars:
            return None

        metadata = dict(metadatas[0] or {})
        return Document(page_content=text, metadata=metadata)

    @staticmethod
    def _doc_key(doc: Document) -> str:
        meta = doc.metadata or {}
        chunk_id = meta.get("chunk_id")
        if chunk_id:
            return f"chunk:{chunk_id}"
        return f"text:{hash(doc.page_content)}"

    @classmethod
    def _append_once(cls, docs: list[Document], seen_keys: set[str], doc: Document) -> None:
        key = cls._doc_key(doc)
        if key in seen_keys:
            return
        seen_keys.add(key)
        docs.append(doc)

    @staticmethod
    def _mark_expanded_parent(parent_doc: Document, source_doc: Document) -> Document:
        source_meta = source_doc.metadata or {}
        parent_meta = dict(parent_doc.metadata or {})
        parent_meta["_context_expanded_from_chunk_id"] = source_meta.get("chunk_id", "")
        parent_meta["_context_expansion"] = "parent_section"
        for key in ("_retrieval_source", "_retrieval_rank", "_retrieval_score", "_fusion_score", "_rerank_score"):
            if key in source_meta:
                parent_meta[key] = source_meta[key]
        return Document(page_content=parent_doc.page_content, metadata=parent_meta)

    # ============================================================
    # 步骤型问题：额外补同一 source 的相邻 section
    # ============================================================
    def _fetch_procedural_extra_sections(
        self,
        expanded: list[Document],
        seen_keys: set[str],
    ) -> list[Document]:
        """为每个 parent section 找同一 source_path 的 page 相邻 section，追加不超过上限。"""
        max_extra = max(0, int(self.cfg["procedural_extra_max_sections"]))
        if max_extra <= 0:
            return []

        # 收集已存在的 source_path -> {page set}
        source_pages: dict[str, set[int]] = {}
        for doc in expanded:
            meta = doc.metadata or {}
            src = str(meta.get("source_path") or "")
            if not src:
                continue
            p = meta.get("page")
            if isinstance(p, int):
                source_pages.setdefault(src, set()).add(p)
            ps = meta.get("page_start")
            if isinstance(ps, int):
                source_pages[src].add(ps)
            pe = meta.get("page_end")
            if isinstance(pe, int):
                source_pages[src].add(pe)
        if not source_pages:
            return []

        # 目标 page = 已存在 page 的 ±1 范围
        target_pages: dict[str, list[int]] = {}
        for src, pages in source_pages.items():
            wanted: list[int] = []
            for p in sorted(pages):
                wanted.extend([p - 1, p + 1])
            # 去重 + 过滤负页码
            seen_want: set[int] = set()
            ordered: list[int] = []
            for w in wanted:
                if w <= 0 or w in seen_want or w in pages:
                    continue
                seen_want.add(w)
                ordered.append(w)
            if ordered:
                target_pages[src] = ordered
        if not target_pages:
            return []

        extras: list[Document] = []
        added_count = 0
        # 按 source_path 批量查询（每个 source 只查一次）
        for src, pages in target_pages.items():
            if added_count >= max_extra * len(expanded):
                break
            found = self._list_sections_by_source_and_pages(
                source_path=src,
                pages=pages,
                exclude_seen_keys=seen_keys,
            )
            if not found:
                continue
            for doc in found:
                if added_count >= max_extra * len(expanded):
                    break
                if self._doc_key(doc) in seen_keys:
                    continue
                seen_keys.add(self._doc_key(doc))
                meta = dict(doc.metadata or {})
                meta["_context_expansion"] = "procedural_adjacent_section"
                doc2 = Document(page_content=doc.page_content, metadata=meta)
                extras.append(doc2)
                added_count += 1
        return extras

    def _list_sections_by_source_and_pages(
        self,
        source_path: str,
        pages: list[int],
        exclude_seen_keys: set[str],
    ) -> list[Document]:
        """在 vector_store 里按 source_path + chunk_level=section + page 做过滤查询。"""
        try:
            max_tokens = int(self.cfg["procedural_extra_max_tokens"])
        except Exception:
            max_tokens = 900
        if not pages:
            return []
        # 尝试 vector_store.get + 客户端过滤（Chroma 的 get 支持 where 子句）
        where_clause = {
            "$and": [
                {"source_path": {"$eq": source_path}},
                {"chunk_level": {"$eq": "section"}},
                {"page": {"$in": pages}},
            ]
        }
        try:
            result = self.vector_store.get(
                where=where_clause,
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            logger.warning(
                f"[ContextExpander] 步骤型扩展相邻 section 查询失败，fallback 不追加：{exc}"
            )
            return []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        out: list[Document] = []
        for text, meta_raw in zip(documents, metadatas):
            text = str(text or "")
            if not text.strip():
                continue
            if estimate_tokens(text) > max_tokens:
                continue
            meta = dict(meta_raw or {})
            doc = Document(page_content=text, metadata=meta)
            if self._doc_key(doc) in exclude_seen_keys:
                continue
            out.append(doc)
        # 按 page 升序返回（相邻 section 按原文顺序）
        out.sort(key=lambda d: int((d.metadata or {}).get("page") or 0))
        return out