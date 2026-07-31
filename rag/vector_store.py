"""RAG 向量存储服务：文件扫描 → 解析 → 分片 → 向量写入 → 幂等更新。"""

from __future__ import annotations
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model.factory import embed_model
from rag.document_normalizer import (
    ChunkBindingContext,
    DataCleaningConfig,
    bind_chunk_metadata,
    normalize_chunks,
    normalize_documents,
)
from rag.document_registry import (
    DocumentRegistryStore,
    EMBEDDING_VERSION,
    PARSER_VERSION,
    build_chunking_version,
    canonicalize_source_path,
)
from rag.semantic_chunker import SemanticChunkConfig, semantic_split_documents
from utils.config_handler import chroma_conf, rag_conf
from utils.file_handler import (
    get_file_sha256_hex,
    listdir_with_allowed_type,
    md_loader,
    pdf_loader,
    txt_loader,
)
from ingestion.loaders.docx_loader import docx_loader
from ingestion.loaders.xlsx_loader import xlsx_loader
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


def build_document_id(source_path: str) -> str:
    """基于规范化来源路径生成稳定 document_id。"""
    canonical_path = canonicalize_source_path(source_path)
    digest = hashlib.sha256(canonical_path.encode("utf-8")).hexdigest()
    return f"doc_{digest[:32]}"


def build_document_version(
    *,
    content_hash: str,
    parser_version: str,
    chunking_version: str,
    embedding_model: str,
    embedding_version: str,
) -> str:
    """生成一次入库版本号；内容或处理参数任一变化都会得到新版本。"""
    raw = "|".join(
        [content_hash, parser_version, chunking_version, embedding_model, embedding_version]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def build_chunk_id(document_id: str, document_version: str, chunk_index: int) -> str:
    """生成稳定且版本隔离的 chunk_id。"""
    return f"{document_id}_v_{document_version}_chunk_{chunk_index:05d}"


class VectorStoreService:
    """向量库服务：负责 RAG 文档增量入库和检索器创建。"""

    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=get_abs_path(chroma_conf["persist_directory"]),
        )

        self.chunking_mode = str(chroma_conf.get("chunking_mode", "recursive")).lower()

        # 旧版 recursive 模式（始终实例化，chunking_mode 回退时用）
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=int(chroma_conf.get("chunk_size", 200)),
            chunk_overlap=int(chroma_conf.get("chunk_overlap", 20)),
            separators=list(chroma_conf.get("separators", ["\n\n", "。", ".", "?", "？", "!", " ", ""])),
            length_function=len,
        )

        # 新版 hierarchical 模式（chunking_mode == "hierarchical" 时用）
        self.hierarchical_cfg: SemanticChunkConfig | None = None
        if self.chunking_mode == "hierarchical":
            hier = chroma_conf.get("hierarchical", {}) or {}
            max_fine_chars: int | None = hier.get("max_fine_chars")
            max_section_chars: int | None = hier.get("max_section_chars")
            allow_section_partition = bool(hier.get("allow_section_partition", True))
            self.hierarchical_cfg = SemanticChunkConfig(
                # tokens 预算保持和默认一致；真正生效的是 max_fine_chars / max_section_chars（按 chroma.yml 来）
                max_section_tokens=int(hier.get("max_section_tokens", 900)) if isinstance(hier.get("max_section_tokens"), int) else 900,
                max_fine_tokens=int(hier.get("max_fine_tokens", 240)) if isinstance(hier.get("max_fine_tokens"), int) else 240,
                max_fine_chars=int(max_fine_chars) if max_fine_chars is not None else None,
                max_section_chars=int(max_section_chars) if max_section_chars is not None else None,
                allow_section_partition=allow_section_partition,
            )

        self.registry = DocumentRegistryStore()
        # v2.3：装载 data_cleaning 阈值配置（四层质量门）
        self.data_cleaning_cfg: DataCleaningConfig = DataCleaningConfig.from_rag_conf(rag_conf)
        self.parser_version = self._build_parser_signature()
        self.chunking_version = self._build_chunking_signature()
        self.embedding_model_name = self._resolve_embedding_model_name()
        self.embedding_version = EMBEDDING_VERSION

    def _build_chunking_signature(self) -> str:
        if self.chunking_mode == "hierarchical" and self.hierarchical_cfg:
            # parent_bind=identity_v1：section 绑定键带员工/文档身份，修复工资表撞 section-00000
            return (
                build_chunking_version(
                    chunk_size=int(chroma_conf.get("chunk_size", 200)),
                    chunk_overlap=int(chroma_conf.get("chunk_overlap", 20)),
                    separators=list(chroma_conf.get("separators", ["\n\n", "。", ".", "?", "？", "!", " ", ""])),
                    chunking_mode="hierarchical",
                    max_fine_chars=getattr(self.hierarchical_cfg, "max_fine_chars", None),
                    max_section_chars=getattr(self.hierarchical_cfg, "max_section_chars", None),
                    allow_section_partition=bool(getattr(self.hierarchical_cfg, "allow_section_partition", True)),
                )
                + ";parent_bind=part_v2"
            )
        return build_chunking_version(
            chunk_size=int(chroma_conf.get("chunk_size", 200)),
            chunk_overlap=int(chroma_conf.get("chunk_overlap", 20)),
            separators=list(chroma_conf.get("separators", ["\n\n", "。", ".", "?", "？", "!", " ", ""])),
            chunking_mode="recursive",
        )

    def get_retriever(self):
        """获取向量检索器。top_k 优先读 rag.top_k，回退到 chroma.k。"""
        rag_config = rag_conf.get("rag", {})
        top_k = rag_config.get("top_k")
        if top_k is None or top_k <= 0:
            top_k = chroma_conf["k"]
        return self.vector_store.as_retriever(search_kwargs={"k": int(top_k)})

    def _build_parser_signature(self) -> str:
        vlm_model = chroma_conf.get("vlm", {}).get("model_name", "qwen-vl-ocr")
        # v2.3：把 data_cleaning 配置名 + 关键阈值混进签名，保证改阈值也会触发重新清洗+入库
        cfg = getattr(self, "data_cleaning_cfg", None)
        if cfg is None:
            dc_tag = "cleaning=default"
        else:
            # 把 L1/L3 的几个关键阈值拼进签名（够用；不用全拼，避免签名太长）
            dc_tag = (
                "cleaning="
                f"{cfg.l1_enable}-{cfg.l3_enable}"
                f"_r{cfg.l1_global_repeat_row_ratio:.2f}"
                f"_t{cfg.l1_toc_min_lines}"
                f"_g{cfg.l1_garbage_char_ratio_max:.3f}"
                f"_pc{cfg.l1_page_min_chars}"
                f"_cj{cfg.l3_density_min_cjk_chars}x{cfg.l3_density_min_cjk_ratio:.2f}"
                f"_en{cfg.l3_density_min_ascii_words}"
                f"_sh{cfg.l3_simhash_hamming_max}"
                f"_md{cfg.report_to_metadata}"
            )
        return f"{PARSER_VERSION};pdf_loader=vlm;vlm_model={vlm_model};normalizer=2.4;{dc_tag}"

    @staticmethod
    def _resolve_embedding_model_name() -> str:
        return str(
            getattr(embed_model, "model", None)
            or rag_conf.get("embedding_model_name")
            or chroma_conf.get("embedding_model_name")
            or "unknown"
        )

    @staticmethod
    def _get_file_documents(read_path: str) -> list[Document]:
        """按文件后缀路由到对应 loader：txt → pdf → md/markdown → doc/docx → xls/xlsx → 兜底。"""
        suffix = read_path.lower()
        if suffix.endswith(".txt"):
            return txt_loader(read_path)
        if suffix.endswith(".pdf"):
            return pdf_loader(read_path)
        if suffix.endswith((".md", ".markdown")):
            return md_loader(read_path)
        if suffix.endswith((".doc", ".docx")):
            return docx_loader(read_path)
        if suffix.endswith((".xlsx", ".xls")):
            return xlsx_loader(read_path)
        return []

    def _enrich_business_metadata(self, docs: list[Document], source_path: str) -> list[Document]:
        """为每个 Document 补充业务 metadata 基础字段：department/doc_type/sensitivity_level/owner_role/effective_date 等。"""
        import os
        basename = os.path.basename(source_path)
        basename_lower = basename.lower()

        if ("工资" in basename) or ("salary" in basename_lower) or ("员工工资表" in basename):
            doc_type = "salary"
            sensitivity_level = "confidential"
            owner_role = "self"
            department = ""
        elif ("财务" in basename) or ("财报" in basename) or ("报表" in basename) or ("finance" in basename_lower):
            doc_type = "finance"
            sensitivity_level = "confidential"
            owner_role = "finance_admin"
            department = "财务部"
        elif (
            ("知识库" in basename)
            or ("制度" in basename)
            or ("规章" in basename)
            or ("手册" in basename)
            or ("政策" in basename)
            or ("年假" in basename)
            or ("请假" in basename)
        ):
            doc_type = "policy"
            sensitivity_level = "internal"
            owner_role = "employee"
            department = ""
        elif basename.endswith((".doc", ".docx", ".md", ".markdown", ".txt")):
            doc_type = "policy"
            sensitivity_level = "internal"
            owner_role = "employee"
            department = ""
        else:
            doc_type = "general"
            sensitivity_level = "internal"
            owner_role = "employee"
            department = ""

        for doc in docs:
            meta = doc.metadata
            existing_dt = str(meta.get("doc_type") or "").strip().lower()
            # md/txt/pdf 等格式标签不是业务类型，应用业务推断覆盖
            if (not existing_dt) or existing_dt in {
                "md",
                "markdown",
                "txt",
                "pdf",
                "doc",
                "docx",
                "general",
            }:
                meta["doc_type"] = doc_type
            if "sensitivity_level" not in meta or not meta["sensitivity_level"]:
                meta["sensitivity_level"] = sensitivity_level
            if "owner_role" not in meta or not meta["owner_role"]:
                meta["owner_role"] = owner_role
            if "department" not in meta or not meta["department"]:
                meta["department"] = department
            if "effective_date" not in meta:
                meta["effective_date"] = ""
            if "source" not in meta or not meta["source"]:
                meta["source"] = source_path
            if "file_name" not in meta or not meta["file_name"]:
                meta["file_name"] = basename
            if "parser_version" not in meta or not meta["parser_version"]:
                meta["parser_version"] = self.parser_version
        return docs

    def _same_processing_signature(self, record: dict | None, *, content_hash: str) -> bool:
        if not record:
            return False
        return (
            record.get("content_hash") == content_hash
            and record.get("parser_version") == self.parser_version
            and record.get("chunking_version") == self.chunking_version
            and record.get("embedding_model") == self.embedding_model_name
            and record.get("embedding_version") == self.embedding_version
        )

    def _can_skip(self, record: dict | None, *, content_hash: str) -> bool:
        if not self._same_processing_signature(record, content_hash=content_hash):
            return False
        status = record.get("status")
        if status == "active":
            return True
        if status != "duplicate":
            return False

        duplicate_of = record.get("duplicate_of")
        if not duplicate_of:
            return False
        canonical_record = self.registry.find_by_document_id(str(duplicate_of))
        return bool(canonical_record and canonical_record.get("status") == "active")

    def _chunk_ids_by_document_id(self, document_id: str) -> set[str]:
        try:
            existing = self.vector_store.get(where={"document_id": document_id})
            return set(existing.get("ids", []) or [])
        except Exception as exc:
            logger.error(
                f"[加载知识库]查询 document_id={document_id} 的分片失败：{exc}",
                exc_info=True,
            )
            raise RuntimeError(f"查询旧分片失败：{document_id}") from exc

    def _delete_chunk_ids(self, chunk_ids: set[str] | list[str]) -> int:
        ids = list(chunk_ids)
        if not ids:
            return 0
        try:
            self.vector_store.delete(ids=ids)
            return len(ids)
        except Exception as exc:
            logger.error(f"[加载知识库]删除分片失败：{exc}", exc_info=True)
            raise RuntimeError("删除向量分片失败") from exc

    def _delete_chunks_by_document_id(self, document_id: str) -> int:
        ids_to_delete = self._chunk_ids_by_document_id(document_id)
        deleted = self._delete_chunk_ids(ids_to_delete)
        if deleted:
            logger.info(
                f"[加载知识库]清理 document_id={document_id} 的分片，共删除 {deleted} 条"
            )
        return deleted

    def _delete_chunks_except(self, document_id: str, keep_ids: set[str]) -> int:
        existing_ids = self._chunk_ids_by_document_id(document_id)
        stale_ids = existing_ids - keep_ids
        deleted = self._delete_chunk_ids(stale_ids)
        if deleted:
            logger.info(
                f"[加载知识库]清理 document_id={document_id} 的旧版本分片，共删除 {deleted} 条"
            )
        return deleted

    def _prune_stale_version_chunks(
        self, document_id: str, *, document_version: str
    ) -> int:
        """幂等收敛：保留当前 version 前缀的分片，删除同 document 下其它版本残留。"""
        current_prefix = f"{document_id}_v_{document_version}_"
        doc_prefix = f"{document_id}_v_"
        existing_ids = self._chunk_ids_by_document_id(document_id)
        stale_ids = {
            cid
            for cid in existing_ids
            if cid.startswith(doc_prefix) and not cid.startswith(current_prefix)
        }
        if not stale_ids:
            return 0
        deleted = self._delete_chunk_ids(stale_ids)
        if deleted:
            logger.info(
                f"[加载知识库]幂等清理 document_id={document_id} "
                f"旧版本残留分片 {deleted} 条（keep version={document_version}）"
            )
        return deleted

    def _mark_duplicate_document(
        self,
        *,
        source_path: str,
        document_id: str,
        content_hash: str,
        equivalent_record: dict,
        existing_record: dict | None,
    ) -> None:
        if existing_record and existing_record.get("status") == "active":
            self._delete_chunks_by_document_id(str(existing_record["document_id"]))

        self.registry.upsert_document(
            document_id=document_id,
            source_path=source_path,
            content_hash=content_hash,
            parser_version=self.parser_version,
            chunking_version=self.chunking_version,
            embedding_model=self.embedding_model_name,
            embedding_version=self.embedding_version,
            chunk_count=0,
            status="duplicate",
            duplicate_of=str(equivalent_record["document_id"]),
        )
        logger.info(
            f"[加载知识库]{source_path} 与 {equivalent_record['source_path']} 内容等价，"
            f"标记为 duplicate_of={equivalent_record['document_id']}，跳过重复向量入库"
        )

    def _rollback_new_chunks(self, chunk_ids: list[str]) -> None:
        """仅删除本次新写入的分片；旧版本分片绝不动，避免文档蒸发。"""
        try:
            self._delete_chunk_ids(set(chunk_ids))
        except Exception:
            logger.warning("[加载知识库]回滚新分片失败，可能需要人工清理 Chroma", exc_info=True)

    def _commit_ingested_document(
        self,
        *,
        document_id: str,
        source_path: str,
        content_hash: str,
        split_document: list[Document],
        chunk_ids: list[str],
        quality_report=None,
    ) -> None:
        """幂等安全提交：写新 → registry 记账 → 删旧。

        尚未记账前失败：只回滚新分片，旧分片保持可检索。
        记账成功后清旧失败：保留新分片，旧残留由下次 skip/重建幂等清理。
        """
        registry_committed = False
        try:
            batch_size = 10
            for i in range(0, len(split_document), batch_size):
                self.vector_store.add_documents(
                    split_document[i : i + batch_size],
                    ids=chunk_ids[i : i + batch_size],
                )

            extra_meta: dict = {}
            if quality_report is not None:
                extra_meta = {
                    "l1_documents_input": quality_report.l1_documents_input,
                    "l1_documents_kept": quality_report.l1_documents_kept,
                    "l1_documents_dropped": quality_report.l1_documents_dropped,
                    "l3_chunks_input": quality_report.l3_chunks_input,
                    "l3_chunks_kept": quality_report.l3_chunks_kept,
                    "l3_chunks_dropped": quality_report.l3_chunks_dropped,
                    "l3_dedup_exact": quality_report.l3_dedup_exact,
                    "l3_dedup_simhash": quality_report.l3_dedup_simhash,
                }
            self.registry.upsert_document(
                document_id=document_id,
                source_path=source_path,
                content_hash=content_hash,
                parser_version=self.parser_version,
                chunking_version=self.chunking_version,
                embedding_model=self.embedding_model_name,
                embedding_version=self.embedding_version,
                chunk_count=len(split_document),
                status="active",
                duplicate_of=None,
                extra=extra_meta or None,
            )
            registry_committed = True

            try:
                self._delete_chunks_except(document_id, set(chunk_ids))
            except Exception as cleanup_exc:
                logger.warning(
                    f"[加载知识库]{source_path} 新版本已记账，清理旧分片失败"
                    f"（下次幂等构建会再清）：{cleanup_exc}",
                    exc_info=True,
                )
        except Exception:
            if not registry_committed:
                self._rollback_new_chunks(chunk_ids)
            raise

    def _mark_failed_if_needed(
        self,
        *,
        document_id: str,
        source_path: str,
        content_hash: str,
        existing_record: dict | None,
    ) -> None:
        if existing_record and existing_record.get("status") == "active":
            return
        self.registry.upsert_document(
            document_id=document_id,
            source_path=source_path,
            content_hash=content_hash,
            parser_version=self.parser_version,
            chunking_version=self.chunking_version,
            embedding_model=self.embedding_model_name,
            embedding_version=self.embedding_version,
            chunk_count=0,
            status="failed",
            duplicate_of=None,
        )

    def _split_documents(self, documents: list[Document]) -> list[Document]:
        """根据 chunking_mode 选择切分器输出 chunks。"""
        if self.chunking_mode == "hierarchical" and self.hierarchical_cfg is not None:
            # 新版：父子两级语义切分（section/fine 都输出，细粒检索，粗粒补全）
            return semantic_split_documents(documents, config=self.hierarchical_cfg)
        # 默认旧版：单级 RecursiveCharacterTextSplitter（保留回退能力）
        return self.spliter.split_documents(documents)

    def load_document(self, source_paths: list[str] | None = None) -> None:
        """扫描知识文件并执行幂等增量入库。

        source_paths 为空时扫描 data_path；传入时只重建指定文件，便于定向修复。
        """
        if source_paths is None:
            allowed_files_path: list[str] = listdir_with_allowed_type(
                get_abs_path(chroma_conf["data_path"]),
                tuple(chroma_conf["allow_knowledge_file_type"]),
            )
        else:
            allowed_exts = {
                str(ext).lower().lstrip(".")
                for ext in chroma_conf["allow_knowledge_file_type"]
            }
            allowed_files_path = []
            for path in source_paths:
                abs_path = get_abs_path(path)
                ext = os.path.splitext(abs_path)[1].lower().lstrip(".")
                if ext in allowed_exts:
                    allowed_files_path.append(abs_path)
                else:
                    logger.warning(f"[加载知识库]{abs_path} 文件类型不在白名单内，跳过")

        for raw_path in allowed_files_path:
            source_path = canonicalize_source_path(raw_path)
            content_hash = get_file_sha256_hex(raw_path)
            if not content_hash:
                logger.warning(f"[加载知识库]{raw_path} 哈希计算失败，跳过")
                continue

            document_id = build_document_id(source_path)
            existing = self.registry.find_by_source_path(source_path)

            if self._can_skip(existing, content_hash=content_hash):
                logger.info(
                    f"[加载知识库]{source_path} 内容和处理参数均无变化（document_id={document_id}），跳过"
                )
                # 幂等收敛：上次若「已记账但清旧失败」，此处补清旧版本残留
                if existing and existing.get("status") == "active":
                    try:
                        skip_version = build_document_version(
                            content_hash=content_hash,
                            parser_version=self.parser_version,
                            chunking_version=self.chunking_version,
                            embedding_model=self.embedding_model_name,
                            embedding_version=self.embedding_version,
                        )
                        self._prune_stale_version_chunks(
                            document_id, document_version=skip_version
                        )
                    except Exception as prune_exc:
                        logger.warning(
                            f"[加载知识库]{source_path} 跳过入库时清理旧分片失败："
                            f"{prune_exc}",
                            exc_info=True,
                        )
                continue

            try:
                equivalent = self.registry.find_active_equivalent_document(
                    content_hash=content_hash,
                    parser_version=self.parser_version,
                    chunking_version=self.chunking_version,
                    embedding_model=self.embedding_model_name,
                    embedding_version=self.embedding_version,
                    exclude_source_path=source_path,
                )
                if equivalent and equivalent.get("document_id") != document_id:
                    self._mark_duplicate_document(
                        source_path=source_path,
                        document_id=document_id,
                        content_hash=content_hash,
                        equivalent_record=equivalent,
                        existing_record=existing,
                    )
                    continue

                # v2.3：L1 Document 级清洗（返回 tuple）；Phase3：先 enrich 业务 metadata
                raw_docs = self._get_file_documents(raw_path)
                raw_docs = self._enrich_business_metadata(raw_docs, source_path)
                norm_result = normalize_documents(
                    raw_docs,
                    config=self.data_cleaning_cfg,
                )
                if isinstance(norm_result, tuple) and len(norm_result) == 2:
                    documents, l1_report = norm_result
                else:  # 兼容：旧接口只返回 list[Document]
                    documents, l1_report = list(norm_result), None

                # L4 QualityReport：L1 报告先打一行
                if l1_report is not None:
                    logger.info(
                        f"[L1清洗]{source_path} {l1_report.summary()}"
                        + (
                            f" | DropReasons={dict(l1_report.l1_drop_reasons)}"
                            if l1_report.l1_documents_dropped
                            else ""
                        )
                    )
                    if l1_report.l1_documents_input > 0 and l1_report.l1_kept_ratio < self.data_cleaning_cfg.l4_kept_ratio_warn:
                        logger.warning(
                            f"[L1清洗告警]{source_path} 文档保留率仅 {l1_report.l1_kept_ratio:.2%}，"
                            f"低于阈值 {self.data_cleaning_cfg.l4_kept_ratio_warn:.2%}，"
                            f"建议人工检查原文质量或放松 rag.yml data_cleaning.l1_* 阈值"
                        )

                if not documents:
                    logger.warning(f"[加载知识库]{source_path} L1 清洗后没有有效文本内容，跳过")
                    self._mark_failed_if_needed(
                        document_id=document_id,
                        source_path=source_path,
                        content_hash=content_hash,
                        existing_record=existing,
                    )
                    continue

                # L2 噪声清理：已在 semantic_chunker._extract_sections 内部执行（复用 strip_l2_noise_line）
                split_chunks_raw = self._split_documents(documents)

                # v2.3：L3 Chunk 级清洗（返回 tuple；把 l1_report 传进去合并）
                norm3_result = normalize_chunks(
                    split_chunks_raw,
                    config=self.data_cleaning_cfg,
                    chunks_report=l1_report,
                )
                if isinstance(norm3_result, tuple) and len(norm3_result) == 2:
                    split_document, quality_report = norm3_result
                else:
                    split_document, quality_report = list(norm3_result), None

                # L4 QualityReport：L3 报告 + 告警
                if quality_report is not None:
                    logger.info(
                        f"[L3清洗]{source_path} {quality_report.summary()}"
                        + (
                            f" | DropReasons={dict(quality_report.l3_drop_reasons)}"
                            if quality_report.l3_chunks_dropped
                            else ""
                        )
                    )
                    if quality_report.l3_chunks_input > 0 and quality_report.l3_kept_ratio < self.data_cleaning_cfg.l4_kept_ratio_warn:
                        logger.warning(
                            f"[L3清洗告警]{source_path} Chunk保留率仅 {quality_report.l3_kept_ratio:.2%}，"
                            f"低于阈值 {self.data_cleaning_cfg.l4_kept_ratio_warn:.2%}，"
                            f"建议检查 l3_density_* 阈值是否过严"
                        )
                    if (
                        quality_report.l3_chunks_input > 0
                        and quality_report.l3_dedup_ratio > self.data_cleaning_cfg.l4_dedup_ratio_warn
                    ):
                        logger.warning(
                            f"[L3去重告警]{source_path} 去重占比 {quality_report.l3_dedup_ratio:.2%}，"
                            f"高于阈值 {self.data_cleaning_cfg.l4_dedup_ratio_warn:.2%}，"
                            f"文档内部冗余过高，建议合并源文件"
                        )

                if not split_document:
                    logger.warning(f"[加载知识库]{source_path} 分片清洗后没有有效文本内容，跳过")
                    self._mark_failed_if_needed(
                        document_id=document_id,
                        source_path=source_path,
                        content_hash=content_hash,
                        existing_record=existing,
                    )
                    continue

                document_version = build_document_version(
                    content_hash=content_hash,
                    parser_version=self.parser_version,
                    chunking_version=self.chunking_version,
                    embedding_model=self.embedding_model_name,
                    embedding_version=self.embedding_version,
                )
                split_document, chunk_ids = bind_chunk_metadata(
                    split_document,
                    context=ChunkBindingContext(
                        document_id=document_id,
                        document_version=document_version,
                        source_path=source_path,
                        content_hash=content_hash,
                        parser_version=self.parser_version,
                        chunking_version=self.chunking_version,
                        embedding_model=self.embedding_model_name,
                        embedding_version=self.embedding_version,
                    ),
                    build_chunk_id=build_chunk_id,
                )

                self._commit_ingested_document(
                    document_id=document_id,
                    source_path=source_path,
                    content_hash=content_hash,
                    split_document=split_document,
                    chunk_ids=chunk_ids,
                    quality_report=quality_report,
                )

                logger.info(
                    f"[加载知识库]{source_path} 加载成功，共 {len(split_document)} 个分片"
                    f"（chunking_mode={self.chunking_mode}, document_id={document_id}, version={document_version}）"
                )
            except Exception as exc:
                logger.error(f"[加载知识库]{source_path} 加载失败：{exc}", exc_info=True)
                self._mark_failed_if_needed(
                    document_id=document_id,
                    source_path=source_path,
                    content_hash=content_hash,
                    existing_record=existing,
                )
                continue

        # 全量扫描时：磁盘已删除的文件做热删除（Chroma chunk + registry）
        if source_paths is None:
            alive = {canonicalize_source_path(p) for p in allowed_files_path}
            self.purge_missing_documents(alive_source_paths=alive)

    def purge_missing_documents(
        self, *, alive_source_paths: set[str] | None = None
    ) -> tuple[int, int]:
        """删除源文件已不在磁盘上的索引（热删除）。

        Returns
        -------
        (purged_doc_count, deleted_chunk_count)
        """
        alive = alive_source_paths
        if alive is None:
            allowed = listdir_with_allowed_type(
                get_abs_path(chroma_conf["data_path"]),
                tuple(chroma_conf["allow_knowledge_file_type"]),
            )
            alive = {canonicalize_source_path(p) for p in allowed}

        purged_docs = 0
        deleted_chunks = 0
        # active / rebuild_pending / failed 都可能仍占着向量；duplicate 一般无独立 chunk
        for rec in self.registry.list_all():
            status = str(rec.get("status") or "")
            if status == "deleted":
                continue
            source_path = str(rec.get("source_path") or "")
            document_id = str(rec.get("document_id") or "")
            if not source_path or not document_id:
                continue
            canonical = canonicalize_source_path(source_path)
            if canonical in alive:
                continue
            # 二次确认：路径不在扫描结果里，且磁盘上也不存在
            if Path(source_path).exists() or Path(canonical).exists():
                continue
            try:
                deleted_chunks += self._delete_chunks_by_document_id(document_id)
            except Exception as exc:
                logger.warning(
                    f"[热删除] 清理 document_id={document_id} chunk 失败：{exc}"
                )
                continue
            self.registry.set_status(document_id, "deleted")
            purged_docs += 1
            logger.info(
                f"[热删除] 源文件已不存在，已下线 document_id={document_id} "
                f"path={source_path}"
            )

        if purged_docs:
            logger.info(
                f"[热删除] 完成：下线文档 {purged_docs} 个，删除 chunk {deleted_chunks} 条"
            )
        return purged_docs, deleted_chunks


if __name__ == "__main__":
    vs = VectorStoreService()
    vs.load_document()
    retriever = vs.get_retriever()
    print(retriever.invoke("2026年3月工资"))
