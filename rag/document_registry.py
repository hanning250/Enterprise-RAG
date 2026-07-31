"""RAG 文档注册表模块。"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.path_tool import get_abs_path
from utils.config_handler import chroma_conf

_default_registry_path = chroma_conf.get(
    "registry_db_path", "data/rag_document_registry.sqlite3"
)
DEFAULT_REGISTRY_DB_PATH = get_abs_path(_default_registry_path)
_UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

PARSER_VERSION = str(chroma_conf.get("parser_version", "1.0"))
EMBEDDING_VERSION = str(chroma_conf.get("embedding_version", "1.0"))


def _utc_now() -> str:
    """获取当前 UTC 时间并格式化为标准字符串。"""
    return datetime.now(timezone.utc).strftime(_UTC_FORMAT)


def canonicalize_source_path(source_path: str) -> str:
    """把来源路径规范化为 registry 使用的稳定字符串。"""
    normalized = Path(source_path).resolve(strict=False).as_posix()
    return normalized.casefold() if os.name == "nt" else normalized


def build_chunking_version(chunk_size: int, chunk_overlap: int, separators: list[str], *,
                           chunking_mode: str = "recursive",
                           max_fine_chars: int | None = None,
                           max_section_chars: int | None = None,
                           allow_section_partition: bool | None = None) -> str:
    """根据分片参数构造稳定的分片策略版本号。"""
    sep_str = "|".join(separators)
    base = f"mode={chunking_mode};cs={chunk_size};co={chunk_overlap};sep={sep_str}"
    if chunking_mode == "hierarchical":
        base += f";fine_chars={max_fine_chars};section_chars={max_section_chars};allow_part={1 if allow_section_partition else 0}"
    return base


class DocumentRegistryStore:
    """RAG 文档注册表：使用 SQLite 记录每个文档的入库状态。"""

    def __init__(self, db_path: str | None = None):
        self.db_path = Path(db_path or DEFAULT_REGISTRY_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        """创建并返回配置良好的 SQLite 连接。"""
        connection = sqlite3.connect(str(self.db_path), timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON;")
        connection.execute("PRAGMA journal_mode = WAL;")
        return connection

    def _ensure_schema(self) -> None:
        """确保 rag_document_registry 表和索引已存在。"""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_document_registry (
                    document_id      TEXT PRIMARY KEY,
                    source_path      TEXT NOT NULL,
                    content_hash     TEXT NOT NULL,
                    parser_version   TEXT NOT NULL,
                    chunking_version TEXT NOT NULL,
                    embedding_model  TEXT NOT NULL,
                    embedding_version TEXT NOT NULL,
                    chunk_count      INTEGER NOT NULL DEFAULT 0,
                    status           TEXT NOT NULL DEFAULT 'active',
                    duplicate_of     TEXT,
                    created_at       TEXT NOT NULL,
                    updated_at       TEXT NOT NULL
                )
                """
            )
            columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(rag_document_registry)")
            }
            if "duplicate_of" not in columns:
                conn.execute("ALTER TABLE rag_document_registry ADD COLUMN duplicate_of TEXT")
            if "quality_report_json" not in columns:
                conn.execute("ALTER TABLE rag_document_registry ADD COLUMN quality_report_json TEXT")

            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_registry_source_path "
                "ON rag_document_registry(source_path)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_registry_status "
                "ON rag_document_registry(status)"
            )

    def find_by_source_path(self, source_path: str) -> dict[str, Any] | None:
        """按文件绝对路径查找一条 registry 记录。"""
        canonical_path = canonicalize_source_path(source_path)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM rag_document_registry WHERE source_path = ?",
                (canonical_path,),
            ).fetchone()
            if row is None and source_path != canonical_path:
                row = conn.execute(
                    "SELECT * FROM rag_document_registry WHERE source_path = ?",
                    (source_path,),
                ).fetchone()
        return dict(row) if row else None

    def find_by_document_id(self, document_id: str) -> dict[str, Any] | None:
        """按稳定 document_id 查找一条 registry 记录。"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM rag_document_registry WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return dict(row) if row else None

    def find_active_equivalent_document(
        self,
        *,
        content_hash: str,
        parser_version: str,
        chunking_version: str,
        embedding_model: str,
        embedding_version: str,
        exclude_source_path: str | None = None,
    ) -> dict[str, Any] | None:
        """查找已经 active 的等价文档。"""
        query = """
            SELECT * FROM rag_document_registry
            WHERE content_hash = ?
              AND parser_version = ?
              AND chunking_version = ?
              AND embedding_model = ?
              AND embedding_version = ?
              AND status = 'active'
        """
        params: list[Any] = [
            content_hash,
            parser_version,
            chunking_version,
            embedding_model,
            embedding_version,
        ]
        if exclude_source_path is not None:
            query += " AND source_path != ?"
            params.append(canonicalize_source_path(exclude_source_path))
        query += " ORDER BY updated_at DESC LIMIT 1"

        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()
        return dict(row) if row else None

    def list_all(self, status: str | None = None) -> list[dict[str, Any]]:
        """列出所有 registry 记录，可按 status 过滤。"""
        query = "SELECT * FROM rag_document_registry"
        params: list[Any] = []
        if status is not None:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY updated_at DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def upsert_document(
        self,
        *,
        document_id: str,
        source_path: str,
        content_hash: str,
        parser_version: str,
        chunking_version: str,
        embedding_model: str,
        embedding_version: str,
        chunk_count: int,
        status: str = "active",
        duplicate_of: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """插入或更新一条文档 registry 记录。

        extra 里的字段会被序列化成 JSON 存到 quality_report_json 列，
        用于持久化数据清洗质量报告（L1/L3 指标）。
        """
        source_path = canonicalize_source_path(source_path)
        existing = self.find_by_source_path(source_path)
        now = _utc_now()
        quality_json: str | None = None
        if extra:
            try:
                quality_json = json.dumps(extra, ensure_ascii=False)
            except Exception:
                quality_json = None

        with self._connect() as conn:
            if existing:
                created_at = existing["created_at"]
                conn.execute(
                    """
                    UPDATE rag_document_registry
                    SET source_path = ?,
                        content_hash = ?,
                        parser_version = ?,
                        chunking_version = ?,
                        embedding_model = ?,
                        embedding_version = ?,
                        chunk_count = ?,
                        status = ?,
                        duplicate_of = ?,
                        quality_report_json = ?,
                        updated_at = ?
                    WHERE document_id = ?
                    """,
                    (
                        source_path,
                        content_hash,
                        parser_version,
                        chunking_version,
                        embedding_model,
                        embedding_version,
                        chunk_count,
                        status,
                        duplicate_of,
                        quality_json,
                        now,
                        existing["document_id"],
                    ),
                )
                return {
                    "document_id": existing["document_id"],
                    "source_path": source_path,
                    "content_hash": content_hash,
                    "parser_version": parser_version,
                    "chunking_version": chunking_version,
                    "embedding_model": embedding_model,
                    "embedding_version": embedding_version,
                    "chunk_count": chunk_count,
                    "status": status,
                    "duplicate_of": duplicate_of,
                    "quality_report_json": quality_json,
                    "created_at": created_at,
                    "updated_at": now,
                }
            else:
                created_at = now
                updated_at = now
                conn.execute(
                    """
                    INSERT INTO rag_document_registry (
                        document_id, source_path, content_hash,
                        parser_version, chunking_version,
                        embedding_model, embedding_version,
                        chunk_count, status, duplicate_of,
                        quality_report_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        source_path,
                        content_hash,
                        parser_version,
                        chunking_version,
                        embedding_model,
                        embedding_version,
                        chunk_count,
                        status,
                        duplicate_of,
                        quality_json,
                        created_at,
                        updated_at,
                    ),
                )
                return {
                    "document_id": document_id,
                    "source_path": source_path,
                    "content_hash": content_hash,
                    "parser_version": parser_version,
                    "chunking_version": chunking_version,
                    "embedding_model": embedding_model,
                    "embedding_version": embedding_version,
                    "chunk_count": chunk_count,
                    "status": status,
                    "duplicate_of": duplicate_of,
                    "quality_report_json": quality_json,
                    "created_at": created_at,
                    "updated_at": updated_at,
                }

    def set_status(self, document_id: str, status: str) -> bool:
        """修改某条记录的状态。"""
        now = _utc_now()
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE rag_document_registry SET status = ?, updated_at = ? WHERE document_id = ?",
                (status, now, document_id),
            )
            return cur.rowcount > 0

    def delete_document(self, document_id: str) -> bool:
        """按 document_id 删除一条 registry 记录。"""
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM rag_document_registry WHERE document_id = ?",
                (document_id,),
            )
            return cur.rowcount > 0
