"""入库幂等提交顺序：先写新 → 记账 → 再删旧。"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from langchain_core.documents import Document


class IngestCommitOrderTests(unittest.TestCase):
    def _make_service(self):
        from rag.vector_store import VectorStoreService

        svc = object.__new__(VectorStoreService)
        svc.vector_store = MagicMock()
        svc.registry = MagicMock()
        svc.parser_version = "p1"
        svc.chunking_version = "c1"
        svc.embedding_model_name = "e1"
        svc.embedding_version = "1.0"
        return svc

    def test_registry_failure_rolls_back_new_only(self):
        svc = self._make_service()
        calls: list[str] = []

        svc.vector_store.add_documents.side_effect = lambda docs, ids=None: calls.append(
            "add"
        )
        svc.registry.upsert_document.side_effect = RuntimeError("registry boom")
        svc._delete_chunks_except = MagicMock(  # type: ignore[method-assign]
            side_effect=lambda *a, **k: calls.append("delete_old")
        )
        svc._rollback_new_chunks = MagicMock(  # type: ignore[method-assign]
            side_effect=lambda ids: calls.append("rollback_new")
        )

        with self.assertRaises(RuntimeError):
            svc._commit_ingested_document(
                document_id="doc_x",
                source_path="/tmp/a.md",
                content_hash="h",
                split_document=[Document(page_content="n1")],
                chunk_ids=["doc_x_v_new_chunk_00000"],
            )

        self.assertEqual(calls, ["add", "rollback_new"])
        svc._delete_chunks_except.assert_not_called()

    def test_cleanup_failure_after_registry_keeps_new(self):
        svc = self._make_service()
        calls: list[str] = []

        svc.vector_store.add_documents.side_effect = lambda docs, ids=None: calls.append(
            "add"
        )
        svc.registry.upsert_document.side_effect = lambda **kwargs: calls.append(
            "upsert"
        )
        svc._delete_chunks_except = MagicMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("chroma delete boom")
        )
        svc._rollback_new_chunks = MagicMock(  # type: ignore[method-assign]
            side_effect=lambda ids: calls.append("rollback_new")
        )

        # 清旧失败只告警，不抛、不回滚新分片
        svc._commit_ingested_document(
            document_id="doc_x",
            source_path="/tmp/a.md",
            content_hash="h",
            split_document=[Document(page_content="n1")],
            chunk_ids=["doc_x_v_new_chunk_00000"],
        )

        self.assertEqual(calls, ["add", "upsert"])
        svc._delete_chunks_except.assert_called_once()
        svc._rollback_new_chunks.assert_not_called()

    def test_happy_path_order_add_upsert_delete(self):
        svc = self._make_service()
        calls: list[str] = []

        svc.vector_store.add_documents.side_effect = lambda docs, ids=None: calls.append(
            "add"
        )
        svc.registry.upsert_document.side_effect = lambda **kwargs: calls.append(
            "upsert"
        )
        svc._delete_chunks_except = MagicMock(  # type: ignore[method-assign]
            side_effect=lambda *a, **k: calls.append("delete_old") or 1
        )
        svc._rollback_new_chunks = MagicMock()  # type: ignore[method-assign]

        svc._commit_ingested_document(
            document_id="doc_x",
            source_path="/tmp/a.md",
            content_hash="h",
            split_document=[Document(page_content="n1")],
            chunk_ids=["doc_x_v_new_chunk_00000"],
        )

        self.assertEqual(calls, ["add", "upsert", "delete_old"])
        svc._rollback_new_chunks.assert_not_called()

    def test_prune_stale_version_chunks_keeps_current_prefix(self):
        svc = self._make_service()
        svc._chunk_ids_by_document_id = lambda document_id: {  # type: ignore[method-assign]
            "doc_a_v_old_chunk_00000",
            "doc_a_v_new_chunk_00000",
            "doc_a_v_new_chunk_00001",
            "unrelated",
        }
        deleted: list[set[str]] = []

        def delete_chunk_ids(ids):
            deleted.append(set(ids))
            return len(ids)

        svc._delete_chunk_ids = delete_chunk_ids  # type: ignore[method-assign]
        n = svc._prune_stale_version_chunks("doc_a", document_version="new")
        self.assertEqual(n, 1)
        self.assertEqual(deleted, [{"doc_a_v_old_chunk_00000"}])


if __name__ == "__main__":
    unittest.main()
