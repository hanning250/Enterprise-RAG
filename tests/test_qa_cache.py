"""QA 缓存：命中/未命中、身份隔离、TTL、清空。"""

from __future__ import annotations

import os
import time
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from langchain_core.documents import Document

import app.auth as auth
import main
from rag.qa_cache import QaCache, QaCacheEntry, get_qa_cache, reset_qa_cache_for_tests
from rag.query_service import QueryResult, RetrievalTrace


class QaCacheUnitTests(unittest.TestCase):
    def setUp(self):
        reset_qa_cache_for_tests()

    def tearDown(self):
        reset_qa_cache_for_tests()

    def test_make_key_isolates_by_user_id(self):
        a = QaCache.make_key("工资多少", user_id="EMP001", roles=["employee"])
        b = QaCache.make_key("工资多少", user_id="EMP002", roles=["employee"])
        self.assertNotEqual(a, b)

    def test_make_key_isolates_by_roles_and_scope(self):
        a = QaCache.make_key("政策", user_id="u1", roles=["hr"], data_scope="company")
        b = QaCache.make_key("政策", user_id="u1", roles=["employee"], data_scope="self")
        self.assertNotEqual(a, b)

    def test_hit_miss_and_lru(self):
        cache = QaCache(max_entries=2, ttl_seconds=600, enabled=True)
        e1 = QaCacheEntry("a1", [], {}, time.monotonic())
        e2 = QaCacheEntry("a2", [], {}, time.monotonic())
        e3 = QaCacheEntry("a3", [], {}, time.monotonic())
        cache.put("k1", e1)
        cache.put("k2", e2)
        self.assertEqual(cache.get("k1").answer, "a1")
        cache.put("k3", e3)  # 应挤掉最久未用的 k2
        self.assertIsNone(cache.get("k2"))
        self.assertEqual(cache.get("k1").answer, "a1")
        self.assertEqual(cache.get("k3").answer, "a3")

    def test_ttl_expiry(self):
        cache = QaCache(max_entries=8, ttl_seconds=1, enabled=True)
        cache.put("k", QaCacheEntry("old", [], {}, time.monotonic() - 2))
        self.assertIsNone(cache.get("k"))

    def test_clear(self):
        cache = QaCache(max_entries=8, ttl_seconds=600, enabled=True)
        cache.put("k", QaCacheEntry("x", [], {}, time.monotonic()))
        self.assertEqual(cache.clear(), 1)
        self.assertIsNone(cache.get("k"))


class QaCacheApiTests(unittest.TestCase):
    def setUp(self):
        self._env_backup = dict(os.environ)
        auth._identity_config_singleton = None
        main._identity_cfg = None
        reset_qa_cache_for_tests()
        os.environ["AUTH_TRUSTED_IDENTITY_SECRET"] = "test-secret"

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env_backup)
        auth._identity_config_singleton = None
        main._identity_cfg = None
        reset_qa_cache_for_tests()

    def _identity(self, user_id: str = "EMP001", roles=None):
        return {
            "user_id": user_id,
            "user_name": "测试",
            "department": "技术部",
            "position": "工程师",
            "roles": roles or ["employee"],
            "data_scope": "self",
        }

    def _headers(self):
        return {"X-Internal-Auth": "test-secret"}

    def test_v2_query_cache_hit_skips_retrieve_and_llm(self):
        auth._identity_config_singleton = None
        main._identity_cfg = None

        doc = Document(
            page_content="年假 5 天",
            metadata={"doc_type": "policy", "sensitivity_level": "internal"},
        )
        mock_result = QueryResult(
            query="年假怎么规定？",
            answer="",
            evidence_docs=[doc],
            trace=RetrievalTrace(),
        )
        mock_qsvc = MagicMock()
        mock_qsvc.retrieve_only.return_value = mock_result
        mock_qsvc.build_context_from_docs.return_value = ("ctx", 10, 1)

        mock_rag = MagicMock()
        mock_rag.summarize.return_value = "根据资料，年假为 5 天。【1】"

        with patch.object(main, "get_query_svc", return_value=mock_qsvc), patch.object(
            main, "get_rag_service", return_value=mock_rag
        ), patch.object(
            main,
            "_summarize_with_answer_policy",
            return_value="根据资料，年假为 5 天。【1】",
        ):
            client = TestClient(main.app)
            body = {
                "query": "年假怎么规定？",
                "summarize": True,
                "identity": self._identity("EMP001"),
            }
            r1 = client.post("/api/v2/rag/query", json=body, headers=self._headers())
            self.assertEqual(r1.status_code, 200, r1.text)
            j1 = r1.json()
            self.assertFalse(j1["trace"]["cache_hit"])
            self.assertIn("年假", j1["answer"])
            self.assertEqual(mock_qsvc.retrieve_only.call_count, 1)

            r2 = client.post("/api/v2/rag/query", json=body, headers=self._headers())
            self.assertEqual(r2.status_code, 200, r2.text)
            j2 = r2.json()
            self.assertTrue(j2["trace"]["cache_hit"])
            self.assertEqual(j2["answer"], j1["answer"])
            # 第二次不应再检索 / 总结
            self.assertEqual(mock_qsvc.retrieve_only.call_count, 1)

    def test_different_user_does_not_share_cache(self):
        auth._identity_config_singleton = None
        main._identity_cfg = None

        doc = Document(
            page_content="公开制度",
            metadata={"doc_type": "policy", "sensitivity_level": "internal"},
        )
        mock_result = QueryResult(
            query="制度是什么",
            answer="",
            evidence_docs=[doc],
            trace=RetrievalTrace(),
        )
        mock_qsvc = MagicMock()
        mock_qsvc.retrieve_only.return_value = mock_result
        mock_qsvc.build_context_from_docs.return_value = ("ctx", 10, 1)

        def _summarize(*, query, scope, filtered_docs, acl_blocked, qsvc, rag_svc):
            return f"答案-{scope.user_id}"

        with patch.object(main, "get_query_svc", return_value=mock_qsvc), patch.object(
            main, "get_rag_service", return_value=MagicMock()
        ), patch.object(
            main, "_summarize_with_answer_policy", side_effect=_summarize
        ):
            client = TestClient(main.app)

            r1 = client.post(
                "/api/v2/rag/query",
                json={
                    "query": "制度是什么",
                    "summarize": True,
                    "identity": self._identity("EMP001"),
                },
                headers=self._headers(),
            )
            self.assertEqual(r1.status_code, 200)
            self.assertEqual(r1.json()["answer"], "答案-EMP001")

            r2 = client.post(
                "/api/v2/rag/query",
                json={
                    "query": "制度是什么",
                    "summarize": True,
                    "identity": self._identity("EMP002"),
                },
                headers=self._headers(),
            )
            self.assertEqual(r2.status_code, 200)
            self.assertFalse(r2.json()["trace"]["cache_hit"])
            self.assertEqual(r2.json()["answer"], "答案-EMP002")
            self.assertEqual(mock_qsvc.retrieve_only.call_count, 2)

    def test_build_clears_qa_cache(self):
        cache = get_qa_cache()
        cache.put("k2", QaCacheEntry("x", [], {}, time.monotonic()))
        self.assertIsNotNone(cache.get("k2"))
        n = get_qa_cache().clear()
        self.assertEqual(n, 1)
        self.assertIsNone(get_qa_cache().get("k2"))


if __name__ == "__main__":
    unittest.main()
