import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.documents import Document

import app.auth as auth
import main
from eval.run_retrieval_eval import load_eval_dataset
from ingestion.loaders.xlsx_loader import xlsx_loader
from rag.access_policy import AccessScope, is_doc_accessible
from rag.answer_policy import (
    DISCLAIMER_HR_CONSULT,
    REFUSAL_INSUFFICIENT_DATA,
    REFUSAL_NO_PERMISSION,
    refusal_or_none,
)
from rag.bm25_retriever import _jieba_tokenize
from rag.context_expander import ContextExpander
from rag.hybrid_retriever import HybridRetriever
from rag.document_normalizer import DataCleaningConfig, normalize_chunks
from rag.query_service import QueryResult


class EnterpriseRegressionTests(unittest.TestCase):
    def setUp(self):
        self._env_backup = dict(os.environ)
        auth._identity_config_singleton = None
        main._identity_cfg = None

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env_backup)
        auth._identity_config_singleton = None
        main._identity_cfg = None

    def test_health_check_does_not_require_identity_headers(self):
        os.environ["AUTH_TRUSTED_IDENTITY_SECRET"] = "test-secret"
        auth._identity_config_singleton = None
        main._identity_cfg = None

        client = TestClient(main.app)
        response = client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_identity_headers_require_trusted_gateway_secret(self):
        os.environ["AUTH_TRUSTED_IDENTITY_SECRET"] = "test-secret"
        auth._identity_config_singleton = None

        missing_secret = auth.parse_identity_from_headers({"X-User-Id": "u1", "X-Roles": "admin"})
        valid_secret = auth.parse_identity_from_headers(
            {
                "X-Internal-Auth": "test-secret",
                "X-User-Id": "u1",
                "X-Roles": "admin",
            }
        )

        self.assertIsNone(missing_secret)
        self.assertIsNotNone(valid_secret)
        self.assertEqual(valid_secret.roles, ["admin"])

    def test_query_filters_acl_before_summarizing(self):
        os.environ["AUTH_TRUSTED_IDENTITY_SECRET"] = "test-secret"
        auth._identity_config_singleton = None
        main._identity_cfg = None

        allowed = Document(
            page_content="allowed internal policy",
            metadata={"doc_type": "policy", "sensitivity_level": "internal"},
        )
        blocked = Document(
            page_content="blocked salary secret",
            metadata={"doc_type": "salary", "sensitivity_level": "confidential", "employee_name": "Other"},
        )

        class FakeQueryService:
            def __init__(self):
                self.summarized_docs = None

            def query_full(self, *args, **kwargs):
                raise AssertionError("query_full must not summarize before ACL filtering")

            def retrieve_only(self, query, *, top_k_override=None, access_scope=None):
                return QueryResult(query=query, evidence_docs=[allowed, blocked])

            def summarize_with_context(self, query, context_docs, **kwargs):
                self.summarized_docs = list(context_docs)
                return "authorized answer"

        fake_qsvc = FakeQueryService()

        class FakeRagService:
            def get_chain(self):
                return object()

        client = TestClient(main.app)
        with patch.object(main, "get_query_svc", return_value=fake_qsvc), patch.object(
            main, "get_rag_service", return_value=FakeRagService()
        ):
            response = client.post(
                "/api/rag/query",
                headers={
                    "X-Internal-Auth": "test-secret",
                    "X-User-Id": "u1",
                    "X-Roles": "employee",
                    "X-Data-Scope": "self",
                },
                json={"query": "工资", "summarize": True},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("authorized answer", response.json()["answer"])
        self.assertIn(DISCLAIMER_HR_CONSULT, response.json()["answer"])
        self.assertEqual(len(response.json()["evidence_docs"]), 1)
        self.assertEqual(fake_qsvc.summarized_docs, [allowed])

    def test_exact_dedup_keeps_same_text_at_different_chunk_levels(self):
        text = "张三 2026年3月 实发工资 22900 元，部门为技术研发部。"
        section = Document(page_content=text, metadata={"chunk_level": "section"})
        fine = Document(page_content=text, metadata={"chunk_level": "fine"})
        cfg = DataCleaningConfig(l3_simhash_dedup=False)

        normalized, _report = normalize_chunks([section, fine], config=cfg)

        levels = sorted(doc.metadata.get("chunk_level") for doc in normalized)
        self.assertEqual(levels, ["fine", "section"])

    def test_retrieval_eval_requires_expected_sources_for_non_refusal_samples(self):
        with TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "bad_eval.jsonl"
            dataset.write_text(
                json.dumps(
                    {
                        "id": "bad-001",
                        "question": "普通问题",
                        "expected_refusal": False,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_eval_dataset(str(dataset))

    def test_xlsx_loader_detects_header_row_and_employee_metadata(self):
        salary_file = next(Path("data").glob("*2026*3*.xlsx"))

        docs = xlsx_loader(str(salary_file))

        with_name = [doc for doc in docs if doc.metadata.get("employee_name")]
        self.assertGreater(len(with_name), 0)
        first_employee = with_name[0]
        self.assertEqual(first_employee.metadata.get("employee_name"), "戴晨")
        self.assertEqual(first_employee.metadata.get("data_month"), 3)
        self.assertIn("实发工资", first_employee.page_content)

    def test_bm25_tokenizer_keeps_chinese_names_and_months(self):
        tokens = _jieba_tokenize("戴晨 2026年3月 实发工资")

        self.assertIn("戴晨", tokens)
        self.assertIn("3月", tokens)
        self.assertIn("实发工资", tokens)

    def test_structured_salary_recall_hits_exact_employee_month(self):
        retriever = HybridRetriever()

        candidates = retriever._structured_salary_retrieve("戴晨 2026年3月 实发工资")

        self.assertGreater(len(candidates), 0)
        top = candidates[0].doc.metadata
        self.assertEqual(top.get("employee_name"), "戴晨")
        self.assertEqual(top.get("data_month"), 3)

    def test_context_expander_keeps_salary_fine_chunk_identity(self):
        """工资表 parent_chunk_id 常误指向文件首条 section；expand 不得换人。"""
        wrong_parent = Document(
            page_content="【2026年3月工资 - 胡沐阳】\n个税:3650.25 元",
            metadata={
                "chunk_id": "parent_00000",
                "chunk_level": "section",
                "doc_type": "salary",
                "employee_name": "胡沐阳",
                "data_month": 3,
            },
        )
        fine = Document(
            page_content="【2026年3月工资 - 王刚】\n个税:3834.25 元",
            metadata={
                "chunk_id": "fine_wanggang",
                "chunk_level": "fine",
                "doc_type": "salary",
                "employee_name": "王刚",
                "data_month": 3,
                "parent_chunk_id": "parent_00000",
            },
        )

        class FakeStore:
            def get(self, ids=None, include=None):
                return {
                    "documents": [wrong_parent.page_content],
                    "metadatas": [wrong_parent.metadata],
                }

        expanded = ContextExpander(FakeStore()).expand([fine], query="王刚2026年3月 个税扣多少？")

        self.assertEqual(len(expanded), 1)
        self.assertEqual(expanded[0].metadata.get("employee_name"), "王刚")
        self.assertIn("3834.25", expanded[0].page_content)

    def test_expand_keeps_fine_when_parent_content_mismatch(self):
        """parent 正文不含 fine 时不得替换（长节绑错 ::0 的兜底）。"""
        wrong_parent = Document(
            page_content="财务报表属于公司一级涉密资料，仅总经理可查阅。",
            metadata={"chunk_id": "parent_head", "chunk_level": "section", "doc_type": "policy"},
        )
        fine = Document(
            page_content="3.5.3 报表审核与提交流程\n1、报表编制：财务会计完成全套报表编制。\n2、初审校验：财务负责人签字确认。",
            metadata={
                "chunk_id": "fine_353",
                "chunk_level": "fine",
                "doc_type": "policy",
                "parent_chunk_id": "parent_head",
            },
        )

        class FakeStore:
            def get(self, ids=None, include=None):
                return {
                    "documents": [wrong_parent.page_content],
                    "metadatas": [wrong_parent.metadata],
                }

        expanded = ContextExpander(FakeStore()).expand([fine], query="报表审核与提交流程")
        self.assertEqual(len(expanded), 1)
        self.assertIn("3.5.3", expanded[0].page_content)
        self.assertNotIn("涉密资料", expanded[0].page_content)

    def test_fine_binds_to_matching_section_part_not_always_zero(self):
        """长节多分段时，含后段内容的 fine 必须绑到对应 ::part，而非 ::0。"""
        from rag.document_normalizer import ChunkBindingContext, bind_chunk_metadata
        from rag.semantic_chunker import SemanticChunkConfig, semantic_split_documents
        from rag.vector_store import build_chunk_id

        # 故意造一段远超 section 预算的正文，迫使多分段；后半含独特锚点
        head = "开头保密条款。" * 40
        tail = "3.5.3 报表审核与提交流程。初审校验由财务负责人签字确认。" * 8
        doc = Document(
            page_content=head + "\n\n" + tail,
            metadata={"source": "data/假制度.docx", "doc_type": "policy"},
        )
        chunks = semantic_split_documents(
            [doc],
            config=SemanticChunkConfig(
                max_fine_chars=120,
                max_section_chars=200,
                allow_section_partition=True,
            ),
        )
        section_parts = [
            c for c in chunks if c.metadata.get("chunk_level") == "section"
        ]
        self.assertGreater(len(section_parts), 1, msg="测试数据应触发多个 section_part")

        bound, _ids = bind_chunk_metadata(
            chunks,
            context=ChunkBindingContext(
                document_id="doc_part_test",
                document_version="v1",
                source_path="data/假制度.docx",
                content_hash="xyz",
                parser_version="2.4",
                chunking_version="test",
                embedding_model="text-embedding-v3",
                embedding_version="1",
            ),
            build_chunk_id=build_chunk_id,
        )
        sections = {
            c.metadata["chunk_id"]: c
            for c in bound
            if c.metadata.get("chunk_level") == "section"
        }
        fines = [
            c
            for c in bound
            if c.metadata.get("chunk_level") == "fine" and "初审校验" in (c.page_content or "")
        ]
        self.assertTrue(fines, msg="应存在含初审校验的 fine")
        for fine in fines:
            parent_id = fine.metadata.get("parent_chunk_id")
            self.assertTrue(parent_id)
            parent = sections[parent_id]
            self.assertIn(
                "初审校验",
                parent.page_content,
                msg="fine 不应绑到不含初审校验的 section_part（常见为 ::0）",
            )
            self.assertNotEqual(int(fine.metadata.get("section_part_index") or -1), -1)

    def test_enterprise_retrieve_only_runs_expand_then_compress(self):
        """API/评测统一编排：retrieve_only 必须经过 expand → compress。"""
        from rag.query_service import EnterpriseQueryService

        fine = Document(
            page_content="fine only",
            metadata={"chunk_id": "f1", "doc_type": "policy", "_retrieval_source": "vector"},
        )
        expanded = Document(
            page_content="expanded section",
            metadata={"chunk_id": "p1", "doc_type": "policy"},
        )
        compressed = Document(
            page_content="compressed evidence",
            metadata={"chunk_id": "c1", "doc_type": "policy"},
        )

        class FakeHybrid:
            mode = "hybrid"
            score_guard = type("SG", (), {"final_top_k": 4})()

            def retrieve_with_intent(self, query, access_scope=None):
                return [fine], None

        class FakeExpander:
            def __init__(self):
                self.calls = 0

            def expand(self, docs, query=""):
                self.calls += 1
                self.last_docs = list(docs)
                return [expanded]

        class FakeCompressor:
            def __init__(self):
                self.calls = 0

            def compress(self, query, docs, intent_result=None):
                self.calls += 1
                self.last_docs = list(docs)
                return [compressed]

        expander = FakeExpander()
        compressor = FakeCompressor()
        svc = EnterpriseQueryService(
            FakeHybrid(),
            context_expander=expander,
            context_compressor=compressor,
        )
        result = svc.retrieve_only("制度问题")

        self.assertEqual(expander.calls, 1)
        self.assertEqual(compressor.calls, 1)
        self.assertEqual(expander.last_docs, [fine])
        self.assertEqual(compressor.last_docs, [expanded])
        self.assertEqual(result.evidence_docs, [compressed])

    def test_retrieve_only_drops_other_salary_months(self):
        from rag.query_service import EnterpriseQueryService

        mar = Document(
            page_content="王刚 3月个税",
            metadata={
                "chunk_id": "m3",
                "doc_type": "salary",
                "employee_name": "王刚",
                "data_month": 3,
            },
        )
        jan = Document(
            page_content="王刚 1月个税",
            metadata={
                "chunk_id": "m1",
                "doc_type": "salary",
                "employee_name": "王刚",
                "data_month": 1,
            },
        )

        class FakeHybrid:
            mode = "hybrid"
            score_guard = type("SG", (), {"final_top_k": 4})()

            def retrieve_with_intent(self, query, access_scope=None):
                return [mar, jan], None

        class Passthrough:
            def expand(self, docs, query=""):
                return list(docs)

            def compress(self, query, docs, intent_result=None):
                return list(docs)

        svc = EnterpriseQueryService(
            FakeHybrid(),
            context_expander=Passthrough(),
            context_compressor=Passthrough(),
        )
        result = svc.retrieve_only("我2026年3月个税扣多少")
        months = [d.metadata.get("data_month") for d in result.evidence_docs]
        self.assertEqual(months, [3])

    def test_salary_rows_bind_unique_parent_chunk_ids(self):
        """同文件多人各自 section-00000 时，bind 后 fine 的 parent 必须指向本人 section。"""
        from rag.document_normalizer import ChunkBindingContext, bind_chunk_metadata
        from rag.semantic_chunker import semantic_split_documents
        from rag.vector_store import build_chunk_id

        rows = [
            Document(
                page_content="【2026年3月工资 - 王刚】\n姓名：王刚\n个税：3834.25 元",
                metadata={
                    "doc_type": "salary",
                    "employee_name": "王刚",
                    "employee_id": "EMP009",
                    "data_month": 3,
                },
            ),
            Document(
                page_content="【2026年3月工资 - 胡沐阳】\n姓名：胡沐阳\n个税：3650.25 元",
                metadata={
                    "doc_type": "salary",
                    "employee_name": "胡沐阳",
                    "employee_id": "EMP040",
                    "data_month": 3,
                },
            ),
        ]
        chunks = semantic_split_documents(rows)
        bound, _ids = bind_chunk_metadata(
            chunks,
            context=ChunkBindingContext(
                document_id="doc_salary_test",
                document_version="v1",
                source_path="data/员工工资表_2026年3月.xlsx",
                content_hash="abc",
                parser_version="2.4",
                chunking_version="test",
                embedding_model="text-embedding-v3",
                embedding_version="1",
            ),
            build_chunk_id=build_chunk_id,
        )

        sections = {
            c.metadata["chunk_id"]: c
            for c in bound
            if c.metadata.get("chunk_level") == "section"
        }
        fines = [c for c in bound if c.metadata.get("chunk_level") == "fine"]
        self.assertGreaterEqual(len(fines), 2)

        parent_ids = set()
        for fine in fines:
            emp = fine.metadata.get("employee_name")
            parent_id = fine.metadata.get("parent_chunk_id")
            self.assertTrue(parent_id, msg=f"{emp} missing parent_chunk_id")
            parent = sections[parent_id]
            self.assertEqual(
                parent.metadata.get("employee_name"),
                emp,
                msg=f"{emp} fine 指向了 {parent.metadata.get('employee_name')} 的 section",
            )
            parent_ids.add(parent_id)
        self.assertEqual(len(parent_ids), len({f.metadata.get("employee_name") for f in fines}))

    def test_employee_can_access_own_confidential_salary(self):
        scope = AccessScope(
            user_id="u1",
            user_name="张三",
            department="研发部",
            roles=["employee"],
            data_scope="self",
        )
        own = Document(
            page_content="张三 实发工资 10000",
            metadata={
                "doc_type": "salary",
                "sensitivity_level": "confidential",
                "employee_name": "张三",
                "department": "研发部",
            },
        )
        other = Document(
            page_content="李四 实发工资 20000",
            metadata={
                "doc_type": "salary",
                "sensitivity_level": "confidential",
                "employee_name": "李四",
                "department": "销售部",
            },
        )

        self.assertTrue(is_doc_accessible(scope, own)[0])
        self.assertFalse(is_doc_accessible(scope, other)[0])

        # 本人薪资即使部门字段与 access_scope 不一致也应放行
        own_other_dept = Document(
            page_content="张三 实发工资 10000",
            metadata={
                "doc_type": "salary",
                "sensitivity_level": "confidential",
                "employee_name": "张三",
                "department": "技术部",
            },
        )
        self.assertTrue(is_doc_accessible(scope, own_other_dept)[0])

    def test_manager_role_is_not_company_wide_salary_access(self):
        scope = AccessScope(
            user_id="u006",
            user_name="周八",
            department="研发部",
            roles=["manager"],
            data_scope="department",
        )
        other_dept = Document(
            page_content="李四 3月工资",
            metadata={
                "doc_type": "salary",
                "sensitivity_level": "confidential",
                "employee_name": "李四",
                "department": "销售部",
            },
        )
        own = Document(
            page_content="周八 3月工资",
            metadata={
                "doc_type": "salary",
                "sensitivity_level": "confidential",
                "employee_name": "周八",
                "department": "研发部",
            },
        )

        self.assertFalse(scope.can_see_company_wide())
        self.assertFalse(is_doc_accessible(scope, other_dept)[0])
        self.assertTrue(is_doc_accessible(scope, own)[0])

    def test_hr_admin_can_access_others_salary(self):
        scope = AccessScope(
            user_id="hr1",
            user_name="HR",
            department="人力资源部",
            roles=["hr_admin"],
            data_scope="company",
        )
        other = Document(
            page_content="李四 实发工资",
            metadata={
                "doc_type": "salary",
                "sensitivity_level": "confidential",
                "employee_name": "李四",
                "department": "销售部",
            },
        )
        self.assertTrue(scope.can_see_company_wide())
        self.assertTrue(is_doc_accessible(scope, other)[0])

    def test_refusal_when_all_evidence_acl_blocked(self):
        os.environ["AUTH_TRUSTED_IDENTITY_SECRET"] = "test-secret"
        auth._identity_config_singleton = None
        main._identity_cfg = None

        blocked = Document(
            page_content="blocked salary secret",
            metadata={
                "doc_type": "salary",
                "sensitivity_level": "confidential",
                "employee_name": "Other",
            },
        )

        class FakeQueryService:
            def retrieve_only(self, query, *, top_k_override=None, access_scope=None):
                return QueryResult(query=query, evidence_docs=[blocked])

            def summarize_with_context(self, *args, **kwargs):
                raise AssertionError("must refuse before summarizing empty ACL result")

        class FakeRagService:
            def get_chain(self):
                return object()

        client = TestClient(main.app)
        with patch.object(main, "get_query_svc", return_value=FakeQueryService()), patch.object(
            main, "get_rag_service", return_value=FakeRagService()
        ):
            response = client.post(
                "/api/rag/query",
                headers={
                    "X-Internal-Auth": "test-secret",
                    "X-User-Id": "u1",
                    "X-User-Name": "ZhangSan",
                    "X-Roles": "employee",
                    "X-Data-Scope": "self",
                },
                json={"query": "Other salary amount", "summarize": True},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["answer"], REFUSAL_NO_PERMISSION)
        self.assertEqual(response.json()["evidence_docs"], [])

    def test_refusal_or_none_uses_acl_blocked_count(self):
        scope = AccessScope(
            user_id="u1",
            user_name="张三",
            roles=["employee"],
            data_scope="self",
        )
        self.assertEqual(
            refusal_or_none(scope, [], acl_blocked_count=3),
            REFUSAL_NO_PERMISSION,
        )
        self.assertEqual(
            refusal_or_none(scope, [], acl_blocked_count=0),
            REFUSAL_INSUFFICIENT_DATA,
        )

    def test_post_process_skips_disclaimer_on_refusal(self):
        from rag.answer_policy import DISCLAIMER_HR_CONSULT, post_process_answer

        docs = [
            Document(
                page_content="policy",
                metadata={"doc_type": "policy"},
            )
        ]
        out = post_process_answer(
            REFUSAL_INSUFFICIENT_DATA,
            docs=docs,
        )
        self.assertEqual(out, REFUSAL_INSUFFICIENT_DATA)
        self.assertNotIn(DISCLAIMER_HR_CONSULT, out)

    def test_purge_missing_documents_marks_deleted(self):
        import shutil
        import tempfile

        from rag.document_registry import DocumentRegistryStore, canonicalize_source_path
        from rag.vector_store import VectorStoreService

        tmp = tempfile.mkdtemp()
        try:
            root = Path(tmp)
            keep = root / "keep.txt"
            keep.write_text("alive", encoding="utf-8")
            keep_path = canonicalize_source_path(str(keep))
            gone_path = canonicalize_source_path(str(root / "gone.txt"))

            registry = DocumentRegistryStore(str(root / "reg.sqlite3"))
            common = dict(
                content_hash="h",
                parser_version="1",
                chunking_version="1",
                embedding_model="m",
                embedding_version="1",
                chunk_count=1,
                status="active",
            )
            registry.upsert_document(document_id="doc_gone", source_path=gone_path, **common)
            registry.upsert_document(document_id="doc_keep", source_path=keep_path, **common)

            class StubStore:
                def __init__(self):
                    self.registry = registry

                def _delete_chunks_by_document_id(self, document_id: str) -> int:
                    return 2

            stub = StubStore()
            purged, chunks = VectorStoreService.purge_missing_documents(
                stub, alive_source_paths={keep_path}
            )

            self.assertEqual(purged, 1)
            self.assertEqual(chunks, 2)
            self.assertEqual(registry.find_by_document_id("doc_gone")["status"], "deleted")
            self.assertEqual(registry.find_by_document_id("doc_keep")["status"], "active")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
