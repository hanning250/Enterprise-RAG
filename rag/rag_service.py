"""RAG 总结服务模块：LLM chain 工厂 + 对统一编排的薄封装。"""

from typing import Optional, Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from rag.hybrid_retriever import HybridRetriever
from rag.query_service import EnterpriseQueryService
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from model.factory import chat_model
from utils.logger_handler import logger


class RagSummarizeService:
    """LLM chain 与查询编排的适配层。

    检索一律委托 ``EnterpriseQueryService.retrieve_only``（含 expand/compress），
    本类主要负责构建/暴露 LCEL chain，供 API summarize 注入。
    """

    def __init__(
        self,
        *,
        query_service: Optional[EnterpriseQueryService] = None,
        hybrid_retriever: Optional[HybridRetriever] = None,
        vector_store: Optional[VectorStoreService] = None,
    ):
        if query_service is not None:
            self.query_service = query_service
            self.hybrid_retriever = query_service.hybrid
            self.vector_store = self.hybrid_retriever.vector_store
        else:
            self.vector_store = vector_store or VectorStoreService()
            self.hybrid_retriever = hybrid_retriever or HybridRetriever(
                vector_store_service=self.vector_store
            )
            self.query_service = EnterpriseQueryService(
                hybrid_retriever=self.hybrid_retriever
            )

        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()
        # 兼容旧属性名（评测/调试曾直接访问 expander）
        self.context_expander = self.query_service.context_expander
        self.context_compressor = self.query_service.context_compressor
        logger.info("[RagSummarizeService] 初始化完成（检索委托 EnterpriseQueryService）")

    def _init_chain(self):
        """初始化 LCEL 调用链。"""
        return self.prompt_template | self.model | StrOutputParser()

    def retriever_docs(self, query: str) -> list[Document]:
        """与线上 API 同款：Hybrid → Expand → Compress。"""
        return self.query_service.retrieve_only(query).evidence_docs

    def get_chain(self) -> Any:
        """暴露内部 LCEL chain 给 EnterpriseQueryService.summarize_with_context 注入。"""
        return self.chain
