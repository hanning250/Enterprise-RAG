"""API Rerank 客户端模块。"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Any, Optional
import httpx
from rag.types import RetrievalCandidate
from utils.config_handler import rag_conf
from utils.logger_handler import logger

DASHSCOPE_RERANK_ENDPOINT_ENV = "DASHSCOPE_RERANK_ENDPOINT"
COMPATIBLE_RERANK_ENDPOINT_ENV = "COMPATIBLE_RERANK_ENDPOINT"


def _env_value(name: str) -> str:
    return str(os.environ.get(name, "") or "").strip()


def _resolve_rerank_config() -> dict:
    """从 rag_conf 和环境变量中解析 Rerank 配置。"""
    rerank_cfg = rag_conf.get("rerank", {}) or {}
    retrieval_cfg = rag_conf.get("retrieval", {}) or {}

    enabled: bool = bool(rerank_cfg.get("enabled", True))
    provider: str = str(rerank_cfg.get("provider", "api")).lower()

    if provider == "dashscope":
        default_ep_for_provider = _env_value(DASHSCOPE_RERANK_ENDPOINT_ENV)
    else:
        default_ep_for_provider = _env_value(COMPATIBLE_RERANK_ENDPOINT_ENV)

    endpoint_raw = rerank_cfg.get("endpoint") or ""
    endpoint_resolved = _resolve_env_placeholder(str(endpoint_raw)).strip()
    if endpoint_resolved:
        endpoint: str = endpoint_resolved
    else:
        endpoint = default_ep_for_provider

    api_key_env_name: str = str(rerank_cfg.get("api_key_env", "DASHSCOPE_API_KEY"))
    api_key: str = str(os.environ.get(api_key_env_name, "") or "")

    model: str = str(rerank_cfg.get("model", "gte-rerank-v2"))
    top_k: int = int(
        rerank_cfg.get("top_k")
        or retrieval_cfg.get("final_top_k")
        or 6
    )
    timeout_seconds: int = int(rerank_cfg.get("timeout_seconds", 20))
    fail_open: bool = bool(rerank_cfg.get("fail_open", True))

    return {
        "enabled": enabled,
        "provider": provider,
        "endpoint": endpoint,
        "api_key": api_key,
        "model": model,
        "top_k": top_k,
        "timeout_seconds": timeout_seconds,
        "fail_open": fail_open,
    }


def _resolve_env_placeholder(value: str) -> str:
    """把形如 ${RERANK_API_BASE_URL} 的占位符替换为真实环境变量值。"""
    if not value:
        return ""
    import re
    pattern = re.compile(r"\$\{([A-Z0-9_]+)\}")

    def _replace(match: re.Match) -> str:
        env_name = match.group(1)
        return os.environ.get(env_name, "")

    return pattern.sub(_replace, value)


class RerankError(RuntimeError):
    """Rerank 异常基类。"""


class RerankConfigError(RerankError):
    """配置缺失导致的异常。"""


class RerankApiError(RerankError):
    """调用 Rerank API 过程中发生的异常。"""


class BaseRerankerClient(ABC):
    """Rerank 客户端抽象基类。"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:
        """对候选文档执行重排。"""
        raise NotImplementedError


class HttpRerankerClient(BaseRerankerClient):
    """通用 HTTP Rerank 客户端，支持 DashScope 和 OpenAI-compatible 格式。"""

    def __init__(self, config: Optional[dict] = None):
        """初始化 HTTP Rerank 客户端。"""
        self.cfg: dict = dict(config) if config is not None else _resolve_rerank_config()

        self.enabled: bool = bool(self.cfg.get("enabled", True))
        self.fail_open: bool = bool(self.cfg.get("fail_open", True))
        self.timeout_seconds: int = int(self.cfg.get("timeout_seconds", 20))
        self.top_k: int = int(self.cfg.get("top_k", 6))

        if not self.enabled:
            logger.info("[Rerank] 配置中 rerank.enabled=false，Rerank 阶段将被跳过")
            return

        self.endpoint: str = str(self.cfg.get("endpoint") or "").strip()
        self.api_key: str = str(self.cfg.get("api_key") or "").strip()
        self.model: str = str(self.cfg.get("model", "gte-rerank-v2"))
        self.provider: str = str(self.cfg.get("provider", "api"))

        missing: list[str] = []
        if not self.endpoint:
            missing.append("endpoint")
        if not self.api_key:
            missing.append("api_key")

        if missing:
            if self.fail_open:
                logger.warning(
                    "[Rerank] 配置缺失："
                    + "、".join(missing)
                    + "，fail_open=true → Rerank 将被跳过，回退到 RRF 融合顺序。"
                )
                self.enabled = False
            else:
                raise RerankConfigError(
                    "[Rerank] 配置缺失：" + "、".join(missing)
                    + "，且 fail_open=false，请检查 config/rag.yml 或环境变量。"
                )

        self._client_kwargs: dict[str, Any] = {
            "timeout": httpx.Timeout(self.timeout_seconds, connect=10.0),
        }

    @staticmethod
    def _is_trivial(candidates: list[RetrievalCandidate]) -> bool:
        """候选数量 ≤ 1 时没必要跑 Rerank。"""
        return len(candidates) <= 1

    def _build_request_body(
        self, query: str, candidates: list[RetrievalCandidate]
    ) -> dict[str, Any]:
        """根据供应商构造请求 Body。"""
        documents: list[str] = [c.doc.page_content or "" for c in candidates]
        top_n = min(self.top_k, len(candidates))

        if self.provider == "dashscope":
            return {
                "model": self.model,
                "input": {
                    "query": query,
                    "documents": documents,
                },
                "parameters": {
                    "top_n": top_n,
                    "return_documents": False,
                },
            }

        return {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": False,
        }

    def _build_request_headers(self) -> dict[str, str]:
        """构造 HTTP 请求头。"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    @staticmethod
    def _parse_response(
        response_json: dict[str, Any],
        candidates: list[RetrievalCandidate],
    ) -> list[tuple[int, float]]:
        """把 Rerank API 响应解析成 (原候选索引, rerank_score) 元组列表。"""
        results_container: Optional[list[dict]] = None

        output = response_json.get("output")
        if isinstance(output, dict) and isinstance(output.get("results"), list):
            results_container = output["results"]
        elif isinstance(response_json.get("results"), list):
            results_container = response_json["results"]
        elif isinstance(response_json.get("data"), list):
            results_container = response_json["data"]

        if results_container is None:
            raise RerankApiError(
                f"[Rerank] API 响应格式无法识别，缺少 output.results/results/data 字段。响应前 800 字={str(response_json)[:800]}"
            )

        parsed: list[tuple[int, float]] = []
        for item in results_container:
            if not isinstance(item, dict):
                continue
            idx_raw = item.get("index")
            if idx_raw is None:
                continue
            try:
                idx = int(idx_raw)
            except (TypeError, ValueError):
                continue
            if idx < 0 or idx >= len(candidates):
                continue

            score_raw = (
                item.get("relevance_score")
                or item.get("score")
                or item.get("relevance")
            )
            try:
                score = float(score_raw) if score_raw is not None else 0.0
            except (TypeError, ValueError):
                score = 0.0

            parsed.append((idx, score))

        if not parsed:
            raise RerankApiError("[Rerank] API 返回了 0 条有效结果，视为异常。")

        return parsed

    def _call_api(
        self, query: str, candidates: list[RetrievalCandidate]
    ) -> list[RetrievalCandidate]:
        """发起 HTTP 请求、解析响应、回填 rerank_score 并排序。"""
        body = self._build_request_body(query, candidates)
        headers = self._build_request_headers()

        t0 = time.perf_counter()
        try:
            request_url = self.endpoint.rstrip("/")
            if self.provider != "dashscope":
                if not request_url.endswith("/rerank"):
                    request_url = request_url + "/rerank"

            with httpx.Client(**self._client_kwargs) as client:
                resp = client.post(request_url, headers=headers, json=body)
                if resp.status_code != 200:
                    raise RerankApiError(
                        f"[Rerank] HTTP {resp.status_code} != 200，body={resp.text[:500]}"
                    )
                try:
                    resp_json = resp.json()
                except ValueError as exc:
                    raise RerankApiError(
                        f"[Rerank] 响应不是合法 JSON：{resp.text[:300]}"
                    ) from exc
        except httpx.HTTPError as exc:
            raise RerankApiError(f"[Rerank] HTTP 网络错误：{exc}") from exc

        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        parsed = self._parse_response(resp_json, candidates)

        for c in candidates:
            c.rerank_score = -1.0

        for idx, score in parsed:
            candidates[idx].rerank_score = score

        reranked = sorted(
            candidates,
            key=lambda c: c.rerank_score if c.rerank_score is not None else -1.0,
            reverse=True,
        )

        logger.info(
            f"[Rerank] HTTP 调用完成：耗时 {elapsed_ms}ms，"
            f"API 返回 {len(parsed)}/{len(candidates)} 条，"
            f"最高分={parsed[0][1]:.4f}（若 parsed 非空）"
            if parsed
            else f"[Rerank] HTTP 调用完成：耗时 {elapsed_ms}ms，但 parsed 为空"
        )
        return reranked

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:
        """对候选执行 Rerank（带 fail_open 兜底）。"""
        if not self.enabled or self._is_trivial(candidates):
            logger.debug(
                f"[Rerank] 跳过：enabled={self.enabled}，候选数={len(candidates)}（≤1 视为无需重排）"
            )
            return list(candidates)

        try:
            return self._call_api(query, candidates)
        except RerankError as exc:
            if self.fail_open:
                logger.warning(
                    f"[Rerank] 重排失败，fail_open=true → 回退到 RRF 顺序。原因：{exc}"
                )
                return list(candidates)
            else:
                logger.error(
                    f"[Rerank] 重排失败，fail_open=false → 抛出异常。原因：{exc}",
                    exc_info=True,
                )
                raise
        except Exception as exc:
            logger.error(
                f"[Rerank] 未预期异常：{exc}，"
                + ("fail_open=true → 回退 RRF 顺序" if self.fail_open else "fail_open=false → 抛出"),
                exc_info=True,
            )
            if self.fail_open:
                return list(candidates)
            raise RerankApiError(f"[Rerank] 未预期异常：{exc}") from exc
