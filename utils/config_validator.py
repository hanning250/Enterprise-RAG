from __future__ import annotations

import os
from dataclasses import dataclass, field

from utils.config_handler import rag_conf


@dataclass(slots=True)
class ConfigIssue:
    """单条配置问题。"""
    level: str
    category: str
    item: str
    message: str


@dataclass(slots=True)
class ConfigValidationResult:
    """配置校验的汇总结果。"""
    issues: list[ConfigIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """没有 error 级问题就算校验通过。"""
        return not any(issue.level == "error" for issue in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.level == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.level == "warning")

    def add_error(self, category: str, item: str, message: str) -> None:
        self.issues.append(ConfigIssue(level="error", category=category, item=item, message=message))

    def add_warning(self, category: str, item: str, message: str) -> None:
        self.issues.append(ConfigIssue(level="warning", category=category, item=item, message=message))

    def to_dict(self) -> dict[str, object]:
        """转成字典，方便通过 API 以 JSON 形式返回。"""
        return {
            "ok": self.ok,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [
                {
                    "level": issue.level,
                    "category": issue.category,
                    "item": issue.item,
                    "message": issue.message,
                }
                for issue in self.issues
            ],
        }

    def report(self) -> str:
        """生成人类可读的文本报告。"""
        lines: list[str] = []
        lines.append("===== 独立 RAG · 配置校验报告 =====")
        status_icon = "✅" if self.ok else "❌"
        status_text = "校验通过" if self.ok else "存在错误，请修复后再启动"
        lines.append(f"{status_icon} {status_text}（{self.error_count} 错误 / {self.warning_count} 警告）")

        if self.ok and self.warning_count == 0:
            lines.append("所有必需配置均已就绪，祝您使用愉快！")
            return "\n".join(lines)

        errors = [issue for issue in self.issues if issue.level == "error"]
        warnings = [issue for issue in self.issues if issue.level == "warning"]

        if errors:
            lines.append("---- 错误项（必须修复）----")
            for issue in errors:
                lines.append(f"  [{issue.category}] {issue.item}: {issue.message}")

        if warnings:
            lines.append("---- 警告项（建议关注）----")
            for issue in warnings:
                lines.append(f"  [{issue.category}] {issue.item}: {issue.message}")

        return "\n".join(lines)


def _looks_like_placeholder(value: str, placeholder: str) -> bool:
    """判断配置值是不是用户还没改的占位符。"""
    normalized = (value or "").strip()
    if not normalized:
        return True
    if normalized == placeholder:
        return True
    if normalized.isupper() and "YOUR" in normalized:
        return True
    return False


def _check_llm(result: ConfigValidationResult) -> None:
    """校验 LLM 聊天模型配置。"""
    category = "LLM"

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or _looks_like_placeholder(api_key, "YOUR_OPENAI_API_KEY_HERE"):
        result.add_error(
            category,
            "OPENAI_API_KEY",
            "环境变量未设置或仍为占位符。请在项目根目录复制 .env 为 .env，填入从 https://api.kkrich.ltd 获取的 API Key。",
        )

    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if not base_url or _looks_like_placeholder(base_url, "YOUR_OPENAI_BASE_URL_HERE"):
        result.add_error(
            category,
            "OPENAI_BASE_URL",
            "环境变量未设置或仍为占位符。请在 .env 中配置 OPENAI_BASE_URL。",
        )

    llm_cfg = rag_conf.get("llm") or {}
    model_name = llm_cfg.get("model_name") or rag_conf.get("chat_model_name")
    if not model_name or not str(model_name).strip():
        result.add_error(
            category,
            "llm.model_name",
            "config/rag.yml 中未配置 llm.model_name（或旧字段 chat_model_name）。请填入模型名，例如 gpt-5.5。",
        )


def _check_rag(result: ConfigValidationResult) -> None:
    """校验 RAG 相关配置（可降级）。"""
    category = "RAG"

    dashscope_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not dashscope_key or _looks_like_placeholder(dashscope_key, "YOUR_DASHSCOPE_API_KEY_HERE"):
        result.add_warning(
            category,
            "DASHSCOPE_API_KEY",
            "阿里云 DashScope API Key 未设置。RAG 知识库检索和 Rerank 功能不可用（纯对话仍可用）。请到 https://dashscope.console.aliyun.com/ 创建 API-KEY 并填入 .env。",
        )

    dashscope_base_url = os.getenv("DASHSCOPE_BASE_URL", "").strip()
    if not dashscope_base_url or _looks_like_placeholder(dashscope_base_url, "YOUR_DASHSCOPE_BASE_URL_HERE"):
        result.add_warning(
            category,
            "DASHSCOPE_BASE_URL",
            "阿里云 DashScope 兼容模式 Base URL 未设置。RAG 向量嵌入调用可能不可用。请在 .env 中配置 DASHSCOPE_BASE_URL。",
        )

    embed_model = rag_conf.get("embedding_model_name")
    if not embed_model or not str(embed_model).strip():
        result.add_warning(
            category,
            "embedding_model_name",
            "config/rag.yml 中 embedding_model_name 为空。即使后续配置好了 DashScope Key，RAG 检索也因缺嵌入模型名无法工作。",
        )

    retrieval_cfg = rag_conf.get("retrieval") or {}
    rerank_cfg = rag_conf.get("rerank") or {}

    rerank_enabled = bool(rerank_cfg.get("enabled", True))
    if not rerank_enabled:
        result.add_warning(
            category,
            "rerank.enabled",
            "当前未启用 Rerank（rerank.enabled=false），双路召回候选不会做相关性精排，检索质量可能下降。建议保持启用。",
        )
    else:
        rerank_model = rerank_cfg.get("model")
        if not rerank_model or not str(rerank_model).strip():
            result.add_warning(
                category,
                "rerank.model",
                "Rerank 已启用但 rerank.model 为空。Rerank 调用会在 fail_open=true 时自动回退到 RRF 顺序。建议填 gte-rerank-v2。",
            )

        api_key_env_name = str(rerank_cfg.get("api_key_env") or "DASHSCOPE_API_KEY")
        if not os.getenv(api_key_env_name, "").strip():
            result.add_warning(
                category,
                f"rerank.api_key_env({api_key_env_name})",
                f"Rerank 的 api_key_env 指向环境变量 {api_key_env_name}，但该变量未设置。Rerank 将在 fail_open=true 时回退到 RRF 顺序。",
            )

        provider = str(rerank_cfg.get("provider", "api")).lower()
        endpoint_env_name = "DASHSCOPE_RERANK_ENDPOINT" if provider == "dashscope" else "COMPATIBLE_RERANK_ENDPOINT"
        if not os.getenv(endpoint_env_name, "").strip():
            result.add_warning(
                category,
                endpoint_env_name,
                f"当前 rerank.provider={provider}，但环境变量 {endpoint_env_name} 未设置。Rerank 将在 fail_open=true 时回退到 RRF 顺序。",
            )

        rerank_top_k = rerank_cfg.get("top_k")
        fusion_top_k = retrieval_cfg.get("fusion_top_k")
        if isinstance(rerank_top_k, int) and isinstance(fusion_top_k, int) and rerank_top_k < fusion_top_k:
            result.add_warning(
                category,
                "rerank.top_k < retrieval.fusion_top_k",
                f"Rerank 送排数 {rerank_top_k} 小于融合候选数 {fusion_top_k}，部分候选会在 Rerank 前被截断，浪费召回预算。建议 rerank.top_k >= fusion_top_k。",
            )

    final_top_k = retrieval_cfg.get("final_top_k")
    rag_top_k = rag_conf.get("rag", {}).get("top_k")
    if isinstance(final_top_k, int) and isinstance(rag_top_k, int) and final_top_k != rag_top_k:
        result.add_warning(
            category,
            "final_top_k vs rag.top_k",
            f"retrieval.final_top_k({final_top_k}) 与 rag.top_k({rag_top_k}) 不一致，实际生效的是前者。建议两个值保持相同以避免困惑。",
        )


def _check_auth(result: ConfigValidationResult) -> None:
    """校验企业身份透传配置。"""
    category = "AUTH"
    require_auth = os.getenv("AUTH_REQUIRE_AUTH", "true").strip().lower() not in (
        "false",
        "0",
        "no",
        "",
    )
    if not require_auth:
        result.add_warning(
            category,
            "AUTH_REQUIRE_AUTH",
            "身份校验已关闭，仅适合本地开发；企业内部知识库生产环境必须开启鉴权。",
        )
        return

    allow_untrusted = os.getenv(
        "AUTH_ALLOW_UNTRUSTED_IDENTITY_HEADERS",
        "false",
    ).strip().lower() in ("true", "1", "yes")
    trusted_secret = os.getenv("AUTH_TRUSTED_IDENTITY_SECRET", "").strip()
    if allow_untrusted:
        result.add_warning(
            category,
            "AUTH_ALLOW_UNTRUSTED_IDENTITY_HEADERS",
            "当前允许直接信任客户端 X-User-* 请求头，仅适合本地开发；生产环境应关闭并配置 AUTH_TRUSTED_IDENTITY_SECRET。",
        )
    elif not trusted_secret:
        result.add_error(
            category,
            "AUTH_TRUSTED_IDENTITY_SECRET",
            "未配置可信身份透传密钥。生产模式下服务不会信任 X-User-* / X-Roles 身份头，请在网关和服务端同时配置该密钥。",
        )


def validate_all_configs() -> ConfigValidationResult:
    """执行全部配置校验，返回汇总结果。"""
    result = ConfigValidationResult()
    _check_auth(result)
    _check_llm(result)
    _check_rag(result)
    return result


if __name__ == "__main__":
    validation_result = validate_all_configs()
    print(validation_result.report())
