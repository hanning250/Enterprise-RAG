"""模型工厂模块。"""
import functools
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from utils.config_handler import rag_conf

logger = logging.getLogger(__name__)

OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"
DASHSCOPE_BASE_URL_ENV = "DASHSCOPE_BASE_URL"
DASHSCOPE_API_KEY_ENV = "DASHSCOPE_API_KEY"

# DashScope 官方 OpenAI 兼容模式 endpoint（作为兜底默认值）。
# 文档：https://help.aliyun.com/zh/model-studio/developer-reference/use-qwen-by-calling-openai-sdk
# 作用：
#   1. 防止用户 .env 里漏写 / 笔误 DASHSCOPE_BASE_URL 导致 base_url=None，
#      然后 SDK 退回 api.openai.com，把 DashScope 的 key 发给 OpenAI → 401 Invalid token。
#   2. 用户若显式 set DASHSCOPE_BASE_URL 指向其他兼容层（例如 Maas 网关），就优先走用户配置。
DASHSCOPE_BASE_URL_DEFAULT = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _env_value(name: str) -> Optional[str]:
    value = os.environ.get(name, "").strip()
    return value or None


def _dashscope_base_url_resolved(*, warn_on_default: bool = True) -> str:
    """返回「最终对 DashScope 生效的 base_url」。

    优先级：DASHSCOPE_BASE_URL env > DASHSCOPE_BASE_URL_DEFAULT。
    当命中 default 时，warn_on_default=True 会打印一次 warning，便于自查 .env 笔误。
    """
    env = _env_value(DASHSCOPE_BASE_URL_ENV)
    if env:
        return env
    if warn_on_default:
        logger.warning(
            "[DashScope] %s 未设置或为空，回退到默认兼容模式 endpoint: %s。"
            " 若你期望走其他网关（如私有化/专属域名/阿里云 Maas 网关），请在 .env 中正确填写 %s。",
            DASHSCOPE_BASE_URL_ENV,
            DASHSCOPE_BASE_URL_DEFAULT,
            DASHSCOPE_BASE_URL_ENV,
        )
    return DASHSCOPE_BASE_URL_DEFAULT


def _mask_key(raw: Optional[str]) -> str:
    """把 API key 打码（只展示前 4 后 4 位），便于日志自查。"""
    if not raw:
        return "<EMPTY>"
    if len(raw) <= 8:
        return "<*" + "*" * max(0, len(raw) - 2) + ">"
    return f"{raw[:4]}***{raw[-4:]}"


class BaseModelFactory(ABC):
    """模型工厂抽象基类。"""

    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        """创建并返回一个模型实例。"""
        pass


class ChatModelFactory(BaseModelFactory):
    """聊天模型工厂类。"""

    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        """创建并返回一个 ChatOpenAI 聊天模型实例。"""
        llm_config = rag_conf.get("llm", {})
        model_name = llm_config.get("model_name") or rag_conf.get("chat_model_name")

        chat_kwargs = {
            "model": model_name,
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "base_url": _env_value(OPENAI_BASE_URL_ENV),
            # 总结链路用 invoke 收集完整答案；streaming=True 在部分网关会拖成「假死」
            "streaming": False,
            "max_retries": int(llm_config.get("max_retries", 1)),
            "timeout": float(llm_config.get("timeout_seconds", 90)),
        }

        max_output_tokens = llm_config.get("max_output_tokens")
        if max_output_tokens is not None and max_output_tokens > 0:
            chat_kwargs["max_tokens"] = max_output_tokens

        return ChatOpenAI(**chat_kwargs)


class JudgeLLMFactory(BaseModelFactory):
    """RAGAS 评估专用 Judge LLM 工厂。

    Judge LLM 完全复用 **DashScope** 的
    ``DASHSCOPE_BASE_URL`` + ``DASHSCOPE_API_KEY``，
    但允许使用独立于线上 ``chat_model`` 的：
    * 模型名（默认 ``llm.judge_llm.judge_model_name``，
      不填就退化到 ``llm.model_name``）
    * 最大输出 token（评估只需短 JSON，省 token）
    * temperature（评估应尽量确定，默认 0.0）
    """

    def generator(self) -> Optional[BaseChatModel]:  # Judge 只造 ChatModel，不做 Embeddings
        llm_config = rag_conf.get("llm", {}) or {}
        judge_config = llm_config.get("judge_llm", {}) or {}

        # ------ P1 #2：删除 "空就 fallback 到 llm.model_name / chat_model_name" 的降级链 ------
        # 为什么删掉：
        #   线上 chat_model（gpt-5.5 / qwen3.5-flash）通常用 OPENAI_BASE_URL + OPENAI_API_KEY，
        #   但 Judge 永远固定发去 DASHSCOPE_BASE_URL + DASHSCOPE_API_KEY。
        #   一旦"不填 judge_model_name 就猜你要用线上"，会把线上模型名（比如 gpt-5.5）
        #   发到 DashScope，DashScope 不认识 → model_not_found / 404，再被第 5 条的静默吞掉
        #   → 0 分。根本没法排查。
        #
        # 现在：judge_model_name 必填，不填就 ValueError 明确告诉用户"要显式配"。
        raw_name = (judge_config.get("judge_model_name") or "").strip()
        if not raw_name:
            raise ValueError(
                "[JudgeLLMFactory] config/rag.yml 要求必填 llm.judge_llm.judge_model_name。"
                " 当前为空。\n"
                "  - 如果你想用 DashScope 专属大模型评估：填例如 qwen3-235b-a22b-instruct-2507\n"
                "  - 如果你明确想复用线上 chat_model（连同它的 OPENAI_BASE_URL / API_KEY）："
                "不要走 fallback，先自己把 chat_model 的模型名显式复制到"
                " llm.judge_llm.judge_model_name，并确认你要的 provider 是正确的。"
            )
        model_name = raw_name

        dashscope_api_key = _env_value(DASHSCOPE_API_KEY_ENV)
        dashscope_base_url = _dashscope_base_url_resolved()
        if not dashscope_api_key:
            raise ValueError(
                f"JudgeLLMFactory 需要环境变量 {DASHSCOPE_API_KEY_ENV} 未设置。"
                " 请在 .env 或系统环境变量中填好 DashScope API Key。"
            )

        chat_kwargs: dict[str, Any] = {
            "model": model_name,
            "api_key": dashscope_api_key,
            "base_url": dashscope_base_url,
            "streaming": False,  # Judge 不需要流式输出，省连接开销
            "max_retries": 3,
            "timeout": 120.0,   # P2 #1：235B 大模型推理慢，给 120 秒默认超时，避免 SDK 默认 60 秒被掐断
        }

        # 独立 max_tokens / temperature
        max_tokens = judge_config.get("max_output_tokens")
        try:
            max_tokens_int = int(max_tokens)  # type: ignore[arg-type]
            if max_tokens_int > 0:
                chat_kwargs["max_tokens"] = max_tokens_int
        except (TypeError, ValueError):
            pass

        temp = judge_config.get("temperature")
        try:
            temp_f = float(temp)  # type: ignore[arg-type]
            chat_kwargs["temperature"] = temp_f
        except (TypeError, ValueError):
            # 不填默认 0.0（评估应尽量确定）
            chat_kwargs["temperature"] = 0.0

        return ChatOpenAI(**chat_kwargs)


class EmbeddingsFactory(BaseModelFactory):
    """向量嵌入模型工厂类。"""

    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        """创建并返回一个 OpenAIEmbeddings 向量嵌入模型实例。"""
        return OpenAIEmbeddings(
            model=rag_conf["embedding_model_name"],
            api_key=os.environ.get("DASHSCOPE_API_KEY"),
            base_url=_dashscope_base_url_resolved(),
            check_embedding_ctx_length=False,
        )


chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()


# Judge LLM 不应该"启动 RAG 服务时就强制初始化"：
# 它仅用于 eval/run_generation_eval.py 这类离线评测脚本。
# 如果把 judge_llm = JudgeLLMFactory().generator() 放在模块顶层，
# 任何业务模块只要 `from model.factory import chat_model` 都会在模块加载期
# 触发 Judge 初始化：DASHSCOPE_API_KEY 缺失、模型名笔误、网络不通，都会
# 让线上聊天/RAG 服务一起挂。这属于"职责污染"，必须彻底断开。
#
# 方案 A：用 @functools.lru_cache(maxsize=1) 做"延迟评估的单例"：
#   - 线上永远不调 get_judge_llm → Judge 完全不实例化；
#   - 评测脚本第一次调用时才实例化；
#   - 多次调用仍然复用同一个实例，不会重复打开 HTTP 连接池。
@functools.lru_cache(maxsize=1)
def get_judge_llm() -> BaseChatModel:
    """获取 RAGAS 评估专用 Judge LLM（延迟初始化，首次调用才真的造实例）。

    注意：线上 RAG/Agent 服务永远不要调用这个函数；只有 eval 脚本才用它。
    """
    factory = JudgeLLMFactory()
    result = factory.generator()
    if result is None:
        raise RuntimeError(
            "JudgeLLMFactory.generator() 返回 None。请检查 config/rag.yml "
            "llm.judge_llm.judge_model_name 是否已填。"
        )
    return result
