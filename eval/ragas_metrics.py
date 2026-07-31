"""Official RAGAS adapter for generation evaluation.

This module is the **only** place where RAGAS-specific imports and result
normalization happen. All generation evaluation uses RAGAS official metrics.

Metrics: faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness
"""

from __future__ import annotations

import json
import logging
import math
import re
import sys
import types
import warnings
from typing import Any, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun


# ======================================================================
# 0. OPTION A：不碰依赖的兼容补丁（解决 langchain-community 0.4.2 sunset 导致 ragas 0.4.3
#    from langchain_community.chat_models.vertexai import ChatVertexAI 崩掉的问题）。
#    我们实际上用 DashScope/通义，永远不会实例化 VertexAI，所以打占位模块即可。
#    此补丁幂等：重复调用没有副作用。
# ======================================================================
def _apply_langchain_community_sunset_patch() -> None:
    # 0.1 vertexai chat
    _CV_NAME = "langchain_community.chat_models.vertexai"
    if _CV_NAME not in sys.modules:
        _mod = types.ModuleType(_CV_NAME)
        _mod.__dict__["ChatVertexAI"] = type("ChatVertexAI", (), {})
        sys.modules[_CV_NAME] = _mod
    elif not hasattr(sys.modules[_CV_NAME], "ChatVertexAI"):
        setattr(sys.modules[_CV_NAME], "ChatVertexAI", type("ChatVertexAI", (), {}))

    # 0.2 vertexai llm (ragas/llms/base.py 也引了)
    _LV_NAME = "langchain_community.llms"
    try:
        import langchain_community.llms  # noqa: F401  # 先 import 确保已进 sys.modules
    except Exception:
        # langchain-community 偶尔出现子包不可导入，兜底造一个
        if _LV_NAME not in sys.modules:
            sys.modules[_LV_NAME] = types.ModuleType(_LV_NAME)
    if _LV_NAME in sys.modules and not hasattr(sys.modules[_LV_NAME], "VertexAI"):
        setattr(sys.modules[_LV_NAME], "VertexAI", type("VertexAI", (), {}))


# 立即执行（模块加载期就打好，避免用户 import ragas 先炸）
_apply_langchain_community_sunset_patch()


_ragas_logger = logging.getLogger("ragas")
_ragas_exec_logger = logging.getLogger("ragas.execution")
for _lg in (_ragas_logger, _ragas_exec_logger):
    _lg.setLevel(logging.WARNING)


# P2 #1：过滤 DashScope OpenAI 兼容层偶尔返回 usage=null 导致的 langchain_openai 警告
# 位置：langchain_openai/chat_models/base.py:550
#   warnings.warn(f"Unexpected type for token usage: {type(new_usage)}")
# 这是无害警告（usage=None 不影响输出内容），但会刷屏并让用户误以为 judge 挂了。
warnings.filterwarnings(
    "ignore",
    message=r"Unexpected type for token usage.*",
    category=UserWarning,
    module=r"langchain_openai\..*",
)
warnings.filterwarnings(
    "ignore",
    message=r"Unexpected type for token usage.*",
    category=UserWarning,
)


class _RagasGenerationSilencer(logging.Filter):
    _KEYWORDS = (
        "LLM returned 1 generations instead of requested",
        "Proceeding with 1 generations",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(kw in msg for kw in self._KEYWORDS)


for _lg_name in ("ragas", "ragas.execution", "ragas.metrics", "root"):
    logging.getLogger(_lg_name).addFilter(_RagasGenerationSilencer())


_FENCE_RE = re.compile(
    r"^```(?:json|JSON)?[ \t]*\n([\s\S]*?)```[ \t]*$",
    re.MULTILINE,
)
_BRACKET_RE = re.compile(r"[\{\[]")


def _coerce_to_valid_json_text(content: Any) -> str:
    """把模型返回的脏 JSON 形式清洗成合法 JSON 字符串。

    RAGAS 的 faithfulness / statement classification 等指标会调用
    ``model_validate_json()``，它要求模型输出是严格的 JSON 对象/数组字面量
    （外层直接是 ``{...}`` 或 ``[...]``，用标准双引号，不允许转义）。

    现实里的 LLM 经常输出下面几种脏形式，本函数全部兼容：

    * Case A：合法 JSON，只是前后有空白或说明文字 → 截取首尾的 ``{ }`` / ``[ ]`` 范围
    * Case B：外层包了 Markdown 代码块 → 去掉 ```json ... ``` 围栏
    * Case C：双编码（最常见的失败根因）：content 本身是字符串，内容是
      ``"{\\"statements\\":[...]}"`` 这种 Python repr 式的转义字符串
      → 用 ``json.loads`` 解一层字符串，直到拿到 dict/list
    * Case D：混用中文引号 / 单引号 → 保守地把中文引号 ``" "`` 替换成 ASCII `"`，
      不处理单引号（太容易误伤）

    返回值：始终是 ``str`` 类型。如果内容根本不是 JSON（纯文本），直接原样返回，
    让 RAGAS 的 parser 自己报错；因为不是 JSON 时我们不应该伪造。
    """
    if content is None:
        return ""

    text = content if isinstance(content, str) else str(content)
    text = text.strip()
    if not text:
        return ""

    # Case B：去掉 Markdown 代码块围栏（只取最外层第一个）
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    # Case C：解双编码。
    # 策略：当首尾字符是 ``"`` 或 ``'`` 时，用 json.loads 尝试解码一次；
    # 如果解出来还是字符串并且首尾又是引号，就迭代直到拿到对象或失败为止。
    # 用最多 3 次上限防死循环。
    decoded: Any = text
    for _ in range(3):
        if isinstance(decoded, str):
            stripped = decoded.strip()
            if len(stripped) >= 2 and (
                (stripped[0] == '"' and stripped[-1] == '"')
                or (stripped[0] == "'" and stripped[-1] == "'")
            ):
                try:
                    parsed = json.loads(stripped)
                except (ValueError, json.JSONDecodeError):
                    # 不是合法字符串 JSON，跳过这层解析
                    break
                decoded = parsed
                continue
        # 已经是 dict / list → 退出
        if isinstance(decoded, (dict, list)):
            break

    # 如果最终解析成了 dict / list，统一 dump 成**紧凑合法**的 JSON 字符串
    if isinstance(decoded, (dict, list)):
        try:
            cleaned = json.dumps(decoded, ensure_ascii=False)
        except (TypeError, ValueError):
            cleaned = text
    else:
        cleaned = decoded if isinstance(decoded, str) else str(decoded)

    cleaned = cleaned.strip()
    if not cleaned:
        return ""

    # Case A：如果仍然不是以 { 或 [ 开头，但文本里含 { / [，
    # 取第一个 { / [ 到最后一个 } / ] 的范围（兼容前后有自然语言解释）
    if cleaned[0] not in "{[":
        first_open = _BRACKET_RE.search(cleaned)
        if first_open is None:
            # 真的不是 JSON，原样返回（让上层 parser 决定）
            return cleaned
        start = first_open.start()
        first_char = cleaned[start]
        end_char = "}" if first_char == "{" else "]"
        end = cleaned.rfind(end_char)
        if end <= start:
            return cleaned
        candidate = cleaned[start : end + 1].strip()
        # 轻量校验：看看是不是合法 JSON；合法才替换
        try:
            json.loads(candidate)
        except (ValueError, json.JSONDecodeError):
            return cleaned
        cleaned = candidate

    # 把中文左右双引号统一替换成 ASCII 双引号（只在确实用了中文引号时才触发，避免误伤字符串内部）
    if "\u201c" in cleaned or "\u201d" in cleaned:
        cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"')

    return cleaned


def _sanitize_generation(gen: ChatGeneration) -> ChatGeneration:
    """对单个 ChatGeneration 的 message content 做 JSON 清洗。"""
    msg = gen.message
    original_content = getattr(msg, "content", None)
    coerced = _coerce_to_valid_json_text(original_content)
    if coerced == original_content:
        return gen
    # 构造一个新的 AIMessage（保持其他字段，替换 content）
    kwargs: dict[str, Any] = {"content": coerced}
    for attr in ("name", "id", "additional_kwargs", "response_metadata"):
        value = getattr(msg, attr, None)
        if value is not None:
            kwargs[attr] = value
    # 保持 message 的实际类型（如果是 HumanMessage / SystemMessage 等非常规子类）
    # 默认用 AIMessage，因为评估链路里模型永远返回 AI 内容
    new_msg: BaseMessage
    try:
        new_msg = msg.__class__(**kwargs)  # type: ignore[call-arg]
    except Exception:
        new_msg = AIMessage(**kwargs)
    return ChatGeneration(message=new_msg, generation_info=gen.generation_info)


def _as_chat_generation(item: Any) -> Optional[ChatGeneration]:
    """健壮地把 chat generation 容器里的元素「规范化」为 ChatGeneration。

    兼容 DashScope / 第三方 provider 偶尔出现的以下脏形式：

    * 真 ChatGeneration 对象 → 直接返回
    * ``(ChatGeneration, extra)`` 二元组（部分 provider 旧实现）→ 取 ``[0]`` 再递归
    * 仅有 ``.message`` 属性的自定义生成对象 → 包一层 ChatGeneration
    * 其他 → ``None``（调用方应跳过）
    """
    if isinstance(item, ChatGeneration):
        return item
    if isinstance(item, tuple):
        if not item:
            return None
        return _as_chat_generation(item[0])
    msg = getattr(item, "message", None)
    if isinstance(msg, BaseMessage):
        info = getattr(item, "generation_info", None)
        return ChatGeneration(message=msg, generation_info=info)
    return None


def _normalize_prompt_generations(items: Any) -> list[ChatGeneration]:
    """把 ChatResult 或 LLMResult 一层里的 prompt generations 清洗成 list[ChatGeneration]。"""
    if not items:
        return []
    if not isinstance(items, (list, tuple)):
        single = _as_chat_generation(items)
        return [single] if single is not None else []
    out: list[ChatGeneration] = []
    for it in items:
        gen = _as_chat_generation(it)
        if gen is not None:
            out.append(gen)
    return out


class _RagasSafeLLMWrapper(BaseChatModel):
    """Safety wrapper applied to the evaluator LLM before handing it to RAGAS.

    RAGAS's internal prompts expect two things from the underlying LLM that most
    API providers (DashScope, many OpenAI-compatible endpoints) do NOT reliably
    deliver:

      1. **Multi-generation support** — RAGAS requests ``n=3`` or higher for
         self-consistency / majority-vote metrics such as ``faithfulness`` and
         ``answer_correctness``.  Providers that silently ignore ``n`` cause a
         warning::

             LLM returned 1 generations instead of requested 3. Proceeding with
             1 generations.

         and, worse, the self-consistency path is skipped entirely.

      2. **Strictly valid JSON output** — RAGAS parses statement-classification
         and statement-extraction outputs with ``model_validate_json``, so any
         wrapping text, code-block fences, smart quotes, or double-encoded JSON
         strings break the metric immediately.

    This wrapper fixes both issues:
      * Always call the underlying LLM with ``n = 1``, then **clone** the single
        ``ChatGeneration`` exactly ``requested_n`` times so RAGAS always sees the
        number of candidates it asked for.
      * Run every returned message content through
        :func:`_coerce_to_valid_json_text` so the caller receives valid JSON
        whenever the model produced something JSON-like.
    """

    _internal: BaseChatModel

    def __init__(self, internal: BaseChatModel):
        super().__init__()
        object.__setattr__(self, "_internal", internal)

    @property
    def _llm_type(self) -> str:
        return getattr(self._internal, "_llm_type", "chat")

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return getattr(self._internal, "_identifying_params", {}) or {}

    def bind(self, **kwargs: Any) -> "_RagasSafeLLMWrapper":
        bound = self._internal.bind(**kwargs)
        return _RagasSafeLLMWrapper(bound)

    def _generate(
        self,
        messages: list[list[BaseMessage]],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        requested_n: int = int(kwargs.pop("n", 1) or 1)
        kwargs["n"] = 1
        raw: ChatResult = self._internal._generate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )
        raw = self._sanitize_chat_result(raw)

        if requested_n <= 1:
            return raw

        # ChatResult 语义：单个 prompt，generations 是「该 prompt 的候选生成列表」——单层 list[ChatGeneration]
        norm_raw = _normalize_prompt_generations(raw.generations)
        if not norm_raw:
            return raw
        first = norm_raw[0]
        cloned_generations = [first for _ in range(requested_n)]
        return ChatResult(
            generations=cloned_generations,
            llm_output=dict(raw.llm_output or {}),
        )

    async def _agenerate(
        self,
        messages: list[list[BaseMessage]],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        requested_n: int = int(kwargs.pop("n", 1) or 1)
        kwargs["n"] = 1
        raw: ChatResult = await self._internal._agenerate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )
        raw = self._sanitize_chat_result(raw)

        if requested_n <= 1:
            return raw

        norm_raw = _normalize_prompt_generations(raw.generations)
        if not norm_raw:
            return raw
        first = norm_raw[0]
        cloned_generations = [first for _ in range(requested_n)]
        return ChatResult(
            generations=cloned_generations,
            llm_output=dict(raw.llm_output or {}),
        )

    @staticmethod
    def _sanitize_chat_result(result: ChatResult) -> ChatResult:
        """对 ChatResult（单层 generations=list[ChatGeneration]）做 tuple 规范化 + JSON sanitize。"""
        norm = _normalize_prompt_generations(result.generations)
        sanitized: list[ChatGeneration] = []
        any_changed = len(norm) != len(result.generations)
        for g in norm:
            sg = _sanitize_generation(g)
            if sg is not g:
                any_changed = True
            sanitized.append(sg)
        if not any_changed:
            return result
        return ChatResult(
            generations=sanitized,
            llm_output=dict(result.llm_output or {}),
        )

    def _combine_llm_outputs(self, llm_outputs: list[Optional[dict[str, Any]]]) -> dict[str, Any]:
        try:
            return self._internal._combine_llm_outputs(llm_outputs)
        except Exception:
            return {}

    def get_num_tokens(self, text: str) -> int:
        try:
            return int(self._internal.get_num_tokens(text))
        except Exception:
            return super().get_num_tokens(text)

    def get_num_tokens_from_messages(self, messages: list[BaseMessage]) -> int:
        try:
            return int(self._internal.get_num_tokens_from_messages(messages))
        except Exception:
            return super().get_num_tokens_from_messages(messages)

    @property
    def _default_params(self) -> dict[str, Any]:
        return getattr(self._internal, "_default_params", {}) or {}


RAGAS_METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
)

# ============================================================
# P1 #4：默认核心指标集合（不依赖 reference，抗关键词污染）
# 原因：
#   context_recall / answer_correctness 需要一段「自然语言 ground truth
#   标准答案」作为 reference 对比语义；但我们的数据集 answer_points 大多是
#   散点关键词（"435" / "官方App" / "证件安全"），不是完整标准答案。
#   直接拿关键词拼接字符串去比对，会让这两个指标严重失真，因此：
#     * DEFAULT_RAGAS_CORE_METRICS = 3 个不依赖 reference 的核心（默认启用）
#     * REFERENCE_SENSITIVE_METRICS  = {context_recall, answer_correctness}
#       （仅当你显式 --metrics ... 传它们才启用；且在报告里标记为「实验性」）
# ============================================================
DEFAULT_RAGAS_CORE_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
)

REFERENCE_SENSITIVE_METRICS = frozenset({"context_recall", "answer_correctness"})

_RAGAS_OUTPUT_ALIASES = {
    "answer_relevancy": ("answer_relevancy", "answer_relevance", "response_relevancy"),
    "context_recall": ("context_recall", "context_recall_classification"),
}


def build_reference(sample: dict[str, Any]) -> str:
    """按 P1 #4 优先级构造 reference：
    1. 优先取 dataset 样本里已有的 `reference_answer`（标准自然语言 ground truth）
    2. 没有就退化成 answer_points 拼接（仅作趋势参考，不建议和 reference_answer 的 run A/B）
    """
    # 1. 已有完整标准答案
    if isinstance(sample, dict):
        ref = (sample.get("reference_answer") or "").strip()
        if ref:
            return ref
        # 兼容老字段名（不同标注工具可能起别的名）
        for alt in ("ground_truth", "expected_answer", "reference"):
            ref = (sample.get(alt) or "").strip()
            if ref:
                return ref
        # 2. 退化：answer_points 拼接
        pts = sample.get("answer_points")
    else:
        pts = None
    return "\n".join(str(p) for p in (pts or []) if str(p).strip())


def reference_from_answer_points(answer_points: list[str] | None) -> str:
    """旧接口，保留兼容（新代码直接用 build_reference 更合适）。"""
    return build_reference({"answer_points": answer_points})


def _load_ragas_objects():
    # 先幂等打一次补丁（防御式：哪怕被 import 顺序改了也安全）
    _apply_langchain_community_sunset_patch()

    # 0.4.3：DeprecationWarning → 优先从 collections 新路径导入（里面暴露的是「类」，
    #        不是模块）；失败回退旧路径 ragas.metrics（0.3.x 旧形式）。
    try:
        from ragas import EvaluationDataset, evaluate
        try:
            # 先取"类模板"，实例化时再注入 llm/embeddings
            from ragas.metrics.collections import (
                AnswerCorrectness as _ACCls,
                AnswerRelevancy as _ARCls,
                ContextPrecision as _CPCls,
                ContextRecall as _CRCls,
                Faithfulness as _FCls,
            )
            metrics_dict = {
                "faithfulness": _FCls(),
                "answer_relevancy": _ARCls(),
                "context_precision": _CPCls(),
                "context_recall": _CRCls(),
                "answer_correctness": _ACCls(),
            }
            # 把 DeprecationWarning 已经消除，不需要再发
        except (ImportError, Exception):
            # 老路径兜底（0.3.x 风格）：尽量实例化成"类"再返回，不是 module
            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter("ignore", DeprecationWarning)
                import ragas.metrics as _old_mod
                from ragas.metrics import (
                    answer_correctness as _legacy_ac,
                    answer_relevancy as _legacy_ar,
                    context_precision as _legacy_cp,
                    context_recall as _legacy_cr,
                    faithfulness as _legacy_faith,
                )
                # 兼容：如果是类就调用它；如果是"默认实例"直接当模板用
                def _to_instance(x: Any) -> Any:
                    try:
                        return x() if isinstance(x, type) else x
                    except Exception:
                        return x
                metrics_dict = {
                    "faithfulness": _to_instance(_legacy_faith),
                    "answer_relevancy": _to_instance(_legacy_ar),
                    "context_precision": _to_instance(_legacy_cp),
                    "context_recall": _to_instance(_legacy_cr),
                    "answer_correctness": _to_instance(_legacy_ac),
                }
    except ImportError as exc:
        raise ImportError(
            "未安装 ragas。请先执行：python -m pip install ragas==0.4.3"
        ) from exc

    return {
        "EvaluationDataset": EvaluationDataset,
        "evaluate": evaluate,
        "metrics": metrics_dict,
    }


def _as_ragas_llm(
    judge_llm: BaseLanguageModel,
) -> Any:
    """把 LangChain BaseChatModel / BaseLLM 转成 ragas 需要的 BaseRagasLLM。

    ragas 0.4.x 用的是 `ragas.llms._LangchainLLMWrapper(...)。
    """
    # 幂等打补丁，保证 import ragas.llms 不炸
    _apply_langchain_community_sunset_patch()

    from ragas.llms import _LangchainLLMWrapper  # noqa: E402
    return _LangchainLLMWrapper(langchain_llm=judge_llm)


def _configure_ragas_metric(
    metric_template: Any,
    *,
    ragas_llm: Any | None,
    ragas_embeddings: Any | None,
    n: int = 1,
) -> Any:
    """基于"指标模板实例"生成一个新实例，注入 LLM/embeddings/n 并正确返回。

    ragas 0.4.3：直接 **Faithfulness(llm=..., embeddings=..., n=...)** 构造函数形式，
    不再需要“把实例二次套壳 type(_m)(llm=...) 这种容易失败的写法。
    """
    cls = metric_template.__class__
    # 看构造函数参数支持哪些字段：
    import inspect as _inspect
    try:
        params = set(_inspect.signature(cls.__init__).parameters.keys())
    except (TypeError, ValueError):
        params = set()

    kwargs: dict[str, Any] = {}
    if "llm" in params and ragas_llm is not None:
        kwargs["llm"] = ragas_llm
    if "embeddings" in params and ragas_embeddings is not None:
        kwargs["embeddings"] = ragas_embeddings
    if "n" in params:
        kwargs["n"] = int(n)
    # 复制模板实例的其他字段（从旧实例到新实例）：复制 _required_columns / prompts 等
    if hasattr(metric_template, "__dict__"):
        for k, v in vars(metric_template).items():
            # 跳过：我们要显式注入的 llm/embeddings/n 不覆盖
            if k in kwargs or k.startswith("__"):
                continue
            if k == "_required_columns" and not kwargs:
                kwargs[k] = v
                continue
            kwargs.setdefault(k, v)
    try:
        return cls(**kwargs)
    except TypeError:
        # 万一有些子类 init 不接收非预期字段（如 *_prompt 之类），退化成：用 ragas 官方 template 里 llm/embeddings 属性 setattr
        obj = metric_template
        if ragas_llm is not None:
            try:
                setattr(obj, "llm", ragas_llm)
            except Exception:
                pass
        if ragas_embeddings is not None:
            try:
                setattr(obj, "embeddings", ragas_embeddings)
            except Exception:
                pass
        return obj



def _normalize_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(score):
        return None
    return max(0.0, min(1.0, score))


def scores_only(results: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    """Extract only valid score values from metric results."""
    return {
        metric: data.get("score")
        for metric, data in results.items()
    }


def _score_from_row(row: dict[str, Any], metric_name: str) -> tuple[float | None, str]:
    aliases = _RAGAS_OUTPUT_ALIASES.get(metric_name, (metric_name,))
    for alias in aliases:
        if alias in row:
            score = _normalize_score(row[alias])
            if score is None:
                return None, f"RAGAS 返回无效分数：{alias}={row[alias]!r}"
            return score, "RAGAS 官方指标"
    return None, "RAGAS 未返回该指标，可能是解析失败或评估异常"


def evaluate_with_ragas(
    samples: list[dict[str, Any]],
    metrics: tuple[str, ...],
    judge_llm: BaseChatModel,
    embeddings: Embeddings | None = None,
    show_progress: bool = False,
    *,
    judge_n: int = 1,
    max_workers: int = 4,
) -> list[dict[str, dict[str, Any]]]:
    """Evaluate generated RAG answers with official RAGAS metrics.

    Parameters
    ----------
    judge_n:
        传给每个 RAGAS metric 的 self-consistency 候选数（对应 ``Faithfulness(n=...)``）。
        默认 1（诚实的"不做伪多样本"）。
        注意：国内绝大多数 OpenAI 兼容 provider（DashScope / Maas / SiliconFlow）实际不会真的
        按 ``n>1`` 输出多条 generation，只会返回 1 条；此时我们的 ``_RagasSafeLLMWrapper``
        会把这 1 条 *克隆 n 份* 交给 ragas，以保证 ragas 的 self-consistency 代码路径不警告、
        不报错。但"同一份克隆 N 份"数学上等价于 n=1，不是真的多数投票，所以 judge_n>1
        **不应用于与 n=1 的 A/B 对比**，仅用于验证 ragas 代码路径。
    """
    if not samples:
        return []
    if judge_n is None or judge_n < 1:  # 兜底防 None
        judge_n = 1
    judge_n = int(judge_n)

    ragas = _load_ragas_objects()
    valid_metrics = tuple(metric for metric in metrics if metric in RAGAS_METRIC_NAMES)
    if not valid_metrics:
        return []

    # 先把评估 LLM 包一层 _RagasSafeLLMWrapper（保证 1) n>1 请求不会因 provider 忽视 n 而丢候选；
    # 2) 模型输出的 JSON 脏值先被清洗成严格合法 JSON，让 ragas 的 model_validate_json 不出错）
    if not isinstance(judge_llm, _RagasSafeLLMWrapper):
        judge_llm = _RagasSafeLLMWrapper(judge_llm)

    # 再转成 ragas 需要的 BaseRagasLLM / BaseRagasEmbeddings
    ragas_llm = _as_ragas_llm(judge_llm)
    ragas_embeddings: Any | None = None
    if embeddings is not None:
        try:
            from ragas.embeddings import _LangchainEmbeddingsWrapper
            ragas_embeddings = _LangchainEmbeddingsWrapper(embeddings=embeddings)
        except Exception:
            # 兜底：老版本 ragas 可能不暴露 wrapper，暂时不传 embeddings（answer_relevancy 等
            # 指标如果严格需要 embeddings，ragas 内部会 fallback 到默认——这里只做"尽量传"）
            ragas_embeddings = None

    dataset_rows = []
    for sample in samples:
        dataset_rows.append({
            "user_input": sample["question"],
            "response": sample["answer"],
            "retrieved_contexts": sample.get("contexts", []),
            "reference": sample.get("reference", ""),
        })

    dataset = ragas["EvaluationDataset"].from_list(dataset_rows)
    metric_templates = [ragas["metrics"][metric] for metric in valid_metrics]
    metric_objects: list[Any] = []
    for _m in metric_templates:
        # 不再写死 n=1（之前"包装器 n>1 复制 / metric 构造又写死 n=1"的自相矛盾已修复）
        configured = _configure_ragas_metric(
            _m,
            ragas_llm=ragas_llm,
            ragas_embeddings=ragas_embeddings,
            n=judge_n,
        )
        metric_objects.append(configured)

    # P2 #2：并发 + 超时配置，通过 max_workers 参数控制
    # - 快模型（qwen-plus / qwen-turbo）：max_workers=4，4 路并发，不触发限流
    # - 慢模型（qwen3-235b / qwen-max）：max_workers=1，串行避免超时
    _run_config = (
        __import__("ragas.run_config", fromlist=["RunConfig"]).RunConfig(
            max_workers=max_workers,
            timeout=300,
            max_retries=3,
        )
    )

    result = ragas["evaluate"](
        dataset,
        metrics=metric_objects,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=False,
        show_progress=show_progress,
        run_config=_run_config,
    )

    score_rows = getattr(result, "scores", [])
    normalized_results: list[dict[str, dict[str, Any]]] = []
    for score_row in score_rows:
        metric_results = {}
        for metric in valid_metrics:
            score, reason = _score_from_row(score_row, metric)
            metric_results[metric] = {
                "metric": metric,
                "score": score,
                "reason": reason,
                "valid": score is not None,
            }
        normalized_results.append(metric_results)

    return normalized_results
