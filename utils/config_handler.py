"""配置文件读取模块"""

import yaml
from utils.path_tool import get_abs_path

try:
    from dotenv import load_dotenv
except ImportError:
    raise ImportError(
        "缺少 python-dotenv 依赖。请先执行 `pip install -r requirements.txt`"
    ) from None

import os as _os
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
_env_path = _PROJECT_ROOT / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path, override=False)
else:
    import warnings as _warnings
    _warnings.warn(
        f"未找到 .env 文件（期望路径：{_env_path}）。"
        "如果通过系统环境变量注入密钥可忽略。",
        stacklevel=2,
    )
del _os, _Path, _PROJECT_ROOT, _env_path


def load_rag_config(config_path: str = get_abs_path("config/rag.yml"), encoding: str = "utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_chroma_config(config_path: str = get_abs_path("config/chroma.yml"), encoding: str = "utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_prompts_config(config_path: str = get_abs_path("config/prompts.yml"), encoding: str = "utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def _validate_llm_context_budget(conf: dict) -> None:
    """校验 LLM 上下文窗口预算是否安全，超窗直接报错（fail-fast）。"""
    import warnings

    RESERVED_PROMPT_TOKENS = 2000

    llm_cfg = conf.get("llm") or {}
    rag_cfg = conf.get("rag") or {}
    context_window = llm_cfg.get("context_window")
    max_output_tokens = llm_cfg.get("max_output_tokens")
    max_context_tokens = rag_cfg.get("max_context_tokens")

    if context_window is None or max_output_tokens is None or max_context_tokens is None:
        return

    total_budget = int(max_context_tokens) + int(max_output_tokens) + RESERVED_PROMPT_TOKENS
    if total_budget >= int(context_window):
        raise ValueError(
            "[RAG预算校验失败] LLM 上下文窗口不足：\n"
            f"  - llm.context_window      = {context_window} tokens\n"
            f"  - rag.max_context_tokens  = {max_context_tokens} tokens\n"
            f"  - llm.max_output_tokens   = {max_output_tokens} tokens\n"
            f"  - reserved_prompt_tokens  = {RESERVED_PROMPT_TOKENS} tokens\n"
            f"  - 合计                    = {total_budget} tokens\n"
            f"  请调小 rag.max_context_tokens 或 llm.max_output_tokens。"
        )

    if total_budget >= int(context_window) * 0.75:
        warnings.warn(
            f"[RAG预算告警] 上下文窗口占用率较高："
            f"{total_budget}/{context_window} tokens（{total_budget/int(context_window)*100:.1f}%）"
        )


rag_conf = load_rag_config()
_validate_llm_context_budget(rag_conf)

chroma_conf = load_chroma_config()
prompts_conf = load_prompts_config()


if __name__ == '__main__':
    print(chroma_conf["collection_name"])
