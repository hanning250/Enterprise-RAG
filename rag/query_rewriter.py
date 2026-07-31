"""Query 改写模块：针对冷门/模糊/过短查询，生成 1~2 条同义改写查询，配合多路召回融合。

设计原则：
  - 不造新 LLM：复用 model.factory.chat_model（同一个 LangChain ChatOpenAI 单例）
  - 有廉价兜底：当 LLM 改写失败 / 配置关闭时，返回 [original_query] 纯数组，不中断链路
  - 少调用：短问题 2 条改写，长问题 1 条改写，极短词（≤4字）走同义词规则直接补
  - 输出格式强约束：只输出 JSON 数组字符串，便于 parse 失败时兜底
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from model.factory import chat_model
from utils.config_handler import rag_conf
from utils.logger_handler import logger

# ── 规则式同义词字典（极短问题兜底；无需 LLM 调用）────────────────────
# 企业内部知识库高频词；后续可扩展为独立配置。
_RULE_SYNONYMS: dict[str, list[str]] = {
    # 薪酬 / 社保
    "工资": ["工资条", "实发工资", "应发工资", "薪资"],
    "薪资": ["工资", "实发工资", "应发工资"],
    "社保": ["社会保险", "五险一金", "社保缴费"],
    "公积金": ["住房公积金", "五险一金", "公积金缴存"],
    "个税": ["个人所得税", "扣税", "税费"],
    # 假期 / 考勤
    "年假": ["带薪年假", "年假天数", "请假制度 年假", "年假申请审批"],
    "请假": ["请假制度", "事假病假年假", "企业微信请假审批"],
    "事假": ["事假申请", "请假制度 事假"],
    "病假": ["病假证明", "请假制度 病假"],
    "调休": ["加班调休", "请假制度 调休"],
    "加班": ["加班费", "加班调休", "加班申请"],
    # 制度 / 财务
    "报销": ["费用报销", "报销流程", "报销制度"],
    "制度": ["公司制度", "内部制度", "管理办法"],
    "合同": ["劳动合同", "合同条款", "签署合同"],
}


@dataclass(frozen=True)
class QueryRewriteResult:
    """Query 改写结果包装类。"""

    original: str
    rewrites: list[str]

    @property
    def all_queries(self) -> list[str]:
        """按优先级排序的完整 query 列表：原文优先，改写随后，去重。"""
        out: list[str] = [self.original]
        for q in self.rewrites:
            if q and q not in out:
                out.append(q)
        return out


def _query_is_short_or_ambiguous(query: str) -> bool:
    """判断是否需要改写。

    触发条件（任一即可）：
      - 总长度 ≤ 4（单字/双字词，例如 "工资"、"请假"）
      - 总长度 ≤ 8 且没有任何 CJK 标点 + 动词，疑似关键词搜索
    """
    text = (query or "").strip()
    if not text:
        return False
    if len(text) <= 4:
        return True
    if len(text) <= 8 and not any(ch in "，。？！、；：,.?!" for ch in text):
        # 短纯名词短语：大概率可以扩一下
        return True
    return False


def _rule_based_rewrites(query: str) -> list[str]:
    """规则式同义词兜底（不调 LLM）。

    遍历整句中出现的关键词，把整句替换成多种等价问法。
    规则命中失败返回空列表，调用方会继续走 LLM。
    """
    text = (query or "").strip()
    if not text:
        return []
    # 1) 精确全匹配
    if text in _RULE_SYNONYMS:
        return list(_RULE_SYNONYMS[text])
    # 2) 部分匹配：把关键词替换 → 生成新问法
    produced: list[str] = []
    for kw, synonyms in _RULE_SYNONYMS.items():
        if kw in text:
            for syn in synonyms:
                alt = text.replace(kw, syn, 1)
                if alt != text and len(alt) <= 40:
                    produced.append(alt)
    # 去重保留顺序
    seen: set[str] = set()
    uniq: list[str] = []
    for s in produced:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq[:2]


_REWRITE_PROMPT_TEMPLATE = """你是一个中文 RAG 查询改写专家。请严格只输出一个 JSON 数组（不要 markdown 代码块、不要解释文字）。

原始查询：
"{original_query}"

任务：
1. 先判断这个查询是否已经足够具体、覆盖了检索时的核心关键词。
2. 如果已经具体、没有歧义：返回 JSON 数组 []（空数组，不做改写）。
3. 如果存在模糊/术语不专业/可以换角度表达：输出 1~2 条同义改写，每条必须：
   - 仍然和原查询意图一致，不能引入新的主题；
   - 补足原句中可能被忽略的领域关键词（例如原句是"工资"，补成"实发工资 工资条"）；
   - 长度 ≤ 40 个汉字；
   - 彼此不重复、也不与原句完全相同。

只输出 JSON 数组字符串。"""


def _llm_based_rewrites(query: str, *, rewrite_count: int) -> list[str]:
    """走 LLM 改写。失败或输出异常时返回空列表（fail-open）。"""
    if chat_model is None:
        return []
    try:
        resp = chat_model.invoke(
            _REWRITE_PROMPT_TEMPLATE.format(original_query=query)
        )
        raw_text = ""
        if isinstance(resp, str):
            raw_text = resp.strip()
        else:
            raw_text = (getattr(resp, "content", "") or "").strip()
        if not raw_text:
            return []

        # 容错提取：把 ```json ... ``` 包的代码块内容先剥掉
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text)
        if m:
            raw_text = m.group(1).strip()
        # 再找数组边界：找第一个 [ 和最后一个 ]
        lb = raw_text.find("[")
        rb = raw_text.rfind("]")
        if lb < 0 or rb < 0 or rb <= lb:
            return []
        arr = json.loads(raw_text[lb : rb + 1])
        if not isinstance(arr, list):
            return []
        out: list[str] = []
        for item in arr:
            if isinstance(item, str) and item.strip():
                s = item.strip()
                if s != query and len(s) <= 60:
                    out.append(s)
        # 最多保留 rewrite_count 条
        return out[:rewrite_count]
    except Exception as exc:
        logger.warning(
            f"[QueryRewrite] LLM 改写失败（fail-open）：query={query[:40]}! 错误={exc}"
        )
        return []


class QueryRewriter:
    """Query 改写器。"""

    def __init__(
        self,
        *,
        enabled: Optional[bool] = None,
        max_rewrites: Optional[int] = None,
        min_chars_for_llm: Optional[int] = None,
        llm_only_for_long: Optional[bool] = None,
    ) -> None:
        cfg = rag_conf.get("query_rewrite", {}) or {}
        self.enabled: bool = bool(
            enabled if enabled is not None else cfg.get("enabled", True)
        )
        self.max_rewrites: int = int(
            max_rewrites if max_rewrites is not None else cfg.get("max_rewrites", 2)
        )
        if self.max_rewrites < 0:
            self.max_rewrites = 0
        elif self.max_rewrites > 3:
            self.max_rewrites = 3

        # 超过该字符数的 query 才用 LLM 改写，更短的只用规则
        self.min_chars_for_llm: int = int(
            min_chars_for_llm
            if min_chars_for_llm is not None
            else cfg.get("min_chars_for_llm", 5)
        )
        self.llm_only_for_long: bool = bool(
            llm_only_for_long
            if llm_only_for_long is not None
            else cfg.get("llm_only_for_long", True)
        )

    def rewrite(self, query: str) -> QueryRewriteResult:
        """对用户查询做改写；始终返回 QueryRewriteResult。"""
        original = (query or "").strip()
        if not original:
            return QueryRewriteResult(original=original, rewrites=[])

        if not self.enabled or self.max_rewrites == 0:
            return QueryRewriteResult(original=original, rewrites=[])

        need_rewrite = _query_is_short_or_ambiguous(original)
        # 企业假期类关键词：即使句子不短也做规则扩写，避免只召回工资/财报噪声
        force_policy_expand = any(
            kw in original for kw in ("年假", "请假", "事假", "病假", "调休", "休假")
        )
        if not need_rewrite and not force_policy_expand:
            return QueryRewriteResult(original=original, rewrites=[])

        # 1) 先规则式：短 query 通常规则就够了，省 LLM 调用
        rule_rw = _rule_based_rewrites(original)
        result_rewrites: list[str] = []
        if rule_rw:
            result_rewrites.extend(rule_rw[: self.max_rewrites])

        remaining = self.max_rewrites - len(result_rewrites)
        use_llm = (
            remaining > 0
            and not force_policy_expand  # 假期类优先规则，少打 LLM
            and (len(original) >= self.min_chars_for_llm or not self.llm_only_for_long)
        )
        if use_llm:
            llm_rw = _llm_based_rewrites(original, rewrite_count=remaining)
            # 去重：规则式产出的不要在 LLM 结果里重复
            exist = set(result_rewrites)
            for s in llm_rw:
                if s and s not in exist and s != original:
                    result_rewrites.append(s)
                    exist.add(s)
                    if len(result_rewrites) >= self.max_rewrites:
                        break

        return QueryRewriteResult(original=original, rewrites=result_rewrites)
