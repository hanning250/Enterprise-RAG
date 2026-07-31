"""答案组织与格式策略（表达层）。原提示词里"工资/财报/制度"三种回答格式、拒答措辞模板收敛成代码函数，不再在超长提示词里堆格式。权限/禁止类规则不在这里——归到 AccessScope 与接口层。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

from langchain_core.documents import Document


REFUSAL_INSUFFICIENT_DATA = "资料不足，无法基于当前知识库回答该问题。"
REFUSAL_NO_PERMISSION = "抱歉，您没有权限访问该信息。如有疑问请联系 HR 或系统管理员。"
REFUSAL_CROSS_DEPARTMENT_SALARY = "抱歉，您只能查询本人工资信息。请使用本人账号登录或联系 HR。"
DISCLAIMER_HR_CONSULT = "（注：以上内容仅供参考，具体执行以公司 HR 部门正式文件为准）"
DISCLAIMER_FINANCE_CONSULT = "（注：财务数据来自历史报表，最新情况以财务部口径为准）"


@dataclass
class AnswerPolicyConfig:
    add_disclaimer: bool = True
    prefer_bullets: bool = True
    show_source: bool = False


def _infer_doc_type_from_meta(meta: dict[str, Any]) -> str:
    dt = str(meta.get("doc_type") or "").strip().lower()
    if dt:
        return dt
    source_lower = str(meta.get("source") or meta.get("source_file") or meta.get("file_name") or "").lower()
    if "工资" in source_lower or "salary" in source_lower:
        return "salary"
    if "财务" in source_lower or "财报" in source_lower or "finance" in source_lower:
        return "finance"
    if "制度" in source_lower or "规章" in source_lower or "手册" in source_lower or "知识库" in source_lower:
        return "policy"
    return "general"


def detect_answer_category(docs: list[Document]) -> str:
    if not docs:
        return "general"
    counts: dict[str, int] = {"salary": 0, "finance": 0, "policy": 0, "general": 0}
    for d in docs:
        meta = dict(d.metadata or {})
        dt = _infer_doc_type_from_meta(meta)
        if dt in counts:
            counts[dt] += 1
        else:
            counts["general"] += 1
    best = "general"
    best_count = 0
    for cat in ["salary", "finance", "policy", "general"]:
        if counts[cat] > best_count:
            best_count = counts[cat]
            best = cat
    return best


def refusal_or_none(
    scope: Optional[Any],
    docs: list[Document],
    *,
    category: str | None = None,
    acl_blocked_count: int = 0,
) -> Optional[str]:
    if category is None:
        category = detect_answer_category(docs)

    if not docs:
        if acl_blocked_count > 0:
            return REFUSAL_NO_PERMISSION
        return REFUSAL_INSUFFICIENT_DATA

    if scope is not None and category == "salary":
        matched = 0
        unmatched = 0
        has_matches_owner = callable(getattr(scope, "matches_owner", None))
        for d in docs:
            meta = dict(d.metadata or {})
            dt = _infer_doc_type_from_meta(meta)
            if dt != "salary":
                continue
            if has_matches_owner and scope.matches_owner(meta):
                matched += 1
            else:
                unmatched += 1
        if unmatched > matched and (matched + unmatched) > 0:
            return REFUSAL_CROSS_DEPARTMENT_SALARY

    return None


def post_process_answer(
    answer: str,
    *,
    docs: list[Document],
    cfg: AnswerPolicyConfig | None = None,
    category: str | None = None,
) -> str:
    if cfg is None:
        cfg = AnswerPolicyConfig()
    if category is None:
        category = detect_answer_category(docs)

    result = answer or ""
    stripped = result.strip()

    if not stripped or len(stripped) < 6:
        return REFUSAL_INSUFFICIENT_DATA

    # 统一拒答措辞，且拒答不加免责声明
    if "资料不足" in stripped:
        return REFUSAL_INSUFFICIENT_DATA
    if REFUSAL_NO_PERMISSION in stripped or stripped.startswith("抱歉，您没有权限"):
        return stripped if REFUSAL_NO_PERMISSION in stripped else REFUSAL_NO_PERMISSION

    if cfg.add_disclaimer:
        disclaimer = None
        if category == "salary":
            disclaimer = DISCLAIMER_HR_CONSULT
        elif category == "finance":
            disclaimer = DISCLAIMER_FINANCE_CONSULT
        elif category == "policy":
            disclaimer = DISCLAIMER_HR_CONSULT
        if disclaimer and disclaimer not in result:
            result = result.rstrip() + "\n\n" + disclaimer

    if cfg.show_source:
        sources: list[str] = []
        seen: set[str] = set()
        for d in docs:
            meta = dict(d.metadata or {})
            src = (
                str(meta.get("file_name") or meta.get("source") or meta.get("source_file") or "").strip()
            )
            if src and src not in seen:
                seen.add(src)
                sources.append(src)
        if sources:
            result = result.rstrip() + "\n资料来源：" + "、".join(sources)

    return result.strip()
