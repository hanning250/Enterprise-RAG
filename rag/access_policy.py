from __future__ import annotations

"""检索层访问控制（ACL）策略：AccessScope 表示用户的可见范围，DocumentAclPolicy 把文档 metadata 与用户 scope 匹配，过滤掉未授权候选。"""

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from langchain_core.documents import Document

from rag.types import RetrievalCandidate
from utils.logger_handler import logger


SENSITIVITY_PUBLIC = "public"
SENSITIVITY_INTERNAL = "internal"
SENSITIVITY_CONFIDENTIAL = "confidential"
SENSITIVITY_RESTRICTED = "restricted"

DOC_TYPE_SALARY = "salary"
DOC_TYPE_FINANCE = "finance"
DOC_TYPE_POLICY = "policy"
DOC_TYPE_REPORT = "report"
DOC_TYPE_GENERAL = "general"

DEFAULT_ROLE_SENSITIVITY_MAP: dict[str, list[str]] = {
    "admin": [SENSITIVITY_PUBLIC, SENSITIVITY_INTERNAL, SENSITIVITY_CONFIDENTIAL, SENSITIVITY_RESTRICTED],
    "hr_admin": [SENSITIVITY_PUBLIC, SENSITIVITY_INTERNAL, SENSITIVITY_CONFIDENTIAL],
    "finance_admin": [SENSITIVITY_PUBLIC, SENSITIVITY_INTERNAL, SENSITIVITY_CONFIDENTIAL],
    "manager": [SENSITIVITY_PUBLIC, SENSITIVITY_INTERNAL, SENSITIVITY_CONFIDENTIAL],
    "employee": [SENSITIVITY_PUBLIC, SENSITIVITY_INTERNAL],
}


@dataclass
class AccessScope:
    """用户访问范围，来自 RequestContext / IdentityInfo。"""
    user_id: str
    user_name: str = ""
    department: str = ""
    roles: list[str] = field(default_factory=list)
    data_scope: str = "self"

    def can_see_company_wide(self) -> bool:
        """data_scope == 'company' 或 角色是 admin/hr_admin/finance_admin 之一。

        manager 仅表示管理岗，不自动获得全公司薪资可见；跨人/跨部门需显式
        data_scope=company 或 HR/财务/管理员角色。
        """
        if self.data_scope == "company":
            return True
        admin_like = {"admin", "hr_admin", "finance_admin"}
        return bool(admin_like & set(self.roles))

    def allowed_sensitivities(self) -> set[str]:
        """根据 roles 看允许的 sensitivity 级别集合；没匹配到就只给 public+internal。"""
        result: set[str] = set()
        role_set = set(self.roles)
        hit = False
        for role, levels in DEFAULT_ROLE_SENSITIVITY_MAP.items():
            if role in role_set:
                hit = True
                result.update(levels)
        if not hit:
            result = {SENSITIVITY_PUBLIC, SENSITIVITY_INTERNAL}
        return result

    def matches_owner(self, meta: dict[str, Any]) -> bool:
        """判断文档 owner_role / owner_name / employee_name 等字段是否匹配本人。
        用于工资类：只有本人能看（除非 scope 能 company-wide 看）。
        """
        if self.can_see_company_wide():
            return True
        doc_owner = str(meta.get("owner_name") or meta.get("employee_name") or "").strip()
        doc_owner_id = str(meta.get("owner_id") or meta.get("employee_id") or "").strip()
        my_name = self.user_name.strip()
        my_id = self.user_id.strip()
        if my_id and doc_owner_id and my_id == doc_owner_id:
            return True
        if my_name and doc_owner and my_name == doc_owner:
            return True
        return False

    def matches_department(self, meta: dict[str, Any]) -> bool:
        """部门匹配：scope.data_scope=company → True；=department 且 doc部门 == self.department → True；=self 只允许文档没写部门或部门匹配。"""
        if self.data_scope == "company":
            return True
        doc_dep = str(meta.get("department") or meta.get("employee_department") or "").strip()
        if not doc_dep:
            return True
        my_dep = self.department.strip()
        if self.data_scope == "department":
            return bool(my_dep and doc_dep == my_dep)
        if not my_dep:
            return not doc_dep
        return doc_dep == my_dep


def _doc_meta(doc: Document | RetrievalCandidate) -> dict[str, Any]:
    """从 Document 或 RetrievalCandidate 中统一提取 metadata。"""
    if isinstance(doc, RetrievalCandidate):
        return dict(doc.doc.metadata or {})
    return dict(doc.metadata or {})


def _doc_sensitivity(meta: dict[str, Any]) -> str:
    """提取文档 sensitivity_level；缺失时根据 doc_type 做默认映射。"""
    s = str(meta.get("sensitivity_level") or "").strip().lower()
    if s in {SENSITIVITY_PUBLIC, SENSITIVITY_INTERNAL, SENSITIVITY_CONFIDENTIAL, SENSITIVITY_RESTRICTED}:
        return s
    dt = str(meta.get("doc_type") or "").strip().lower()
    if dt in {DOC_TYPE_SALARY, DOC_TYPE_FINANCE}:
        return SENSITIVITY_CONFIDENTIAL
    if dt in {DOC_TYPE_POLICY, DOC_TYPE_REPORT}:
        return SENSITIVITY_INTERNAL
    return SENSITIVITY_PUBLIC


def _doc_type(meta: dict[str, Any]) -> str:
    """提取文档 doc_type；缺失时根据 source/source_file 文件名推断。"""
    dt = str(meta.get("doc_type") or "").strip().lower()
    if dt:
        return dt
    source_lower = str(meta.get("source") or meta.get("source_file") or "").lower()
    if "工资" in source_lower or "salary" in source_lower or "xlsx" in source_lower:
        return DOC_TYPE_SALARY
    if "财务" in source_lower or "财报" in source_lower or "finance" in source_lower:
        return DOC_TYPE_FINANCE
    if "制度" in source_lower or "规章" in source_lower or "手册" in source_lower or "知识库" in source_lower:
        return DOC_TYPE_POLICY
    if "报告" in source_lower or "report" in source_lower:
        return DOC_TYPE_REPORT
    return DOC_TYPE_GENERAL


def is_doc_accessible(scope: AccessScope, doc: Document | RetrievalCandidate) -> tuple[bool, str]:
    """单文档权限判断。返回 (允许, 原因字符串)。

    工资文档：
    1) 必须本人或公司级授权角色（HR/财务/管理员 / data_scope=company）
    2) 本人查看自己的薪资时，允许 confidential（不受 employee 敏感级上限限制）
    3) 公司级授权仍需通过敏感级与部门校验

    其他文档：敏感级 → 部门匹配。
    """
    meta = _doc_meta(doc)
    sens = _doc_sensitivity(meta)
    dt = _doc_type(meta)
    allowed_sens = scope.allowed_sensitivities()

    if dt == DOC_TYPE_SALARY:
        if not scope.matches_owner(meta):
            return False, "工资文档仅本人或HR/财务/管理员可见"
        # 公司级授权：仍遵守敏感级 + 部门
        if scope.can_see_company_wide():
            if sens not in allowed_sens:
                return False, f"sensitivity={sens} 不在允许集合 {sorted(allowed_sens)}"
            if not scope.matches_department(meta):
                return False, (
                    f"文档部门不匹配 (doc={meta.get('department') or ''}, "
                    f"scope={scope.department}, data_scope={scope.data_scope})"
                )
            return True, "ok"
        # 本人自查：owner 已匹配即可（部门字段可能与档案不一致，不以部门误拦本人）
        return True, "ok"

    if sens not in allowed_sens:
        return False, f"sensitivity={sens} 不在允许集合 {sorted(allowed_sens)}"
    if not scope.matches_department(meta):
        return False, (
            f"文档部门不匹配 (doc={meta.get('department') or ''}, "
            f"scope={scope.department}, data_scope={scope.data_scope})"
        )
    return True, "ok"


def filter_candidates_by_acl(scope: Optional[AccessScope], candidates: Iterable[RetrievalCandidate]) -> tuple[list[RetrievalCandidate], int, list[str]]:
    """对 RetrievalCandidate 列表做 ACL 过滤。
    返回 (通过的候选, 被过滤数, 被过滤chunk_id列表)。
    scope 为 None 时视为 company 级匿名（只看 public/internal），打 warning。
    """
    if scope is None:
        logger.warning("[ACL] scope=None，使用受限匿名 scope（仅 public+internal）")
        scope = AccessScope(user_id="anonymous", roles=["employee"], data_scope="company")
    passed: list[RetrievalCandidate] = []
    blocked_ids: list[str] = []
    blocked_count = 0
    for cand in candidates:
        ok, reason = is_doc_accessible(scope, cand)
        if ok:
            passed.append(cand)
        else:
            blocked_count += 1
            cid = str(_doc_meta(cand).get("chunk_id") or "")
            if cid:
                blocked_ids.append(cid)
            logger.debug(f"[ACL] 过滤掉候选: chunk_id={cid or '?'} 原因={reason}")
    if blocked_count:
        logger.info(f"[ACL] {len(passed)}/{len(list(candidates))} 候选通过，{blocked_count} 条被过滤")
    return passed, blocked_count, blocked_ids


def filter_documents_by_acl(scope: Optional[AccessScope], docs: Iterable[Document]) -> tuple[list[Document], int, list[str]]:
    """对 LangChain Document 列表做 ACL 过滤（接口与上面一致，供上层 QueryService 最后再做一次强校验）。"""
    if scope is None:
        logger.warning("[ACL] scope=None，使用受限匿名 scope（仅 public+internal）")
        scope = AccessScope(user_id="anonymous", roles=["employee"], data_scope="company")
    passed: list[Document] = []
    blocked_ids: list[str] = []
    blocked_count = 0
    for d in docs:
        ok, reason = is_doc_accessible(scope, d)
        if ok:
            passed.append(d)
        else:
            blocked_count += 1
            cid = str(_doc_meta(d).get("chunk_id") or "")
            if cid:
                blocked_ids.append(cid)
            logger.debug(f"[ACL] 过滤掉Document: chunk_id={cid or '?'} 原因={reason}")
    return passed, blocked_count, blocked_ids
