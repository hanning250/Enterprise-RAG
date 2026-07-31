"""身份认证与权限控制模块。

提供请求上下文解析、身份配置加载、角色校验等功能，
不依赖 FastAPI/Starlette，可被非 Web 进程安全导入。
"""

from __future__ import annotations

import os
import uuid
import hmac
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import yaml

from utils.logger_handler import logger
from utils.path_tool import get_abs_path


DataScope = Literal["self", "department", "company"]

_HEADER_USER_ID = "X-USER-ID"
_HEADER_USER_NAME = "X-USER-NAME"
_HEADER_DEPARTMENT = "X-DEPARTMENT"
_HEADER_ROLES = "X-ROLES"
_HEADER_DATA_SCOPE = "X-DATA-SCOPE"
_HEADER_REQUEST_ID = "X-REQUEST-ID"
_HEADER_CLIENT_IP = "X-CLIENT-IP"
_HEADER_INTERNAL_AUTH = "X-INTERNAL-AUTH"


@dataclass
class RequestContext:
    """请求身份上下文，封装单次请求的用户、角色、数据权限等信息。"""

    user_id: str
    user_name: str = ""
    department: str = ""
    roles: list[str] = field(default_factory=list)
    data_scope: DataScope = "self"
    request_id: str = ""
    client_ip: str = ""

    def to_dict(self) -> dict:
        """将上下文序列化为字典，便于日志或跨进程传递。"""
        return {
            "user_id": self.user_id,
            "user_name": self.user_name,
            "department": self.department,
            "roles": list(self.roles),
            "data_scope": self.data_scope,
            "request_id": self.request_id,
            "client_ip": self.client_ip,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RequestContext":
        """从字典反序列化为 RequestContext 实例。"""
        return cls(
            user_id=str(data.get("user_id", "")),
            user_name=str(data.get("user_name", "")),
            department=str(data.get("department", "")),
            roles=list(data.get("roles") or []),
            data_scope=data.get("data_scope", "self"),
            request_id=str(data.get("request_id", "")),
            client_ip=str(data.get("client_ip", "")),
        )


@dataclass
class IdentityConfig:
    """身份系统全局配置。

    从环境变量与 config/auth.yml 合并加载，优先顺序：
    环境变量 > auth.yml > 代码默认值。
    """

    require_auth: bool = True
    cors_allow_origins: list[str] = field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://localhost:3000",
            "http://127.0.0.1:8765",
        ]
    )
    admin_roles: list[str] = field(
        default_factory=lambda: ["admin", "hr_admin", "finance_admin"]
    )
    build_endpoint_roles: list[str] = field(default_factory=lambda: ["admin"])
    trusted_identity_secret: str = ""
    allow_untrusted_identity_headers: bool = False


_identity_config_singleton: Optional[IdentityConfig] = None


def load_identity_config() -> IdentityConfig:
    """单例加载 IdentityConfig。

    首次调用时读取环境变量与 ``config/auth.yml``，后续调用返回同一实例。
    """
    global _identity_config_singleton
    if _identity_config_singleton is not None:
        return _identity_config_singleton

    cfg = IdentityConfig()

    env_require_auth = os.environ.get("AUTH_REQUIRE_AUTH")
    if env_require_auth is not None:
        cfg.require_auth = env_require_auth.strip().lower() not in (
            "false",
            "0",
            "no",
            "",
        )

    env_cors = os.environ.get("AUTH_CORS_ALLOW")
    if env_cors is not None:
        origins = [o.strip() for o in env_cors.split(",") if o.strip()]
        if origins:
            cfg.cors_allow_origins = origins

    env_admin_roles = os.environ.get("AUTH_ADMIN_ROLES")
    if env_admin_roles is not None:
        roles = [r.strip() for r in env_admin_roles.split(",") if r.strip()]
        if roles:
            cfg.admin_roles = roles

    env_build_roles = os.environ.get("AUTH_BUILD_ENDPOINT_ROLES")
    if env_build_roles is not None:
        roles = [r.strip() for r in env_build_roles.split(",") if r.strip()]
        if roles:
            cfg.build_endpoint_roles = roles

    env_trusted_identity_secret = os.environ.get("AUTH_TRUSTED_IDENTITY_SECRET")
    if env_trusted_identity_secret is not None:
        cfg.trusted_identity_secret = env_trusted_identity_secret.strip()

    env_allow_untrusted = os.environ.get("AUTH_ALLOW_UNTRUSTED_IDENTITY_HEADERS")
    if env_allow_untrusted is not None:
        cfg.allow_untrusted_identity_headers = env_allow_untrusted.strip().lower() in (
            "true",
            "1",
            "yes",
        )

    auth_yml_path = get_abs_path("config/auth.yml")
    if Path(auth_yml_path).exists():
        try:
            with open(auth_yml_path, "r", encoding="utf-8") as f:
                yml_data = yaml.load(f, Loader=yaml.FullLoader) or {}
            if isinstance(yml_data, dict):
                if "require_auth" in yml_data and env_require_auth is None:
                    cfg.require_auth = bool(yml_data["require_auth"])
                if "cors_allow_origins" in yml_data and env_cors is None:
                    val = yml_data["cors_allow_origins"]
                    if isinstance(val, list) and val:
                        cfg.cors_allow_origins = [str(x) for x in val]
                if "admin_roles" in yml_data and env_admin_roles is None:
                    val = yml_data["admin_roles"]
                    if isinstance(val, list) and val:
                        cfg.admin_roles = [str(x) for x in val]
                if "build_endpoint_roles" in yml_data and env_build_roles is None:
                    val = yml_data["build_endpoint_roles"]
                    if isinstance(val, list) and val:
                        cfg.build_endpoint_roles = [str(x) for x in val]
                if "trusted_identity_secret" in yml_data and env_trusted_identity_secret is None:
                    cfg.trusted_identity_secret = str(yml_data["trusted_identity_secret"] or "").strip()
                if "allow_untrusted_identity_headers" in yml_data and env_allow_untrusted is None:
                    cfg.allow_untrusted_identity_headers = bool(yml_data["allow_untrusted_identity_headers"])
            logger.info(f"已加载身份配置文件：{auth_yml_path}")
        except Exception as exc:
            logger.warning(f"读取 auth.yml 失败，使用默认配置：{exc}")
    else:
        logger.info("未找到 config/auth.yml，使用环境变量 + 默认身份配置")

    if not cfg.require_auth:
        logger.warning("AUTH_REQUIRE_AUTH=False，身份校验已关闭，将使用匿名上下文")
    elif not cfg.trusted_identity_secret and not cfg.allow_untrusted_identity_headers:
        logger.warning(
            "未配置 AUTH_TRUSTED_IDENTITY_SECRET，且未开启 "
            "AUTH_ALLOW_UNTRUSTED_IDENTITY_HEADERS；业务接口将拒绝信任客户端身份头。"
        )
    elif cfg.allow_untrusted_identity_headers:
        logger.warning(
            "AUTH_ALLOW_UNTRUSTED_IDENTITY_HEADERS=True，仅适合本地开发；"
            "生产环境必须通过可信网关注入身份并配置 AUTH_TRUSTED_IDENTITY_SECRET。"
        )

    _identity_config_singleton = cfg
    return cfg


def _normalize_headers(headers: dict) -> dict[str, str]:
    """将 headers 统一为 {大写KEY: 字符串值} 形式，忽略非字符串键。"""
    result: dict[str, str] = {}
    for k, v in headers.items():
        if v is None:
            continue
        try:
            key_upper = str(k).strip().upper()
        except Exception:
            continue
        if not key_upper:
            continue
        result[key_upper] = str(v).strip()
    return result


def _short_request_id() -> str:
    """生成 12 字符的短 hex request_id。"""
    return uuid.uuid4().hex[:12]


def is_trusted_identity_request(headers: dict, cfg: Optional[IdentityConfig] = None) -> bool:
    """判断请求中的身份字段是否来自可信内部网关。"""
    cfg = cfg or load_identity_config()
    if not cfg.require_auth:
        return True
    if cfg.allow_untrusted_identity_headers:
        return True
    if not cfg.trusted_identity_secret:
        return False

    norm = _normalize_headers(headers or {})
    supplied = norm.get(_HEADER_INTERNAL_AUTH, "")
    if supplied.lower().startswith("bearer "):
        supplied = supplied[7:].strip()
    return hmac.compare_digest(supplied, cfg.trusted_identity_secret)


def _safe_data_scope(raw: str) -> DataScope:
    """校验并转换 data_scope，非法值回退为 self。"""
    if raw in ("self", "department", "company"):
        return raw  # type: ignore[return-value]
    if raw:
        logger.warning(f"非法 data_scope={raw!r}，回退为 'self'")
    return "self"


def parse_identity_from_headers(headers: dict) -> RequestContext | None:
    """从请求头解析身份信息并构造 RequestContext。

    支持的 Header（大小写不敏感）：
    X-User-Id, X-User-Name, X-Department, X-Roles(逗号分隔),
    X-Data-Scope, X-Request-Id, X-Client-Ip。

    - 当 config.require_auth=True 且缺失 X-User-Id 时返回 None。
    - 当 config.require_auth=False 时始终返回匿名上下文，跳过校验。
    """
    cfg = load_identity_config()
    norm = _normalize_headers(headers or {})

    if not cfg.require_auth:
        return RequestContext(
            user_id="anonymous",
            user_name="",
            department="",
            roles=[],
            data_scope="company",
            request_id=norm.get(_HEADER_REQUEST_ID) or _short_request_id(),
            client_ip=norm.get(_HEADER_CLIENT_IP, ""),
        )

    if not is_trusted_identity_request(norm, cfg):
        logger.warning("身份头不是可信网关注入，拒绝解析 X-User-* / X-Roles")
        return None

    user_id = norm.get(_HEADER_USER_ID, "")
    if not user_id:
        logger.debug("缺少 X-User-Id header，身份解析失败")
        return None

    raw_roles = norm.get(_HEADER_ROLES, "")
    roles = [r.strip() for r in raw_roles.split(",") if r.strip()] if raw_roles else []

    raw_scope = norm.get(_HEADER_DATA_SCOPE, "")
    data_scope = _safe_data_scope(raw_scope)

    request_id = norm.get(_HEADER_REQUEST_ID) or _short_request_id()

    return RequestContext(
        user_id=user_id,
        user_name=norm.get(_HEADER_USER_NAME, ""),
        department=norm.get(_HEADER_DEPARTMENT, ""),
        roles=roles,
        data_scope=data_scope,
        request_id=request_id,
        client_ip=norm.get(_HEADER_CLIENT_IP, ""),
    )


def require_request_context(ctx: Optional[RequestContext]) -> RequestContext:
    """校验 RequestContext 是否存在，不存在则抛出 PermissionError。

    以纯函数形式实现，便于在 FastAPI Depends 或普通代码中复用。
    """
    if ctx is None:
        raise PermissionError("缺少身份上下文")
    return ctx


def has_any_role(ctx: RequestContext, allowed_roles: list[str]) -> bool:
    """通用角色匹配：用户持有任一 allowed_roles 即返回 True。"""
    if not allowed_roles:
        return True
    user_roles = ctx.roles or []
    for r in allowed_roles:
        if r in user_roles:
            return True
    return False


def is_authorized_for_build(ctx: RequestContext, cfg: IdentityConfig) -> bool:
    """判断当前用户是否有权限调用 /api/rag/build 接口。"""
    return has_any_role(ctx, cfg.build_endpoint_roles)
