"""进程内 QA 缓存：按「问题 + 身份」隔离，避免串权限答案。"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

from utils.config_handler import rag_conf
from utils.logger_handler import logger


def _qa_cache_config() -> dict[str, Any]:
    cfg = (rag_conf.get("cache") or {}).get("qa") or {}
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "max_entries": max(8, int(cfg.get("max_entries", 256))),
        "ttl_seconds": max(0, int(cfg.get("ttl_seconds", 600))),
    }


@dataclass(frozen=True)
class QaCacheEntry:
    answer: str
    evidence_docs: list[dict[str, Any]]
    trace: dict[str, Any]
    created_at: float


class QaCache:
    """线程安全的 LRU + TTL QA 缓存。"""

    def __init__(
        self,
        *,
        max_entries: int = 256,
        ttl_seconds: int = 600,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._data: OrderedDict[str, QaCacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    @classmethod
    def from_config(cls) -> "QaCache":
        cfg = _qa_cache_config()
        return cls(
            max_entries=cfg["max_entries"],
            ttl_seconds=cfg["ttl_seconds"],
            enabled=cfg["enabled"],
        )

    @staticmethod
    def make_key(
        query: str,
        *,
        user_id: str = "",
        user_name: str = "",
        department: str = "",
        roles: list[str] | None = None,
        data_scope: str = "self",
        summarize: bool = True,
    ) -> str:
        role_part = ",".join(sorted(str(r).strip() for r in (roles or []) if str(r).strip()))
        raw = "|".join(
            [
                (query or "").strip(),
                (user_id or "").strip(),
                (user_name or "").strip(),
                (department or "").strip(),
                role_part,
                (data_scope or "self").strip(),
                "1" if summarize else "0",
            ]
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[QaCacheEntry]:
        if not self.enabled or not key:
            return None
        now = time.monotonic()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.misses += 1
                return None
            if self.ttl_seconds > 0 and (now - entry.created_at) > self.ttl_seconds:
                self._data.pop(key, None)
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return entry

    def put(self, key: str, entry: QaCacheEntry) -> None:
        if not self.enabled or not key:
            return
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = entry
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def clear(self) -> int:
        with self._lock:
            n = len(self._data)
            self._data.clear()
            return n


_qa_cache_singleton: QaCache | None = None
_qa_cache_lock = threading.Lock()


def get_qa_cache() -> QaCache:
    global _qa_cache_singleton
    with _qa_cache_lock:
        if _qa_cache_singleton is None:
            _qa_cache_singleton = QaCache.from_config()
            logger.info(
                f"[QaCache] 初始化：enabled={_qa_cache_singleton.enabled} "
                f"max={_qa_cache_singleton.max_entries} ttl={_qa_cache_singleton.ttl_seconds}s"
            )
        return _qa_cache_singleton


def reset_qa_cache_for_tests() -> None:
    """测试用：重置单例。"""
    global _qa_cache_singleton
    with _qa_cache_lock:
        _qa_cache_singleton = None
