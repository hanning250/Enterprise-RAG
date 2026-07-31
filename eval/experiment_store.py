"""RAG 评估实验参数版本化存储 + 历史对比查询。

设计目标：
1. **零外部依赖**：不用 SQLite / 数据库，本地 JSON 文件持久化。
   路径：eval/reports/_experiment_registry.json
2. **参数指纹**：所有检索/RAG/生成参数 → hash 成唯一 run_id，
   方便同一套配置重复运行，结果可以对比均值/方差。
3. **结果聚合**：对同一个 params_hash 的多次 run，可以自动聚合
   avg/std/max/min，便于判断效果提升是否显著。
4. **与两个现有脚本无缝集成**：
   - eval/run_retrieval_eval.py → 写 retrieval experiments
   - eval/run_generation_eval.py → 写 generation experiments
5. **向后兼容**：旧版本脚本产生的报告文件不会被覆盖，
   只是多一条 registry 记录。
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from utils.path_tool import get_abs_path

REGISTRY_PATH = "eval/reports/_experiment_registry.json"
MAX_RUNS_PER_EXPERIMENT = 50  # 同一 params_hash 最多保留最近 N 次 run，避免文件无限增长
MAX_TOTAL_EXPERIMENTS = 500  # 最多保留最近 500 个不同参数的实验


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _registry_abs_path() -> Path:
    return Path(get_abs_path(REGISTRY_PATH))


def _load_registry_raw() -> dict[str, Any]:
    path = _registry_abs_path()
    if not path.exists():
        return {"version": 1, "experiments": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"version": 1, "experiments": {}}
        data.setdefault("version", 1)
        data.setdefault("experiments", {})
        return data
    except Exception:
        # 损坏就新建，不要影响评估脚本运行
        return {"version": 1, "experiments": {}}


def _save_registry_raw(data: dict[str, Any]) -> None:
    path = _registry_abs_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # 先写 tmp 再 rename，避免中途写入损坏
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _canonical_config_dict(cfg: dict[str, Any]) -> dict[str, Any]:
    """把参数 dict 做 canononical 化（键排序 + 字符串转值标准化），
    确保相同参数不管字典传入顺序如何，都能得到相同 hash。"""
    if not isinstance(cfg, dict):
        return {"_value": cfg}

    def _norm(v: Any) -> Any:
        if isinstance(v, dict):
            return {k: _norm(v[k]) for k in sorted(v.keys())}
        if isinstance(v, list):
            return [_norm(x) for x in v]
        if isinstance(v, tuple):
            return [_norm(x) for x in v]
        # 浮点数：保留 6 位有效数字，避免 0.3500000001 和 0.35 被当成不同参数
        if isinstance(v, float):
            return round(v, 6)
        if isinstance(v, bool):
            return bool(v)
        if isinstance(v, int):
            return int(v)
        return v

    return _norm(cfg)


def compute_params_hash(params: dict[str, Any]) -> str:
    """计算参数指纹（前 12 位 MD5，足够做对比且不冗长）。"""
    canon = _canonical_config_dict(params)
    payload = json.dumps(canon, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]


@dataclass
class ExperimentRun:
    """一次具体的评估运行记录。"""

    run_id: str
    run_at: str  # ISO 8601
    duration_sec: float = 0.0
    sample_count: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    report_files: dict[str, str] = field(default_factory=dict)  # e.g. {"csv": "...", "md": "..."}
    extra: dict[str, Any] = field(default_factory=dict)  # 任何补充信息（失败、二检次数等）

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentRecord:
    """一个参数版本（同一 params_hash 多次运行的聚合）。"""

    params_hash: str
    experiment_type: str  # "retrieval" | "generation"
    params: dict[str, Any]
    first_run_at: str
    last_run_at: str
    runs: list[ExperimentRun] = field(default_factory=list)
    aggregated_metrics: dict[str, dict[str, float]] = field(default_factory=dict)  # {metric: {avg, std, min, max, count}}
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "params_hash": self.params_hash,
            "experiment_type": self.experiment_type,
            "params": self.params,
            "first_run_at": self.first_run_at,
            "last_run_at": self.last_run_at,
            "runs": [r.to_dict() for r in self.runs],
            "aggregated_metrics": self.aggregated_metrics,
            "tags": self.tags,
            "notes": self.notes,
        }


def _recompute_aggregated(record: ExperimentRecord) -> None:
    """基于 runs 重新计算每个 metric 的 avg/std/min/max/count。"""
    from math import sqrt

    metric_values: dict[str, list[float]] = {}
    for r in record.runs:
        for k, v in (r.metrics or {}).items():
            if isinstance(v, (int, float)):
                metric_values.setdefault(k, []).append(float(v))

    agg: dict[str, dict[str, float]] = {}
    for k, vals in metric_values.items():
        n = len(vals)
        if n == 0:
            continue
        mean = sum(vals) / n
        if n >= 2:
            var = sum((x - mean) ** 2 for x in vals) / (n - 1)  # 样本方差
            std = sqrt(var)
        else:
            std = 0.0
        agg[k] = {
            "avg": round(mean, 6),
            "std": round(std, 6),
            "min": round(min(vals), 6),
            "max": round(max(vals), 6),
            "count": n,
        }
    record.aggregated_metrics = agg


def _collect_env_meta() -> dict[str, Any]:
    return {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.platform(),
    }


def register_experiment_run(
    *,
    experiment_type: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    report_files: Optional[dict[str, str]] = None,
    duration_sec: float = 0.0,
    sample_count: int = 0,
    tags: Optional[list[str]] = None,
    notes: str = "",
    extra: Optional[dict[str, Any]] = None,
) -> tuple[str, str]:
    """写入一次评估运行记录，返回 (params_hash, run_id)。"""
    if experiment_type not in ("retrieval", "generation"):
        # 兼容未来扩展
        pass

    params_full: dict[str, Any] = {
        "experiment_type": experiment_type,
        "params": dict(params or {}),
        "env": _collect_env_meta(),
    }
    params_hash = compute_params_hash(params_full)

    now_iso = datetime.now().isoformat(timespec="seconds")
    run_id = f"{params_hash}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    run = ExperimentRun(
        run_id=run_id,
        run_at=now_iso,
        duration_sec=round(float(duration_sec), 3),
        sample_count=int(sample_count),
        metrics={k: round(float(v), 6) if isinstance(v, (int, float)) else v for k, v in (metrics or {}).items()},
        report_files=dict(report_files or {}),
        extra=dict(extra or {}),
    )

    registry = _load_registry_raw()
    experiments: dict[str, Any] = registry["experiments"]

    if params_hash in experiments:
        raw = experiments[params_hash]
        # 恢复为 ExperimentRecord
        runs = [ExperimentRun(**r) for r in raw.get("runs", [])]
        record = ExperimentRecord(
            params_hash=raw.get("params_hash", params_hash),
            experiment_type=raw.get("experiment_type", experiment_type),
            params=raw.get("params", params_full["params"]),
            first_run_at=raw.get("first_run_at", now_iso),
            last_run_at=now_iso,
            runs=runs,
            aggregated_metrics=raw.get("aggregated_metrics", {}),
            tags=list(raw.get("tags", [])),
            notes=str(raw.get("notes", "")),
        )
    else:
        record = ExperimentRecord(
            params_hash=params_hash,
            experiment_type=experiment_type,
            params=params_full["params"],
            first_run_at=now_iso,
            last_run_at=now_iso,
            runs=[],
            aggregated_metrics={},
            tags=list(tags or []),
            notes=notes,
        )

    record.last_run_at = now_iso
    if tags:
        for t in tags:
            if t and t not in record.tags:
                record.tags.append(t)
    if notes and not record.notes:
        record.notes = notes

    record.runs.append(run)
    # 裁剪旧 runs
    if len(record.runs) > MAX_RUNS_PER_EXPERIMENT:
        record.runs = record.runs[-MAX_RUNS_PER_EXPERIMENT:]

    _recompute_aggregated(record)

    experiments[params_hash] = record.to_dict()

    # 全局最多保留最近 MAX_TOTAL_EXPERIMENTS 个（按 last_run_at 排序）
    if len(experiments) > MAX_TOTAL_EXPERIMENTS:
        items = sorted(
            experiments.items(),
            key=lambda kv: kv[1].get("last_run_at", ""),
            reverse=True,
        )
        experiments = dict(items[:MAX_TOTAL_EXPERIMENTS])
        registry["experiments"] = experiments

    _save_registry_raw(registry)
    return params_hash, run_id


def list_experiments(
    *,
    experiment_type: Optional[str] = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """列出所有实验（默认只看最近 N 个，返回简化摘要）。"""
    registry = _load_registry_raw()
    items = list(registry["experiments"].values())
    if experiment_type:
        items = [x for x in items if x.get("experiment_type") == experiment_type]
    items.sort(key=lambda x: x.get("last_run_at", ""), reverse=True)
    out = []
    for it in items[:limit]:
        summary = {
            "params_hash": it["params_hash"],
            "experiment_type": it.get("experiment_type", "?"),
            "last_run_at": it.get("last_run_at", ""),
            "first_run_at": it.get("first_run_at", ""),
            "run_count": len(it.get("runs", [])),
            "tags": it.get("tags", []),
            "notes": it.get("notes", ""),
            "aggregated_metrics": it.get("aggregated_metrics", {}),
            "params": it.get("params", {}),
        }
        out.append(summary)
    return out


def get_experiment(params_hash: str) -> Optional[dict[str, Any]]:
    """按 hash 取单个实验的完整详情（含所有 runs）。"""
    registry = _load_registry_raw()
    return registry["experiments"].get(params_hash)


def compare_experiments(
    params_hashes: list[str],
    *,
    metrics: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """对比多个 params_hash 的聚合指标，返回对比表。"""
    rows = []
    for ph in params_hashes:
        exp = get_experiment(ph)
        if not exp:
            continue
        agg = exp.get("aggregated_metrics", {})
        metric_keys = list(metrics or agg.keys())
        row = {
            "params_hash": ph,
            "experiment_type": exp.get("experiment_type", "?"),
            "run_count": len(exp.get("runs", [])),
            "last_run_at": exp.get("last_run_at", ""),
            "params": exp.get("params", {}),
        }
        for k in metric_keys:
            a = agg.get(k) or {}
            row[f"{k}_avg"] = a.get("avg")
            row[f"{k}_std"] = a.get("std")
            row[f"{k}_min"] = a.get("min")
            row[f"{k}_max"] = a.get("max")
        rows.append(row)
    return rows


def export_all_registry_json(target_path: Optional[str] = None) -> str:
    """把整个 registry 导出到指定路径（默认 eval/reports/_experiment_registry_export.json）。"""
    if target_path:
        out_path = Path(get_abs_path(target_path))
    else:
        out_path = _registry_abs_path().with_name("_experiment_registry_export.json")
    data = _load_registry_raw()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return str(out_path)
