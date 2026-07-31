"""RAG 检索评估脚本（第一阶段：仅检索指标）。

用途：
    读取评估集（eval/datasets/rag_eval.jsonl），逐条执行 **线上同款**
    ``EnterpriseQueryService.retrieve_only`` / ``RagSummarizeService.retriever_docs``
    全链路检索（Hybrid → Expander → Compressor），
    计算 Hit@K / MRR@K / Recall@K，输出 CSV + Markdown 报告。

运行方式：
    python eval/run_retrieval_eval.py
    python eval/run_retrieval_eval.py --dataset eval/datasets/rag_eval.jsonl
    python eval/run_retrieval_eval.py --top-k 5 --ks 1,3,5
    python eval/run_retrieval_eval.py --load-documents
    python eval/run_retrieval_eval.py --mode vector
    python eval/run_retrieval_eval.py --no-rerank
    python eval/run_retrieval_eval.py --vector-top-k 20 --bm25-top-k 20

输出：
    eval/reports/retrieval_eval_YYYYMMDD_HHMMSS.csv    — 详细结果（每条样本的得分）
    eval/reports/retrieval_eval_YYYYMMDD_HHMMSS.md     — 汇总报告（含配置参数和均值）

评估集格式（JSONL）：
    每行一个 JSON 对象，至少包含：
        id                 : 样本唯一标识
        question           : 用户问题
        expected_sources   : 期望命中的来源列表，每个元素含 source_path（可选 page）

指标说明：
    Hit@K    : 前 K 条结果中是否至少命中一条正确来源（0/1，求均值即命中率）
    MRR@K    : 第一条正确来源的排名倒数（排名越靠前分越高）
    Recall@K : 前 K 条结果覆盖了多少比例的期望来源

路径归一化：
    评估集的 source_path 使用相对项目根的路径（如 data/xxx.txt），
    评估时会自动调用 canonicalize_source_path() 转换成和 metadata 一致的格式，
    确保路径匹配不会因为绝对路径/大小写差异而失败。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os  # noqa: E402  noqa: F401

from eval.metrics import (  # noqa: E402
    compute_all_retrieval_metrics,
    average_metrics,
    hit_at_k,
)
from eval.experiment_store import register_experiment_run  # noqa: E402
from rag.document_registry import canonicalize_source_path  # noqa: E402
from rag.rag_service import RagSummarizeService  # noqa: E402
from rag.score_guard import ScoreGuard  # noqa: E402
from utils.config_handler import chroma_conf, rag_conf  # noqa: E402
from utils.path_tool import get_abs_path  # noqa: E402

DEFAULT_DATASET = "eval/datasets/rag_eval.jsonl"
REPORTS_DIR = "eval/reports"
DEFAULT_KS = (1, 3, 5)
SALARY_FILE_PATTERN = "data/员工工资表_2026年{month}月.xlsx"
FINANCE_REPORT_SOURCE = "data/财务报表.pdf"
INTERNAL_KB_SOURCE = "data/韩宁科技有限公司内部全套知识库.docx"

# ===== 阶段 6：表格召回准确率指标 (TFHA) =====
# 表格类样本的 category 白名单
TABLE_CATEGORIES: set[str] = {"salary_table", "finance_table"}


def calc_table_field_hit_accuracy(
    dataset_samples: list[dict[str, Any]],
    per_sample_evidence: dict[str, list],
) -> dict[str, Any]:
    """计算 Table Field Hit Accuracy (TFHA) —— 表格字段召回准确率。

    定义：对所有 category ∈ {salary_table, finance_table} 的表格数值问答样本，
    检查其检索命中的 evidence_docs 的 page_content 中，**是否包含被问的字段标签文本**
    （如"个税""营业成本""净利润"）。只要任意一个 field label 在任意一条 evidence doc
    的文本里出现，就算该样本命中。TFHA 越接近 1 越好。

    字段标签优先级：
      1. 样本自带 ``expected_answer_contains_field_labels``（专用字段标签列表）
      2. 退回使用 ``expected_answer_contains``（期望答案包含的子串）

    Parameters
    ----------
    dataset_samples : list[dict]
        原始评估集样本（含 category / expected_answer_contains_field_labels 等字段）。
    per_sample_evidence : dict[str, list[Document]]
        {sample_id -> 该样本检索出的 evidence doc 列表（需有 page_content 属性）}。

    Returns
    -------
    dict with keys: tfha (float 0~1), hit_count (int), total_table_samples (int)
    """
    table_samples: list[dict[str, Any]] = [
        s for s in dataset_samples
        if s.get("category") in TABLE_CATEGORIES
    ]
    total = len(table_samples)
    if total == 0:
        return {"tfha": 0.0, "hit_count": 0, "total_table_samples": 0}

    hit_count = 0
    for s in table_samples:
        sid = s.get("id", "")
        field_labels: list[str] = list(
            s.get("expected_answer_contains_field_labels")
            or s.get("expected_answer_contains")
            or []
        )
        if not field_labels:
            continue

        docs = per_sample_evidence.get(sid) or []
        any_field_hit = False
        for field in field_labels:
            field_text = str(field).strip()
            if not field_text:
                continue
            # 检查任意一条 evidence doc 中包含该字段文本
            for doc in docs:
                content = (getattr(doc, "page_content", None) or "")
                if field_text in content:
                    any_field_hit = True
                    break
            if any_field_hit:
                break
        if any_field_hit:
            hit_count += 1

    return {
        "tfha": hit_count / total if total > 0 else 0.0,
        "hit_count": hit_count,
        "total_table_samples": total,
    }


def format_tfha_report(tfha_result: dict[str, Any]) -> str:
    """把 TFHA 指标格式化成控制台可读的分隔框。"""
    tfha_pct = float(tfha_result.get("tfha", 0.0)) * 100
    hit = int(tfha_result.get("hit_count", 0))
    total = int(tfha_result.get("total_table_samples", 0))
    lines = [
        "==== Table Retrieval Accuracy ==",
        f"Table Field Hit Accuracy (TFHA) :  {tfha_pct:.1f}%  ({hit}/{total})",
        "================================",
    ]
    return "\n".join(lines)


def load_eval_dataset(dataset_path: str) -> list[dict[str, Any]]:
    """加载 JSONL 格式的评估集。"""
    abs_path = get_abs_path(dataset_path)
    samples: list[dict[str, Any]] = []
    with open(abs_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"⚠️  第 {line_no} 行 JSON 解析失败，跳过：{e}")
                continue

            expected_refusal = bool(sample.get("expected_refusal", False))
            expected_sources = sample.get("expected_sources")
            if not expected_refusal and not expected_sources:
                inferred_sources = infer_expected_sources_from_sample(sample)
                if inferred_sources:
                    sample["expected_sources"] = inferred_sources
                    expected_sources = inferred_sources
            if not expected_refusal and not expected_sources:
                sid = sample.get("id", f"line-{line_no}")
                raise ValueError(
                    f"评估样本 {sid} 缺少 expected_sources；"
                    "非拒答样本必须标注期望来源，避免检索指标被空标签污染。"
                )
            if expected_refusal and expected_sources is None:
                sample["expected_sources"] = []
            samples.append(sample)
    return samples


def infer_expected_sources_from_sample(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """从现有 required_source_keywords 推断文件级 expected_sources。

    这是对旧评估集的兼容层；新评估样本仍建议显式写 expected_sources。
    """
    import re

    hints = [str(x) for x in sample.get("required_source_keywords") or []]
    question = str(sample.get("question") or sample.get("query") or "")
    text = " ".join([question, *hints])
    paths: list[str] = []

    salary_months = {
        int(m)
        for m in re.findall(r"员工工资表_2026年(\d{1,2})月", text)
        if 1 <= int(m) <= 12
    }
    if "Q1" in text.upper():
        salary_months.update({1, 2, 3})
    if "Q2" in text.upper():
        salary_months.update({4, 5, 6})
    for m in re.findall(r"2026年(\d{1,2})月", text):
        month = int(m)
        if 1 <= month <= 12 and "工资" in text:
            salary_months.add(month)
    if not salary_months and "员工工资表" in text:
        salary_months.update(range(1, 7))

    for month in sorted(salary_months):
        paths.append(SALARY_FILE_PATTERN.format(month=month))

    if "财务报表" in text or "利润表" in text or "营业收入" in text or "净利润" in text:
        paths.append(FINANCE_REPORT_SOURCE)
    if "内部知识库" in text or "知识库" in text:
        paths.append(INTERNAL_KB_SOURCE)

    deduped: list[str] = []
    for path in paths:
        if path not in deduped:
            deduped.append(path)
    return [{"source_path": path} for path in deduped]


def normalize_source_for_matching(path: str) -> str:
    """把相对/绝对路径统一成 registry 使用的稳定字符串。"""
    if not path:
        return ""
    raw_path = Path(path)
    if raw_path.is_absolute():
        return canonicalize_source_path(str(raw_path))
    return canonicalize_source_path(get_abs_path(path))


def expected_to_tuples(expected_sources: list[dict[str, Any]]) -> list[tuple[str, int | None]]:
    """把 expected_sources 转成 (source_path, page) 元组列表，路径已归一化。"""
    result: list[tuple[str, int | None]] = []
    for src in expected_sources:
        path = str(src.get("source_path") or "")
        path = normalize_source_for_matching(path)
        page = src.get("page")
        if page is not None:
            try:
                page = int(page)
            except (ValueError, TypeError):
                page = None
        result.append((path, page))
    return result


def docs_to_source_list(docs: list[Any]) -> list[dict[str, Any]]:
    """把检索到的 Document 列表转成指标模块需要的 source 字典列表，路径已归一化。"""
    result: list[dict[str, Any]] = []
    for doc in docs:
        meta = doc.metadata or {}
        source_path = meta.get("source_path") or meta.get("source") or ""
        source_path = normalize_source_for_matching(source_path)
        result.append({
            "source_path": source_path,
            "page": meta.get("page"),
        })
    return result


def _apply_eval_overrides(
    rag: RagSummarizeService,
    *,
    mode: str,
    no_rerank: bool,
    vector_top_k: int,
    bm25_top_k: int,
    fusion_top_k: int,
    final_top_k: int,
    min_score: float,
) -> None:
    """把 CLI 实验参数覆盖到 RagSummarizeService 的 HybridRetriever 上。"""
    hr = rag.hybrid_retriever
    hr.mode = mode
    hr.vector_top_k = vector_top_k
    hr.bm25_top_k = bm25_top_k
    hr.fusion.fusion_top_k = fusion_top_k
    hr.score_guard = ScoreGuard(
        min_score=min_score,
        final_top_k=final_top_k,
    )
    hr.reranker.enabled = not no_rerank


def collect_experiment_config(
    top_k: int,
    ks: tuple[int, ...],
    mode: str,
    vector_top_k: int,
    bm25_top_k: int,
    fusion_top_k: int,
    final_top_k: int,
    rerank_enabled: bool,
    min_score: float,
) -> dict[str, Any]:
    """收集当前实验的配置参数，写入报告便于对比。"""
    return {
        "parser_version": str(chroma_conf.get("parser_version", "unknown")),
        "chunk_size": chroma_conf.get("chunk_size", "unknown"),
        "chunk_overlap": chroma_conf.get("chunk_overlap", "unknown"),
        "top_k": top_k,
        "ks": ",".join(str(k) for k in ks),
        "eval_pipeline": "EnterpriseQueryService.retrieve_only",
        "retrieval_mode": mode,
        "vector_top_k": vector_top_k,
        "bm25_top_k": bm25_top_k,
        "fusion_top_k": fusion_top_k,
        "final_top_k": final_top_k,
        "rerank_enabled": "true" if rerank_enabled else "false",
        "min_score": min_score,
        "embedding_model": str(rag_conf.get("embedding_model_name") or chroma_conf.get("embedding_model_name") or "unknown"),
        "collection_name": chroma_conf.get("collection_name", "unknown"),
    }


def run_evaluation(
    dataset_path: str,
    top_k: int,
    ks: tuple[int, ...],
    load_documents: bool,
    mode: str,
    no_rerank: bool,
    vector_top_k: int,
    bm25_top_k: int,
    fusion_top_k: int,
    final_top_k: int,
    min_score: float,
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, Any], dict[str, Any], float, dict[str, Any]]:
    """执行完整的检索评估，返回（逐条结果, 平均指标, 配置信息, 完整参数快照, duration_sec, tfha_result）。"""
    import time as _time
    _start_ts = _time.time()
    print("初始化 RAG 服务（EnterpriseQueryService 统一编排）...")
    rag = RagSummarizeService()

    if load_documents:
        print("加载知识库文档（增量入库）...")
        rag.vector_store.load_document()
        print("知识库加载完成。")

    _apply_eval_overrides(
        rag,
        mode=mode,
        no_rerank=no_rerank,
        vector_top_k=vector_top_k,
        bm25_top_k=bm25_top_k,
        fusion_top_k=fusion_top_k,
        final_top_k=final_top_k,
        min_score=min_score,
    )

    print(f"检索模式：{mode}（EnterpriseQueryService: Hybrid → Expander → Compressor）")
    print(f"参数：vector_top_k={vector_top_k}, bm25_top_k={bm25_top_k}, "
          f"fusion_top_k={fusion_top_k}, final_top_k={final_top_k}, "
          f"eval_top_k={top_k}, min_score={min_score}, "
          f"rerank={'禁用' if no_rerank else '启用'}")

    # 加载评估集
    print(f"加载评估集：{dataset_path}")
    samples = load_eval_dataset(dataset_path)
    print(f"共 {len(samples)} 条评估样本。\n")

    per_sample_results: list[dict[str, Any]] = []
    all_metrics: list[dict[str, float]] = []

    # 样本分组统计：expected_sources 中哪些标注了 page，哪些没有
    samples_with_page: int = 0
    samples_without_page: int = 0
    metrics_with_page: list[dict[str, float]] = []
    metrics_without_page: list[dict[str, float]] = []
    # 文件级 vs 页码级 双维度对比：同一条样本，按 expected_sources 里是否标了 page 分别算两套 hit
    page_level_metrics_all: list[dict[str, float]] = []

    # ---- 阶段 6 新增：保存每条样本完整证据文档，用于 TFHA（表格字段召回准确率）计算 ----
    per_sample_evidence_docs: dict[str, list] = {}

    for idx, sample in enumerate(samples, start=1):
        qid = sample.get("id", f"q{idx:03d}")
        question = sample.get("question", "")
        expected = expected_to_tuples(sample.get("expected_sources", []))

        if not question:
            print(f"[{qid}] 跳过（无 question 字段）")
            continue

        if bool(sample.get("expected_refusal", False)) and not expected:
            zero_metrics = {name: 0.0 for k in ks for name in (f"hit_at_{k}", f"mrr_at_{k}", f"recall_at_{k}")}
            row = {
                "id": qid,
                "question": question,
                "expected_count": 0,
                "has_page_expected": "skip",
                "pipeline_evidence_count": 0,
                "eval_doc_count": 0,
                "query_intent": "permission_refusal",
                "top3_sources": "",
                "hit_at_any": "skip",
                "page_hit_gap": "expected_refusal",
                **zero_metrics,
            }
            per_sample_results.append(row)
            print(f"[{idx:2d}/{len(samples)}] {qid} [SKIP] 权限拒答样本不参与检索来源指标")
            continue

        # 判断该样本是否标注了至少一个 page 信息
        has_page_any = any(exp[1] is not None for exp in expected)
        if has_page_any:
            samples_with_page += 1
        else:
            samples_without_page += 1

        # 执行检索：与线上一致的 retriever_docs 全链路
        pipeline_docs = rag.retriever_docs(question)
        docs = pipeline_docs[:top_k]
        intent = (pipeline_docs[0].metadata or {}).get("_query_intent", "") if pipeline_docs else ""
        # ---- 阶段 6 新增：保存完整 pipeline_docs（后续用于 TFHA）----
        per_sample_evidence_docs[qid] = list(pipeline_docs)

        # 转成 source 列表（路径已归一化）
        retrieved_sources = docs_to_source_list(docs)

        # ---- 文件级 + 页码级 双维度指标计算 ----
        # A) 主指标：按评估集 expected 原数据计算（支持 page 混合）
        metrics = compute_all_retrieval_metrics(retrieved_sources, expected, ks=ks)

        # B) 页码级指标：对 expected_sources 中有 page 的样本，单独再算一遍
        #    「严格页码命中」（必须文件相同 + page 都对得上）
        page_level_metrics = metrics  # 默认复用主指标
        if has_page_any:
            # 构造严格版 expected：page=None 的先不管，只对有 page 的做强校验（这里主指标其实已经做了）
            # 再构造一份「文件级放宽版」：把所有 expected 的 page 全部丢到 None，看看"只看文件"时命中率多少
            file_level_expected: list[tuple[str, int | None]] = []
            for sp, _p in expected:
                file_level_expected.append((sp, None))
            file_level_metrics = compute_all_retrieval_metrics(
                retrieved_sources, file_level_expected, ks=ks
            )
            # 严格主指标 就是 metrics；把两份指标取差值：页码要求 vs 不要求的命中率差
            page_level_metrics_all.append(file_level_metrics)
            metrics_with_page.append(metrics)
        else:
            metrics_without_page.append(metrics)

        # hit_at_any 通过 hit_at_k(retrieved, expected, len(retrieved)) 实现
        hit_any = hit_at_k(retrieved_sources, expected, len(retrieved_sources)) == 1.0

        # 命中详情（前 3 条来源，带页码）
        top_sources_with_page: list[str] = []
        for s in retrieved_sources[:3]:
            sp = s["source_path"]
            p = s.get("page")
            base = os.path.basename(sp) if sp else "?"
            if p is not None:
                top_sources_with_page.append(f"{base}:p{p}")
            else:
                top_sources_with_page.append(base)

        # 页码 vs 文件级 的差值（对有 page 标注的样本）
        page_hit_diff = ""
        if has_page_any:
            file_hit = file_level_metrics.get("hit_at_1", 0.0)
            page_hit = metrics.get("hit_at_1", 0.0)
            delta = file_hit - page_hit
            page_hit_diff = f"ΔHit@1(file-page)={delta:.2f}"

        row = {
            "id": qid,
            "question": question,
            "expected_count": len(expected),
            "has_page_expected": "✓" if has_page_any else "✗",
            "pipeline_evidence_count": len(pipeline_docs),
            "eval_doc_count": len(docs),
            "query_intent": intent or "unknown",
            "top3_sources": "; ".join(top_sources_with_page),
            "hit_at_any": "✓" if hit_any else "✗",
            "page_hit_gap": page_hit_diff,
            **metrics,
        }
        per_sample_results.append(row)
        all_metrics.append(metrics)

        hit_str = "✓" if hit_any else "✗"
        page_tag = "[PAGE]" if has_page_any else "[FILE]"
        print(f"[{idx:2d}/{len(samples)}] {qid} {page_tag} {hit_str}  {question[:40]}...  {page_hit_diff}")

    # 计算均值
    avg = average_metrics(all_metrics)

    # 分组均值
    avg_with_page = average_metrics(metrics_with_page) if metrics_with_page else {}
    avg_without_page = average_metrics(metrics_without_page) if metrics_without_page else {}
    avg_page_file_diff: dict[str, float] = {}
    if avg_with_page and page_level_metrics_all:
        avg_file_level_on_page_samples = average_metrics(page_level_metrics_all)
        for k in avg_with_page:
            avg_page_file_diff[f"pagegap_{k}"] = (
                avg_file_level_on_page_samples.get(k, 0.0) - avg_with_page.get(k, 0.0)
            )

    # 聚合维度：按「页码标注分组」的分层统计一起写进 config
    config = collect_experiment_config(
        top_k=top_k,
        ks=ks,
        mode=mode,
        vector_top_k=vector_top_k,
        bm25_top_k=bm25_top_k,
        fusion_top_k=fusion_top_k,
        final_top_k=final_top_k,
        rerank_enabled=not no_rerank,
        min_score=min_score,
    )
    config["samples_with_page"] = samples_with_page
    config["samples_without_page"] = samples_without_page
    config["avg_with_page"] = json.dumps(avg_with_page, ensure_ascii=False)
    config["avg_without_page"] = json.dumps(avg_without_page, ensure_ascii=False)
    if avg_page_file_diff:
        config["avg_page_file_gap"] = json.dumps(avg_page_file_diff, ensure_ascii=False)

    duration_sec = round(_time.time() - _start_ts, 3)

    # ===== 阶段 6 新增：计算 TFHA 表格字段召回准确率 =====
    tfha_result: dict[str, Any] = calc_table_field_hit_accuracy(
        samples, per_sample_evidence_docs
    )

    # 完整参数快照（experiment_store 做 hash 时会自动 canonicalize）
    full_params_snapshot: dict[str, Any] = {
        "dataset_path": dataset_path,
        "ks": list(ks),
        "top_k": top_k,
        "eval_pipeline": "EnterpriseQueryService.retrieve_only",
        "load_documents": bool(load_documents),
        "mode": mode,
        "no_rerank": bool(no_rerank),
        "vector_top_k": vector_top_k,
        "bm25_top_k": bm25_top_k,
        "fusion_top_k": fusion_top_k,
        "final_top_k": final_top_k,
        "min_score": min_score,
        "collection_name": config.get("collection_name"),
        "parser_version": config.get("parser_version"),
        "chunk_size": config.get("chunk_size"),
        "chunk_overlap": config.get("chunk_overlap"),
        "embedding_model": config.get("embedding_model"),
        "retrieval_extra": {
            "concurrent_recall": rag_conf.get("retrieval", {}).get("concurrent_recall"),
            "enable_second_pass": rag_conf.get("retrieval", {}).get("enable_second_pass"),
            "second_pass_max_expansion": rag_conf.get("retrieval", {}).get("second_pass_max_expansion"),
        },
        "score_guard_extra": {
            "strategy": rag_conf.get("score_guard", {}).get("strategy"),
            "min_candidates_after_filter": rag_conf.get("score_guard", {}).get("min_candidates_after_filter"),
            "second_pass_expansion_factor": rag_conf.get("score_guard", {}).get("second_pass_expansion_factor"),
            "second_pass_relax_min_score_factor": rag_conf.get("score_guard", {}).get("second_pass_relax_min_score_factor"),
        },
        "query_rewrite_extra": {
            "enabled": rag_conf.get("query_rewrite", {}).get("enabled"),
            "max_rewrites": rag_conf.get("query_rewrite", {}).get("max_rewrites"),
            "enable_multi_query_recall": rag_conf.get("query_rewrite", {}).get("enable_multi_query_recall"),
        },
        "samples_with_page": samples_with_page,
        "samples_without_page": samples_without_page,
        "table_field_hit_accuracy": tfha_result,  # 阶段 6：TFHA 结果进快照
    }
    # 把 TFHA 写进 config，便于 Markdown 报告顶部展示
    config["table_field_hit_accuracy"] = json.dumps(tfha_result, ensure_ascii=False)
    return per_sample_results, avg, config, full_params_snapshot, duration_sec, tfha_result


def write_csv_report(
    per_sample_results: list[dict[str, Any]],
    avg_metrics: dict[str, float],
    config: dict[str, Any],
    report_path: str,
) -> None:
    """把评估结果写入 CSV 文件。"""
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(per_sample_results[0].keys()) if per_sample_results else []

    with open(report_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_sample_results:
            writer.writerow(row)

        writer.writerow({})
        avg_row = {k: "" for k in fieldnames}
        avg_row["id"] = "AVERAGE"
        for key, value in avg_metrics.items():
            if key in avg_row:
                avg_row[key] = round(value, 4)
        writer.writerow(avg_row)

        writer.writerow({})
        config_row = {k: "" for k in fieldnames}
        config_row["id"] = "CONFIG"
        config_row["question"] = json.dumps(config, ensure_ascii=False)
        writer.writerow(config_row)


def write_markdown_report(
    per_sample_results: list[dict[str, Any]],
    avg_metrics: dict[str, float],
    config: dict[str, Any],
    ks: tuple[int, ...],
    report_path: str,
) -> None:
    """生成 Markdown 格式的评估报告。"""
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append(f"# RAG 检索评估报告\n")
    lines.append(f"**生成时间**：{now}\n")

    lines.append("## 实验配置\n")
    lines.append("| 参数 | 值 |")
    lines.append("|------|----|")
    for key, value in config.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")

    lines.append("## 整体指标\n")
    lines.append("| 指标 | 分数 |")
    lines.append("|------|------|")
    for key, value in sorted(avg_metrics.items()):
        lines.append(f"| {key} | {value:.4f} |")
    lines.append("")

    lines.append("## 逐条结果\n")
    display_cols = ["id", "hit_at_any", "question"] + [f"hit_at_{k}" for k in ks] + [f"mrr_at_{ks[-1]}", f"recall_at_{ks[-1]}"]
    header = "| " + " | ".join(display_cols) + " |"
    sep = "|" + "|".join(["---"] * len(display_cols)) + "|"
    lines.append(header)
    lines.append(sep)

    for row in per_sample_results:
        values = []
        for col in display_cols:
            v = row.get(col, "")
            if isinstance(v, float):
                values.append(f"{v:.4f}")
            else:
                text = str(v)
                if col == "question" and len(text) > 40:
                    text = text[:40] + "..."
                values.append(text)
        lines.append("| " + " | ".join(values) + " |")

    lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 检索评估（Hit@K / MRR@K / Recall@K）")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help=f"评估集路径，默认：{DEFAULT_DATASET}")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="指标计算时考察的前 K 条 evidence（默认 max(final_top_k, max(ks))，与线上一致）",
    )
    parser.add_argument("--ks", type=str, default=",".join(str(k) for k in DEFAULT_KS), help=f"要计算的 K 值列表")
    parser.add_argument("--load-documents", action="store_true", help="先执行知识库增量入库")
    parser.add_argument("--mode", choices=["hybrid", "vector", "bm25"], default="hybrid", help="检索模式")
    parser.add_argument("--no-rerank", action="store_true", help="跳过 Rerank")

    # 实验参数（显式控制各阶段）
    retrieval_cfg = rag_conf.get("retrieval", {}) or {}
    score_guard_cfg = rag_conf.get("score_guard", {}) or {}
    rerank_cfg = rag_conf.get("rerank", {}) or {}

    parser.add_argument("--vector-top-k", type=int, default=retrieval_cfg.get("vector_top_k", 20), help="向量召回条数")
    parser.add_argument("--bm25-top-k", type=int, default=retrieval_cfg.get("bm25_top_k", 20), help="BM25 召回条数")
    parser.add_argument("--fusion-top-k", type=int, default=retrieval_cfg.get("fusion_top_k", 30), help="融合后条数")
    parser.add_argument("--final-top-k", type=int, default=retrieval_cfg.get("final_top_k", 6), help="最终输出条数")
    parser.add_argument("--min-score", type=float, default=score_guard_cfg.get("min_score", rerank_cfg.get("min_score", 0.35)), help="ScoreGuard 最小分数")

    parser.add_argument("--output-prefix", default="retrieval_eval", help="报告文件名前缀")
    parser.add_argument("--tags", type=str, default="", help="逗号分隔的实验标签（写入 experiment registry）")
    parser.add_argument("--notes", type=str, default="", help="本次实验备注")
    parser.add_argument("--no-register", action="store_true", help="不写入 experiment registry（仅出报告文件）")
    args = parser.parse_args()

    ks = tuple(int(k.strip()) for k in args.ks.split(",") if k.strip())
    final_top_k = int(args.final_top_k)
    top_k = (
        int(args.top_k)
        if args.top_k is not None
        else max(final_top_k, max(ks) if ks else final_top_k)
    )

    print("=" * 60)
    print("  RAG 检索评估")
    print("=" * 60)

    (
        per_sample,
        avg,
        config,
        full_params_snapshot,
        duration_sec,
        tfha_result,
    ) = run_evaluation(
        dataset_path=args.dataset,
        top_k=args.top_k,
        ks=ks,
        load_documents=args.load_documents,
        mode=args.mode,
        no_rerank=args.no_rerank,
        vector_top_k=args.vector_top_k,
        bm25_top_k=args.bm25_top_k,
        fusion_top_k=args.fusion_top_k,
        final_top_k=args.final_top_k,
        min_score=args.min_score,
    )

    if not per_sample:
        print("\n❌ 没有有效的评估样本。")
        return 1

    # ===== 阶段 6 新增：TFHA 并入整体指标（均值报告统一出口）=====
    avg["table_field_hit_accuracy"] = float(tfha_result.get("tfha", 0.0))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = Path(get_abs_path(REPORTS_DIR))
    csv_path = reports_dir / f"{args.output_prefix}_{timestamp}.csv"
    md_path = reports_dir / f"{args.output_prefix}_{timestamp}.md"
    tfha_json_path = reports_dir / f"{args.output_prefix}_{timestamp}_tfha.json"

    write_csv_report(per_sample, avg, config, str(csv_path))
    write_markdown_report(per_sample, avg, config, ks, str(md_path))

    # ===== 阶段 6 新增：导出 TFHA 独立 JSON 报告 =====
    try:
        tfha_report_content: dict[str, Any] = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "dataset_path": args.dataset,
            "sample_count": len(per_sample),
            "tfha_result": tfha_result,
            "tfha_report_text": format_tfha_report(tfha_result),
        }
        with open(tfha_json_path, "w", encoding="utf-8") as _tjf:
            json.dump(tfha_report_content, _tjf, ensure_ascii=False, indent=2)
    except Exception as _tfha_exc:
        import warnings
        warnings.warn(f"[TFHAReport] 写入 JSON 报告失败，忽略：{_tfha_exc}")

    params_hash = run_id = None
    if not args.no_register:
        try:
            tags_list = [t.strip() for t in (args.tags or "").split(",") if t.strip()]
            params_hash, run_id = register_experiment_run(
                experiment_type="retrieval",
                params=full_params_snapshot,
                metrics=dict(avg),
                report_files={
                    "csv": str(csv_path),
                    "md": str(md_path),
                    "tfha_json": str(tfha_json_path),
                },
                duration_sec=duration_sec,
                sample_count=len(per_sample),
                tags=tags_list,
                notes=args.notes,
                extra={
                    "config_summary": config,
                    "table_field_hit_accuracy": tfha_result,
                },
            )
        except Exception as _exc:
            # registry 写入失败不要影响报告生成（fail-open）
            import warnings
            warnings.warn(f"[ExperimentRegistry] 写入失败，忽略：{_exc}")

    print("\n" + "=" * 60)
    print("  评估结果汇总")
    print("=" * 60)
    print(f"样本数：{len(per_sample)}")
    for key, value in sorted(avg.items()):
        print(f"  {key:<26s}: {value:.4f}")
    # ===== 阶段 6 新增：控制台打印 TFHA 分隔框 =====
    print()
    print(format_tfha_report(tfha_result))
    print(f"\nCSV 报告            ：{csv_path}")
    print(f"MD  报告            ：{md_path}")
    print(f"TFHA 指标 JSON      ：{tfha_json_path}")
    if params_hash and run_id:
        print(f"\n📦 参数版本 params_hash ：{params_hash}")
        print(f"🏷️  实验 run_id          ：{run_id}")
        print(f"     （可在 eval/reports/_experiment_registry.json 中查看历史对比）")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
