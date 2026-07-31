"""RAG 文档清洗 + Chunk 清洗 + 四层质量门（L1/L2/L3/L4）。

改造于 v2.3：
- L1：Document 级清洗（全局重复行/TOC 整页丢弃/OCR 乱码比/信息密度兜底/黑名单行）
- L3：Chunk 级清洗（信息密度门槛/精确去重/64bit SimHash 近似去重/Markdown 表格行兜底）
- L4：入库 QualityReport（可观测性 + 告警阈值，fail-open 不中断流程）
- L2 噪声清理（装饰线/指引行剥离）在 semantic_chunker.py 中落地。
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document


# ======================================================================
# 正则：L1 Document 级
# ======================================================================
_BLANK_LINE_RE = re.compile(r"\n{3,}")
_HORIZONTAL_SPACE_RE = re.compile(r"[ \t\r\f\v]+")
_SYMBOL_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)
_PAGE_NUMBER_RE = re.compile(r"^\s*(?:第\s*\d{1,4}\s*页|-\s*\d{1,4}\s*-|\d{1,4}\s*/\s*\d{1,4})\s*$")
_SINGLE_CJK_LINE_RE = re.compile(r"^[\u4e00-\u9fff]$")

# TOC / 目录行："1.1 项目概述 ............... 5" 或 "1.1 项目概述……… 5" 或 "目录  1  2  3"
_TOC_LINE_RE = re.compile(
    r"^(?P<prefix>\d+(?:[.\-、]\d+)*|[一二三四五六七八九十百千万]+[章节篇])?\s*"
    r"(?P<title>.+?)"
    r"\s*(?:[.·•…\-—_\s]{3,}|[ \t]{2,})\s*"
    r"(?P<page>\d{1,4}|[一二三四五六七八九十百千万]{1,3})\s*$"
)

# OCR 乱码字符类（逐个字符判断，避免 re 引擎对 \U+100000+ 长码位在 [] 内的误匹配）：
#   - CJK 私有区 A：U+E000 - U+F8FF
#   - CJK 补充私有区 A/B：ord(c) >= 0xF0000 且 <= 0x10FFFD
#   - 不可打印 ASCII 控制字符（除 \n \r \t 外，\r 会在 normalize 阶段转成 \n 不匹配）
def _is_garbage_char(c: str) -> bool:
    if not c:
        return False
    o = ord(c)
    if 0xE000 <= o <= 0xF8FF:  # CJK 私有区
        return True
    if 0xF0000 <= o <= 0x10FFFD:  # 补充私有区 A/B
        return True
    if (0x00 <= o <= 0x08) or (0x0B <= o <= 0x1F) or o == 0x7F:  # 控制字符除 \n \r \t
        return True
    return False

# "见第 X 节 / 参考附件 X" 纯指引行（L2 切分层剥离用，L3 也识别）
_REF_DIRECTIVE_RE = re.compile(
    r"^\s*(?:参?见|参考|查阅|详见|参见|另见)\s*"
    r"(?:第?\s*[一二三四五六七八九十百千万0-9]+(?:[.\-、][一二三四五六七八九十百千万0-9]+)*\s*"
    r"(?:章|节|小节|部分|段|条|章|篇|页)"
    r"|附件\s*[0-9A-Z]+|"
    r"附录\s*[A-Z0-9]+)"
    r"\s*[。，,.]?\s*$"
)

# Markdown 装饰线（连续符号长行 / --- / *** / ### ...）
_MD_DECOR_RE = re.compile(r"^\s*[-*_=#·•…\s]{8,}\s*$")

# ======================================================================
# 字符统计：CJK 字符 + ASCII 字母数字（信息密度计算的"有效字"）
# ======================================================================
_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_ALNUM_WORD_RE = re.compile(r"\b[a-zA-Z0-9_]{2,}\b")
_MD_TABLE_ALIGN_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")
_MD_TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")


# ======================================================================
# 配置 dataclass（从 rag.yml data_cleaning 段读出后，构造给 normalize 函数使用）
# ======================================================================
@dataclass(frozen=True)
class DataCleaningConfig:
    """四层清洗阈值配置。所有字段均有保守默认值。"""

    # L1 Document 级
    l1_enable: bool = True
    l1_global_repeat_row_ratio: float = 0.60
    l1_toc_min_lines: int = 5
    l1_garbage_char_ratio_max: float = 0.03
    l1_page_min_chars: int = 80
    l1_blacklist_lines: tuple[str, ...] = (
        "版权所有",
        "©",
        "仅供参考",
        "不构成任何",
        "confidential",
        "internal use only",
        "保密",
    )

    # L2 切分层（开关只做校验；实际正则由 semantic_chunker 导入使用）
    l2_enable: bool = True
    l2_strip_md_decor: bool = True
    l2_strip_ref_directives: bool = True

    # L3 Chunk 级
    l3_enable: bool = True
    l3_density_min_cjk_chars: int = 15
    l3_density_min_cjk_ratio: float = 0.30
    l3_density_min_ascii_words: int = 8
    l3_exact_dedup: bool = True
    l3_simhash_dedup: bool = True
    l3_simhash_hamming_max: int = 3
    l3_table_require_data_rows: int = 1

    # L4 QualityReport
    l4_kept_ratio_warn: float = 0.60
    l4_dedup_ratio_warn: float = 0.30
    report_to_metadata: bool = True

    @classmethod
    def from_rag_conf(cls, rag_conf: dict[str, Any] | None) -> "DataCleaningConfig":
        dc = (rag_conf or {}).get("data_cleaning", {}) or {}
        bl = dc.get("l1_blacklist_lines")
        if isinstance(bl, list):
            blacklist = tuple(str(x).strip() for x in bl if str(x).strip())
        else:
            blacklist = cls.l1_blacklist_lines
        return cls(
            l1_enable=bool(dc.get("l1_enable", True)),
            l1_global_repeat_row_ratio=float(dc.get("l1_global_repeat_row_ratio", 0.60)),
            l1_toc_min_lines=int(dc.get("l1_toc_min_lines", 5)),
            l1_garbage_char_ratio_max=float(dc.get("l1_garbage_char_ratio_max", 0.03)),
            l1_page_min_chars=int(dc.get("l1_page_min_chars", 80)),
            l1_blacklist_lines=blacklist,
            l2_enable=bool(dc.get("l2_enable", True)),
            l2_strip_md_decor=bool(dc.get("l2_strip_md_decor", True)),
            l2_strip_ref_directives=bool(dc.get("l2_strip_ref_directives", True)),
            l3_enable=bool(dc.get("l3_enable", True)),
            l3_density_min_cjk_chars=int(dc.get("l3_density_min_cjk_chars", 15)),
            l3_density_min_cjk_ratio=float(dc.get("l3_density_min_cjk_ratio", 0.30)),
            l3_density_min_ascii_words=int(dc.get("l3_density_min_ascii_words", 8)),
            l3_exact_dedup=bool(dc.get("l3_exact_dedup", True)),
            l3_simhash_dedup=bool(dc.get("l3_simhash_dedup", True)),
            l3_simhash_hamming_max=int(dc.get("l3_simhash_hamming_max", 3)),
            l3_table_require_data_rows=int(dc.get("l3_table_require_data_rows", 1)),
            l4_kept_ratio_warn=float(dc.get("l4_kept_ratio_warn", 0.60)),
            l4_dedup_ratio_warn=float(dc.get("l4_dedup_ratio_warn", 0.30)),
            report_to_metadata=bool(dc.get("report_to_metadata", True)),
        )


# ======================================================================
# QualityReport：L4 入库质量门报告
# ======================================================================
@dataclass
class QualityReport:
    # L1 Document 级
    l1_documents_input: int = 0
    l1_documents_kept: int = 0
    l1_documents_dropped: int = 0
    l1_drop_reasons: Counter = field(default_factory=Counter)

    # L3 Chunk 级
    l3_chunks_input: int = 0
    l3_chunks_kept: int = 0
    l3_chunks_dropped: int = 0
    l3_dedup_exact: int = 0
    l3_dedup_simhash: int = 0
    l3_drop_reasons: Counter = field(default_factory=Counter)

    # 汇总
    @property
    def l1_kept_ratio(self) -> float:
        if self.l1_documents_input <= 0:
            return 1.0
        return self.l1_documents_kept / self.l1_documents_input

    @property
    def l3_kept_ratio(self) -> float:
        if self.l3_chunks_input <= 0:
            return 1.0
        return self.l3_chunks_kept / self.l3_chunks_input

    @property
    def l3_dedup_ratio(self) -> float:
        if self.l3_chunks_input <= 0:
            return 0.0
        return (self.l3_dedup_exact + self.l3_dedup_simhash) / self.l3_chunks_input

    def summary(self) -> str:
        return (
            f"[QualityReport] L1 docs in={self.l1_documents_input} kept={self.l1_documents_kept} "
            f"drop={self.l1_documents_dropped}(kept={self.l1_kept_ratio:.2%}) | "
            f"L3 chunks in={self.l3_chunks_input} kept={self.l3_chunks_kept} "
            f"drop={self.l3_chunks_dropped}(kept={self.l3_kept_ratio:.2%}) "
            f"dedup_exact={self.l3_dedup_exact} dedup_simhash={self.l3_dedup_simhash}"
            f"(dedup_ratio={self.l3_dedup_ratio:.2%})"
        )


# ======================================================================
# ChunkBindingContext（保持原有接口不变）
# ======================================================================
@dataclass(frozen=True)
class ChunkBindingContext:
    """chunk 源数据绑定所需的稳定上下文。"""

    document_id: str
    document_version: str
    source_path: str
    content_hash: str
    parser_version: str
    chunking_version: str
    embedding_model: str
    embedding_version: str


def build_text_hash(text: str) -> str:
    """基于清洗后的 chunk 文本生成短 hash。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def infer_text_source(metadata: dict) -> str:
    """从 loader metadata 推断文本来源。
    返回值会被 normalize_documents 写进 metadata["text_source"]。
    """
    if metadata.get("layout_ocr"):
        return "layout_ocr"
    if metadata.get("ocr"):
        return "ocr"
    doc_type = (metadata.get("doc_type") or "").lower()
    if doc_type in ("md", "markdown"):
        return "md"
    if doc_type in ("docx", "doc", "word"):
        return "docx"
    source = str(metadata.get("source", "")).lower()
    if source.endswith(".pdf"):
        return "layout_ocr"
    if source.endswith(".txt"):
        return "txt"
    if source.endswith((".md", ".markdown")):
        return "md"
    if source.endswith((".doc", ".docx")):
        return "docx"
    return "unknown"


# ======================================================================
# 基础规范化函数（L0 共用，保持不变）
# ======================================================================
def normalize_document_text(text: str) -> str:
    """页面级清洗：只去噪和规范化，不改写语义。"""
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _join_single_cjk_lines(normalized)

    lines: list[str] = []
    for raw_line in normalized.split("\n"):
        line = _HORIZONTAL_SPACE_RE.sub(" ", raw_line).strip()
        if not line:
            lines.append("")
            continue
        if _PAGE_NUMBER_RE.match(line):
            continue
        lines.append(line)

    return _BLANK_LINE_RE.sub("\n\n", "\n".join(lines)).strip()


def normalize_chunk_text(text: str) -> str:
    """chunk 级清洗：压缩噪声，保留可检索关键词。"""
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _HORIZONTAL_SPACE_RE.sub(" ", normalized)
    normalized = _BLANK_LINE_RE.sub("\n\n", normalized)
    return normalized.strip()


# ======================================================================
# L1 Document 级清洗（新增核心）
# ======================================================================
def _build_global_repeat_rows(documents: list[Document], ratio: float, blacklist: tuple[str, ...]) -> set[str]:
    """统计同一文档集内所有行的出现频率；频率 >= max(2, N*ratio) 且非空非标题，作为页眉/页脚行返回。

    - 纯空行 / 页码行 / 极短的标题行不进黑名单（避免误删唯一出现的章节名）
    - 命中 l1_blacklist_lines 的短免责声明行也一并加入返回，不看频率（即使 1 次也删）
    """
    if not documents or ratio <= 0:
        return set()

    counter: Counter[str] = Counter()
    total_docs = len(documents)
    threshold = max(2, int(round(total_docs * max(0.0, min(1.0, ratio)))))

    for doc in documents:
        text = doc.page_content or ""
        if not text:
            continue
        seen_this_doc: set[str] = set()
        for raw in text.split("\n"):
            line = _HORIZONTAL_SPACE_RE.sub(" ", raw).strip()
            if not line or len(line) <= 2:
                continue
            if _PAGE_NUMBER_RE.match(line):
                continue
            # 每行在同一页里重复出现多次也只计数 1 次（更稳）
            if line in seen_this_doc:
                continue
            seen_this_doc.add(line)
            counter[line] += 1

    repeat_rows: set[str] = {line for line, cnt in counter.items() if cnt >= threshold}

    # 黑名单命中（即使只出现 1 次，单行也删）
    # 只在该行比较短时（< 200 字）删；防止"版权所有"出现在一整段正文中间被误删
    if blacklist:
        for doc in documents:
            for raw in (doc.page_content or "").split("\n"):
                line = _HORIZONTAL_SPACE_RE.sub(" ", raw).strip()
                if not line or len(line) >= 200:
                    continue
                low = line.lower()
                if any(kw.lower() in low for kw in blacklist):
                    repeat_rows.add(line)
    return repeat_rows


def _is_toc_page(text: str, min_lines: int) -> bool:
    """一页内命中 TOC_LINE_RE 的行数 >= min_lines 即判为目录页（整页丢弃）。"""
    if not text:
        return False
    hit = 0
    total = 0
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        total += 1
        if _TOC_LINE_RE.match(line):
            hit += 1
    # 还需满足"命中行数至少占该页总非空行的 50%"（防止误判"第 5 页 表-1..."这类单 TOC 行）
    if total <= 0:
        return False
    return hit >= min_lines and (hit / total) >= 0.5


def _garbage_ratio(text: str) -> float:
    if not text:
        return 0.0
    gc = sum(1 for c in text if _is_garbage_char(c))
    return gc / len(text)


def _informative_chars(text: str) -> int:
    """字符数统计：CJK 字 + ASCII 字母数字词（每个词按 1 个"有效字"加权 1 比较保守）。
    仅用在 l1_page_min_chars 密度兜底判断，不够精确但够快。
    """
    if not text:
        return 0
    cjk = len(_CJK_CHAR_RE.findall(text))
    words = len(_ALNUM_WORD_RE.findall(text))
    return cjk + words  # 保守：每个英文词算 1 个"有效位"


# ======================================================================
# L3 统计函数
# ======================================================================
def _cjk_stats(text: str) -> tuple[int, int, float]:
    if not text:
        return 0, 0, 0.0
    cjk = len(_CJK_CHAR_RE.findall(text))
    total = len(text)
    return cjk, total, (cjk / total if total > 0 else 0.0)


def _md_table_data_row_count(text: str) -> int:
    """统计 Markdown 表格里的数据行数量（排除表头/对齐线）。"""
    if not text:
        return 0
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return 0
    data_rows = 0
    header_candidate_seen = False
    for i, line in enumerate(lines):
        s = line.strip()
        is_table_row = bool(_MD_TABLE_ROW_RE.match(s))
        is_align = bool(_MD_TABLE_ALIGN_RE.match(s))
        if is_align:
            header_candidate_seen = True
            continue
        if is_table_row:
            # 表头对齐线前的那一行算表头，对齐线之后才算数据行
            if header_candidate_seen:
                data_rows += 1
            else:
                if i + 1 < len(lines) and _MD_TABLE_ALIGN_RE.match(lines[i + 1].strip()):
                    header_candidate_seen = True
                else:
                    # 无法判断是否表头，保守算数据行（避免把只有 1 条数据的小表整段丢掉）
                    data_rows += 1
    return data_rows


# ======================================================================
# SimHash 64bit 近似去重（纯 Python，不引第三方库）
# ======================================================================
def _simhash64(text: str, window: int = 2) -> int:
    """Minimal SimHash 64-bit.

    滑动窗口调优：CJK 单字默认 bigram（window=2）；ASCII 单词默认 unigram+bigram。
    对中文 RAG 场景，bigram（两字滑窗）对"仅改少量字/同义词替换"这种近似更敏感。
    """
    t = (text or "").strip()
    if not t:
        return 0
    tokens: list[str] = []
    cjks = _CJK_CHAR_RE.findall(t)
    words = _ALNUM_WORD_RE.findall(t)
    w = max(2, int(window))
    # CJK bigram（两字滑窗）
    for i in range(max(0, len(cjks) - w + 1)):
        tokens.append("".join(cjks[i : i + w]))
    # CJK unigram（单独每个字也算一份，避免 window=2 时仅有一个字的文本 token 太少）
    for ch in cjks:
        tokens.append(ch)
    # ASCII 单词：unigram + bigram
    for wd in words:
        tokens.append(wd.lower())
    for i in range(max(0, len(words) - 1)):
        tokens.append(f"{words[i]} {words[i+1]}".lower())
    if not tokens:
        digest = hashlib.sha256(t.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big", signed=False)
    vec = [0] * 64
    for tok in tokens:
        digest = hashlib.sha256(tok.encode("utf-8")).digest()
        hv = int.from_bytes(digest[:8], "big", signed=False)
        for bit in range(64):
            if (hv >> bit) & 1:
                vec[bit] += 1
            else:
                vec[bit] -= 1
    fp = 0
    for bit in range(64):
        if vec[bit] > 0:
            fp |= 1 << bit
    return fp


def _hamming64(a: int, b: int) -> int:
    x = a ^ b
    return bin(x).count("1")


# ======================================================================
# normalize_documents（L1 主入口）：返回 清洗后 docs + report
# 额外兼容：对外调用方如果只想拿 list[Document]，可忽略第二个返回值（保持向后兼容）
# ======================================================================
def normalize_documents(
    documents: list[Document],
    *,
    config: DataCleaningConfig | None = None,
) -> list[Document] | tuple[list[Document], QualityReport]:
    """清洗 loader 产出的页面级 Document。

    v2.3 升级：可选返回 (docs, QualityReport)；默认 config 为 None 时仍走保守默认值。
    """
    cfg = config or DataCleaningConfig()
    report = QualityReport()
    report.l1_documents_input = len(documents)
    normalized_documents: list[Document] = []

    if not cfg.l1_enable:
        # 关闭 L1：只做原有基础规范化 + 旧 _is_informative_text
        for doc in documents:
            text = normalize_document_text(doc.page_content)
            if not _is_informative_text(text):
                report.l1_documents_dropped += 1
                report.l1_drop_reasons["legacy_uninformative"] += 1
                continue
            metadata = dict(doc.metadata)
            metadata.setdefault("text_source", infer_text_source(metadata))
            normalized_documents.append(Document(page_content=text, metadata=metadata))
            report.l1_documents_kept += 1
        return normalized_documents, report

    # Step 1. 全局重复行 + 黑名单行（跨页统计后再删除）
    repeat_rows = _build_global_repeat_rows(
        documents,
        ratio=cfg.l1_global_repeat_row_ratio,
        blacklist=cfg.l1_blacklist_lines,
    )

    for doc in documents:
        text_base = normalize_document_text(doc.page_content)
        # 1.1 删除全局重复行（页眉页脚/版权/免责）
        stripped_lines: list[str] = []
        for raw in text_base.split("\n"):
            line = raw.strip()
            if line and line in repeat_rows:
                continue
            stripped_lines.append(raw if line else "")
        text = "\n".join(stripped_lines)
        text = _BLANK_LINE_RE.sub("\n\n", text).strip()

        # 1.2 TOC 整页丢弃
        if _is_toc_page(text, min_lines=cfg.l1_toc_min_lines):
            report.l1_documents_dropped += 1
            report.l1_drop_reasons["toc_page"] += 1
            continue

        # 1.3 OCR 乱码占比超阈值丢弃
        gr = _garbage_ratio(text)
        if gr > max(0.0, float(cfg.l1_garbage_char_ratio_max)):
            report.l1_documents_dropped += 1
            report.l1_drop_reasons[f"garbage_ratio_gt_{cfg.l1_garbage_char_ratio_max}"] += 1
            continue

        # 1.4 信息密度兜底（整页有效字太少）
        if _informative_chars(text) < int(cfg.l1_page_min_chars):
            report.l1_documents_dropped += 1
            report.l1_drop_reasons[f"page_chars_lt_{cfg.l1_page_min_chars}"] += 1
            continue

        # 1.5 旧的 _is_informative_text 兜底（空串/纯符号/<=1 字）
        if not _is_informative_text(text):
            report.l1_documents_dropped += 1
            report.l1_drop_reasons["legacy_uninformative"] += 1
            continue

        metadata = dict(doc.metadata)
        metadata.setdefault("text_source", infer_text_source(metadata))
        # 写入 L1 清洗特征到 metadata（后续 L4 quality_score 计算会用到）
        metadata["_l1_repeat_rows_removed"] = max(0, (len(text_base.split("\n")) - len(stripped_lines)))
        metadata["_l1_garbage_ratio"] = round(gr, 4)
        normalized_documents.append(Document(page_content=text, metadata=metadata))
        report.l1_documents_kept += 1

    return normalized_documents, report


# ======================================================================
# normalize_chunks（L3 主入口）：返回 清洗后 chunks + report
# 兼容：未传入 chunks_report 时在函数内部新建；上层调用方把 L1 report 传进来合并即可
# ======================================================================
def normalize_chunks(
    chunks: list[Document],
    *,
    config: DataCleaningConfig | None = None,
    chunks_report: QualityReport | None = None,
) -> list[Document] | tuple[list[Document], QualityReport]:
    """清洗并过滤 splitter 产出的 chunk（新增 L3 密度/精确去重/近似去重/表格兜底）。"""
    cfg = config or DataCleaningConfig()
    report = chunks_report or QualityReport()
    report.l3_chunks_input = len(chunks)
    output: list[Document] = []

    if not cfg.l3_enable:
        # 关闭 L3：只做原有基础规范化
        for chunk in chunks:
            text = normalize_chunk_text(chunk.page_content)
            if not _is_informative_text(text):
                report.l3_chunks_dropped += 1
                report.l3_drop_reasons["legacy_uninformative"] += 1
                continue
            meta = dict(chunk.metadata)
            output.append(Document(page_content=text, metadata=meta))
        report.l3_chunks_kept = len(output)
        return output, report

    # Step 1：先做基础规范化 + 密度门槛过滤（精确/近似去重之前先把明显垃圾清掉，省计算）
    first_pass: list[Document] = []
    min_cjk_chars = int(cfg.l3_density_min_cjk_chars)
    min_cjk_ratio = float(cfg.l3_density_min_cjk_ratio)
    min_ascii_words = int(cfg.l3_density_min_ascii_words)
    min_table_rows = int(cfg.l3_table_require_data_rows)

    for chunk in chunks:
        text = normalize_chunk_text(chunk.page_content)
        reasons: list[str] = []
        # 0. 旧规则兜底（空串/纯符号/<=1字）：直接 drop，不看后面任何例外
        if not _is_informative_text(text):
            reasons.append("legacy_uninformative")

        # 1. 表格兜底（优先级最高）：如果被识别为 Markdown 表格 chunk
        #    - table_data_rows >= min_table_rows → 直接通过（不被 CJK/ASCII 门槛误杀）
        #    - table_data_rows <  min_table_rows 且存在对齐线 → 直接 drop（空壳表头）
        #    - 不是 Markdown 表格 → 走 CJK/ASCII 正常门槛
        table_data_rows = _md_table_data_row_count(text)
        has_table_align = bool(_MD_TABLE_ALIGN_RE.search(text or ""))
        is_md_table = bool(has_table_align) or (bool(_MD_TABLE_ROW_RE.match((text or "").split("\n", 1)[0])) and table_data_rows >= 0)
        table_pass = (table_data_rows >= min_table_rows)

        if is_md_table:
            # Markdown 表格：看数据行够不够，够就放（不管 CJK），不够就丢（table_rows_lt_N）
            if table_pass:
                # 通过：不追加任何 density 原因；如果 reasons 里只有 legacy_uninformative 的否定（reasons 空）→ 放行
                pass
            else:
                reasons.append(f"table_rows_lt_{min_table_rows}")
        elif not reasons:
            # 非 Markdown 表格：走 CJK/ASCII 密度门槛
            cjk_cnt, _tot, cjk_ratio = _cjk_stats(text)
            ascii_words = len(_ALNUM_WORD_RE.findall(text))
            cjk_pass = (cjk_cnt >= min_cjk_chars) and (cjk_ratio >= min_cjk_ratio)
            en_pass = ascii_words >= min_ascii_words
            if not (cjk_pass or en_pass):
                if cjk_cnt < min_cjk_chars:
                    reasons.append(f"cjk_chars_lt_{min_cjk_chars}")
                elif cjk_ratio < min_cjk_ratio:
                    reasons.append(f"cjk_ratio_lt_{min_cjk_ratio}")
                else:
                    reasons.append("ascii_words_low_or_table_invalid")

        if reasons:
            report.l3_chunks_dropped += 1
            for r in reasons:
                report.l3_drop_reasons[r] += 1
            continue
        meta = dict(chunk.metadata)
        first_pass.append(Document(page_content=text, metadata=meta))

    # Step 2：精确去重（按 chunk_level + chunk_hash）。
    # section 和 fine 即使文本完全相同，也承担不同职责：fine 参与召回，
    # section 用于上下文补全，不能互相去重。
    kept_after_exact: list[Document] = []
    if cfg.l3_exact_dedup:
        seen_hashes: dict[tuple[str, str], Document] = {}
        for c in first_pass:
            level = str(c.metadata.get("chunk_level") or "unknown")
            h = build_text_hash(c.page_content)
            key = (level, h)
            if key in seen_hashes:
                report.l3_dedup_exact += 1
                # 合并 dedup 引用计数到第一条
                first = seen_hashes[key]
                first.metadata["dedupe_ref_count"] = int(first.metadata.get("dedupe_ref_count", 0)) + 1
                # 尽量保留同层级内内容更长的那条
                if len(first.page_content) < len(c.page_content):
                    # 新的那条内容更长，把 seen_hashes 里的指针换成这条（同时把老 dedupe_ref_count 迁移）
                    refc = int(first.metadata.get("dedupe_ref_count", 0))
                    new_meta = dict(c.metadata)
                    new_meta["dedupe_ref_count"] = refc
                    new_doc = Document(page_content=c.page_content, metadata=new_meta)
                    seen_hashes[key] = new_doc
                continue
            seen_hashes[key] = c
        kept_after_exact = list(seen_hashes.values())
    else:
        kept_after_exact = list(first_pass)

    # Step 3：SimHash 近似去重（同 chunk_level 组内比较；保留内容更长的那条）
    kept_after_simhash: list[Document] = []
    if cfg.l3_simhash_dedup:
        # 先按 chunk_level 分组（section 不跟 fine 混比，避免父子互相覆盖）
        groups: dict[str, list[Document]] = {}
        for c in kept_after_exact:
            lvl = str(c.metadata.get("chunk_level") or "unknown")
            groups.setdefault(lvl, []).append(c)
        ham_max = int(cfg.l3_simhash_hamming_max)
        for _lvl, group in groups.items():
            # 按内容长度降序：优先保留长文本，短的如果和已保留的任一指纹撞 Hamming<=阈值 就丢
            sorted_group = sorted(group, key=lambda x: len(x.page_content or ""), reverse=True)
            kept_fps: list[tuple[int, Document]] = []
            for c in sorted_group:
                fp = _simhash64(c.page_content or "")
                drop = False
                for kfp, _ in kept_fps:
                    if _hamming64(fp, kfp) <= ham_max:
                        drop = True
                        break
                if drop:
                    report.l3_dedup_simhash += 1
                    continue
                kept_fps.append((fp, c))
            kept_after_simhash.extend(doc for _fp, doc in kept_fps)
    else:
        kept_after_simhash = kept_after_exact

    # Step 4：写 L3 统计 + quality_score（L4 报告开关）
    final_out: list[Document] = []
    for c in kept_after_simhash:
        meta = dict(c.metadata)
        if cfg.report_to_metadata:
            cjk_cnt, _tot, cjk_ratio = _cjk_stats(c.page_content or "")
            ascii_words = len(_ALNUM_WORD_RE.findall(c.page_content or ""))
            table_data = _md_table_data_row_count(c.page_content or "")
            # 0-100 粗评分：CJK密度 50 + 英文词 20 + 表格数据行 20 + 非重复加分 10
            score = 0
            score += min(50, int(cjk_cnt * 2) + int(cjk_ratio * 50))
            score += min(20, ascii_words)
            score += min(20, table_data * 5)
            score += max(0, 10 - int(meta.get("dedupe_ref_count", 0)) * 2)
            meta["quality_score"] = max(0, min(100, score))
            meta["cleaning_cjk_chars"] = cjk_cnt
            meta["cleaning_cjk_ratio"] = round(cjk_ratio, 3)
            meta["cleaning_ascii_words"] = ascii_words
            meta["cleaning_table_data_rows"] = table_data
        final_out.append(Document(page_content=c.page_content, metadata=meta))

    report.l3_chunks_kept = len(final_out)
    report.l3_chunks_dropped = report.l3_chunks_input - report.l3_chunks_kept
    # 补正：dropped 应该等于「密度门槛 drop」+「精确去重 drop」+「近似去重 drop」
    #       但 L3_dropped 实际以 input - kept 为准，上面的 "l3_chunks_dropped 增量 + dedup 计数"作为统计明细
    report.l3_chunks_dropped = max(report.l3_chunks_dropped, report.l3_chunks_input - report.l3_chunks_kept)
    output = final_out
    return output, report


# ======================================================================
# bind_chunk_metadata（保持原有接口 + 兼容质量报告后的 chunks）
# ======================================================================
def bind_chunk_metadata(
    chunks: list[Document],
    *,
    context: ChunkBindingContext,
    build_chunk_id,
) -> tuple[list[Document], list[str]]:
    """给 chunk 绑定稳定源数据，并返回与 Chroma ids 对齐的 chunk_id 列表。"""
    bound_chunks: list[Document] = []
    chunk_ids: list[str] = []

    section_bind_key_to_chunk_id: dict[str, str] = {}
    section_part_to_first_chunk_id: dict[str, str] = {}
    section_local_to_first_part_chunk_id: dict[str, str] = {}
    first_pass_bound: list[tuple[Document, str]] = []

    for idx, chunk in enumerate(chunks):
        chunk_id = build_chunk_id(context.document_id, context.document_version, idx)
        metadata = dict(chunk.metadata)
        metadata.update(
            {
                "chunk_id": chunk_id,
                "document_id": context.document_id,
                "document_version": context.document_version,
                "chunk_index": idx,
                "source_path": context.source_path,
                "content_hash": context.content_hash,
                "chunk_hash": build_text_hash(chunk.page_content),
                "parser_version": context.parser_version,
                "chunking_version": context.chunking_version,
                "embedding_model": context.embedding_model,
                "embedding_version": context.embedding_version,
                "text_source": metadata.get("text_source") or infer_text_source(metadata),
            }
        )
        bind_key = metadata.get("_semantic_section_bind_key")
        if metadata.get("chunk_level") == "section" and bind_key:
            section_bind_key_to_chunk_id[str(bind_key)] = chunk_id
            section_local_id = metadata.get("section_local_id") or ""
            section_part_to_first_chunk_id[str(bind_key)] = chunk_id
            if section_local_id and section_local_id not in section_local_to_first_part_chunk_id:
                section_local_to_first_part_chunk_id[str(section_local_id)] = chunk_id
        first_pass_bound.append((Document(page_content=chunk.page_content, metadata=metadata), chunk_id))

    for doc, chunk_id in first_pass_bound:
        metadata = dict(doc.metadata)
        if metadata.get("chunk_level") == "fine":
            parent_local_id = str(metadata.get("parent_local_id") or "")
            if parent_local_id:
                # 优先精确 bind_key（含 ::part）；勿先查 first_part，否则会把多分段 fine 全打到 ::0
                parent_chunk_id = section_bind_key_to_chunk_id.get(parent_local_id)
                if parent_chunk_id is None and "::" not in parent_local_id:
                    parent_chunk_id = section_bind_key_to_chunk_id.get(f"{parent_local_id}::0")
                if parent_chunk_id is None and "::" not in parent_local_id:
                    parent_chunk_id = section_local_to_first_part_chunk_id.get(parent_local_id)
                if parent_chunk_id:
                    metadata["parent_chunk_id"] = parent_chunk_id
        metadata.pop("_semantic_section_bind_key", None)
        bound_chunks.append(Document(page_content=doc.page_content, metadata=metadata))
        chunk_ids.append(chunk_id)

    return bound_chunks, chunk_ids


# ======================================================================
# 辅助函数（保持不变）
# ======================================================================
def _join_single_cjk_lines(text: str) -> str:
    """合并 OCR 常见的单字竖排断行。"""
    lines = text.split("\n")
    result: list[str] = []
    buffer: list[str] = []

    def flush_buffer() -> None:
        if buffer:
            result.append("".join(buffer))
            buffer.clear()

    for line in lines:
        stripped = line.strip()
        if _SINGLE_CJK_LINE_RE.match(stripped):
            buffer.append(stripped)
            continue
        flush_buffer()
        result.append(line)

    flush_buffer()
    return "\n".join(result)


def _is_informative_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if _SYMBOL_ONLY_RE.match(stripped):
        return False
    if len(stripped) <= 1:
        return False
    return True


# ======================================================================
# L2 切分层提供的导出正则（供 semantic_chunker 复用，避免重复写正则）
# ======================================================================
def strip_l2_noise_line(line: str, *, decor: bool = True, ref_directive: bool = True) -> str | None:
    """L2 单行噪声清理：返回 None 表示该行应丢弃，否则返回原 line（原样不破坏换行/缩进）。

    只做"明显无价值"的整行剥离，不改任何其他行内容。供 semantic_chunker 在 section_buffer 入队前调用。
    """
    s = (line or "").strip()
    if not s:
        return line  # 空行保留（段落需要分行结构）
    if decor and _MD_DECOR_RE.match(s):
        return None
    if ref_directive and _REF_DIRECTIVE_RE.match(s):
        return None
    return line
