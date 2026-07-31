"""标题/FAQ 感知的语义切分模块。

第一版目标：
- fine chunk 参与向量召回；
- section chunk 不参与召回，只作为 parent 上下文补全来源；
- metadata 保留 parent/section 信息，便于检索后恢复完整上下文。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document

from rag.context_budget import estimate_tokens
from rag.document_normalizer import strip_l2_noise_line

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FAQ_RE = re.compile(r"^\s*(\d{1,3})\.\s+\*\*(.+?)\*\*\s*$")
# Word/纯文本常见编号标题：3.5.3 报表审核与提交流程
_NUMBERED_HEADING_RE = re.compile(
    r"^((?:\d+\.){1,5}\d+)\s+([^\s].{0,80})$"
)
_SECTION_SEPARATOR_RE = re.compile(r"^={5,}\s*$")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[。！？!?；;])")

# L2 噪声清理开关：从配置中读；默认 True（保持 rag.yml 中 l2_strip_md_decor / l2_strip_ref_directives 的默认行为一致）
# 注意：此处不走 config 注入，避免修改 _extract_sections 签名；真实生产中可通过 SemanticChunkConfig 加字段控制
_L2_STRIP_DECOR_DEFAULT: bool = True
_L2_STRIP_REF_DEFAULT: bool = True

# Markdown 表格识别正则（和 file_handler 里的跨页表格合并用同一套规则，保持一致）
# 注：和 utils/file_handler.py 里的 _MD_TABLE_ALIGN_*RE 保持同步更新；这里单独重写一份是为了避免循环 import
_MD_TABLE_ALIGN_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")
_MD_TABLE_ALIGN_LOOSE_RE = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")
_MD_TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")
_MD_TABLE_LEADING_PIPE = re.compile(r"^\s*\|")


def _count_md_table_pipes(line: str) -> int:
    return (line or "").count("|")


def _looks_like_md_table_line(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if _MD_TABLE_ALIGN_RE.match(s) or _MD_TABLE_ALIGN_LOOSE_RE.match(s):
        return True
    if _MD_TABLE_ROW_RE.match(s):
        return True
    return _count_md_table_pipes(s) >= 3


def _looks_like_md_table_header_candidate(lines: list[str], idx: int) -> bool:
    if idx + 1 >= len(lines):
        return False
    first = lines[idx].rstrip()
    second = lines[idx + 1].rstrip()
    if _MD_TABLE_ALIGN_RE.match(second) or _MD_TABLE_ALIGN_LOOSE_RE.match(second):
        return _MD_TABLE_LEADING_PIPE.match(first) is not None or _count_md_table_pipes(first) >= 2
    return False


def _extract_markdown_tables(text: str) -> list[str]:
    """把输入文本拆分成「普通段落块」和「Markdown 表格块」的混合列表。

    每个 Markdown 表格（含表头+对齐线+N 行）整体作为一个原子字符串返回，
    不会被 _iter_semantic_blocks 拆成「每行一个 block」打包，从而避免 200/900 字预算
    从表格行中间硬切；超长表格（行数过多）会在下游 pack 阶段按「表头 + N 行」滚动切片，
    每个切片都自带完整表头副本。
    """
    lines = text.split("\n")
    blocks: list[str] = []
    buf_normal: list[str] = []
    i = 0

    def flush_normal() -> None:
        if buf_normal:
            t = "\n".join(buf_normal).strip()
            if t:
                blocks.append(t)
            buf_normal.clear()

    while i < len(lines):
        line = lines[i]
        # 情况 1：遇到表头（下一行是对齐线）→ 开启一个表格块
        if _looks_like_md_table_header_candidate(lines, i):
            flush_normal()
            table_lines = [lines[i], lines[i + 1]]
            i += 2
            while i < len(lines) and _looks_like_md_table_line(lines[i]):
                table_lines.append(lines[i])
                i += 1
            blocks.append("\n".join(table_lines))
            continue
        # 情况 2：单独遇到对齐线（但前一行不是标准表头），或者一行像表格但没有前导表头
        #        → 当普通行处理，避免误判
        buf_normal.append(line)
        i += 1

    flush_normal()
    return blocks or [text.strip()] if text.strip() else []


def _split_huge_markdown_table_by_rows(
    md_table_text: str,
    *,
    max_chars: int,
    max_tokens: int,
    use_chars: bool,
) -> list[str]:
    """超大 Markdown 表格（行数很多）→ 按「表头副本 + N 行数据」滚动切片。

    每个切片**前两行**都会复制完整的「表头 + 对齐线」，保证被切出去的 chunk 读起来行列名依然对得上，
    不会出现只有数据行、不知道每列是什么的情况。

    说明：之前版本把「表头预算」提前扣掉，再用剩余预算打包数据行，最后用 ``[header, align, *current]``
    拼切片，理论上每个切片应该都有表头。但实测会丢表头——原因是 residual_budget 计算时用的是
    ``(header+"\\n"+align)`` 的 size，而实际切片里还有若干换行符、行开头的空白字符，打包时偶尔会超，
    导致部分切片被上层 _pack_blocks / _split_by_budget 再次按字符数硬切（从而砍掉表头）。

    现在改成：每次拼切片时，严格用「整体 size（header+align+data） ≤ max_chars/max_tokens」
    判定是否能装下，从根本上避免二次硬切。若单条数据行已经超预算，兜底：每行一个切片（自带表头）。
    """
    lines = md_table_text.split("\n")
    if len(lines) < 4:  # 至少表头+对齐线+2 行数据才算"大"
        return [md_table_text] if md_table_text.strip() else []
    header = lines[0]
    align = lines[1]
    data_rows = [r for r in lines[2:] if r.strip()]

    def _size(text: str) -> int:
        return len(text) if use_chars else estimate_tokens(text)

    slices: list[str] = []
    current: list[str] = []

    def current_text() -> str:
        if not current:
            return ""
        return "\n".join([header, align, *current]).strip()

    for row in data_rows:
        candidate = "\n".join([header, align, *current, row]).strip()
        # 判断 candidate 是否超预算；如果 current 为空，说明还没开始装，这行必须先装进去（哪怕超预算，后续兜底）
        if current and _size(candidate) > (max_chars if use_chars else max_tokens):
            slices.append(current_text())
            current = [row]
        else:
            current.append(row)
    if current:
        slices.append(current_text())

    # 兜底：如果没有任何切片（数据行极少或打包异常），或者个别切片仍然超预算导致二次硬切概率高，
    # 对超预算切片再按逐条数据行兜底切一下，确保**所有切片**都自带表头副本
    safe_slices: list[str] = []
    for s in slices:
        if _size(s) <= (max_chars if use_chars else max_tokens):
            safe_slices.append(s)
            continue
        # 超预算：拆成 1 行一个切片（每个都带表头副本）
        lns = [ln for ln in s.split("\n") if ln.strip()]
        # lns 里已经自带 header+align+数据行了
        hdr_line = lns[0] if _looks_like_md_table_line(lns[0]) else header
        align_line = (
            lns[1]
            if len(lns) > 1
            and (
                _MD_TABLE_ALIGN_RE.match(lns[1].strip())
                or _MD_TABLE_ALIGN_LOOSE_RE.match(lns[1].strip())
            )
            else align
        )
        for data_line in lns[2:] or lns:
            # 如果 data_line 本身就是表头/对齐线，跳过
            s_data = data_line.strip()
            if (
                s_data == hdr_line.strip()
                or _MD_TABLE_ALIGN_RE.match(s_data)
                or _MD_TABLE_ALIGN_LOOSE_RE.match(s_data)
            ):
                continue
            safe_slices.append("\n".join([hdr_line.rstrip(), align_line.rstrip(), data_line.rstrip()]).strip())
    result = [s for s in safe_slices if s.strip()]
    return result if result else ([md_table_text.strip()] if md_table_text.strip() else [])


@dataclass(frozen=True)
class SemanticChunkConfig:
    """语义切分参数。

    同时支持 tokens 预算和字符数预算；谁设置就用谁，都没设置时按默认 tokens 走。
    优先读 *_chars 字段（和项目 chroma.yml 默认字符数匹配，用户口头上说的 200/900 就是字符数）。
    """

    max_section_tokens: int = 900
    max_fine_tokens: int = 240
    max_section_chars: int | None = None
    max_fine_chars: int | None = None
    allow_section_partition: bool = True

    @property
    def _use_chars(self) -> bool:
        return (self.max_section_chars is not None) and (self.max_fine_chars is not None)

    def _budget_ok(self, *, for_section: bool, value: str | None = None, n_tokens: int = 0, n_chars: int = 0) -> bool:
        if for_section:
            if self.max_section_chars is not None:
                return (n_chars if value is None else len(value)) <= self.max_section_chars
            return (n_tokens if value is None else estimate_tokens(value)) <= self.max_section_tokens
        if self.max_fine_chars is not None:
            return (n_chars if value is None else len(value)) <= self.max_fine_chars
        return (n_tokens if value is None else estimate_tokens(value)) <= self.max_fine_tokens

    def _split_long(self, text: str, *, for_section: bool) -> list[str]:
        if for_section and self.max_section_chars is not None:
            max_chars = int(self.max_section_chars)
            return [text[i : i + max_chars].strip() for i in range(0, len(text), max_chars) if text[i : i + max_chars].strip()]
        if (not for_section) and self.max_fine_chars is not None:
            max_chars = int(self.max_fine_chars)
            return [text[i : i + max_chars].strip() for i in range(0, len(text), max_chars) if text[i : i + max_chars].strip()]
        max_chars = int(self.max_section_tokens * 2) if for_section else max(200, int(self.max_fine_tokens * 2))
        return [text[i : i + max_chars].strip() for i in range(0, len(text), max_chars) if text[i : i + max_chars].strip()]


@dataclass(frozen=True)
class _Section:
    title: str
    path: list[str]
    content: str
    source_metadata: dict


def semantic_split_documents(
    documents: list[Document],
    *,
    config: SemanticChunkConfig | None = None,
) -> list[Document]:
    """把 loader 输出的文档切成 section/fine 两级 chunk。

    输出的 chunk 分两类（通过 metadata["chunk_level"] 区分）：
    - "section": 粗粒 section chunk（900 字符），会写入 Chroma 但不参与召回，只作为 ContextExpander 的 parent。
                 输出时 metadata 写入 "_semantic_section_bind_key" = parent_local_id（section_local_id），
                 由 vector_store 的 bind_chunk_metadata 在分配完 chunk_id 后，把 fine chunk 的 parent_chunk_id 指向对应 section chunk 的 chunk_id。
    - "fine"   : 细粒 fine chunk（200 字符），参与召回，输出时 metadata 带 parent_local_id 对应 section 的绑定键。

    注意：工资表等多 Document 合入同一文件时，每人都会产生 section-00000；
    parent_local_id 必须带上员工/文档身份后缀，否则 bind 阶段会全部撞到第一个 section。
    长节再切成多个 section_part 时，fine 的 parent_local_id 必须带 ::part_index，
    指向内容所在段，禁止一律绑到 ::0。
    """
    cfg = config or SemanticChunkConfig()
    chunks: list[Document] = []

    for doc_index, doc in enumerate(documents):
        identity = _section_identity_suffix(doc.metadata or {}, doc_index)
        for section_index, section in enumerate(_extract_sections(doc)):
            parent_local_id = f"section-{section_index:05d}__{identity}"
            section_meta = dict(section.source_metadata)
            section_meta.update(
                {
                    "chunk_level": "section",
                    "parent_local_id": "",
                    "section_local_id": parent_local_id,
                    "_semantic_section_bind_key": parent_local_id,  # 给 vector_store 建 ID 映射用
                    "section_title": section.title,
                    "section_path": " > ".join(section.path),
                    "section_index": section_index,
                }
            )

            # -------- section 级切分（粗粒 900）：Markdown 表格按行滚动切分，绝不从行中间截断 --------
            if cfg.allow_section_partition:
                section_texts = _split_section_content_with_table_awareness(
                    section.content,
                    max_tokens=cfg.max_section_tokens,
                    max_chars=cfg.max_section_chars,
                )
            else:
                section_texts = [section.content.strip()] if section.content.strip() else []

            for section_part_index, section_text in enumerate(section_texts):
                meta = dict(section_meta)
                meta["section_part_index"] = section_part_index
                meta["section_part_count"] = len(section_texts)
                # 多分段时每段独立 bind_key；fine 必须挂到「内容所在」那一段，不能一律 ::0
                if len(section_texts) > 1:
                    meta["_semantic_section_bind_key"] = f"{parent_local_id}::{section_part_index}"
                chunks.append(Document(page_content=section_text, metadata=meta))

            # -------- fine 级切分（细粒 200）：表格按「表头副本 + N 行」滚动切分，每块自带完整表头 --------
            fine_texts = _split_fine_chunks_with_table_awareness(
                section.content,
                max_tokens=cfg.max_fine_tokens,
                max_chars=cfg.max_fine_chars,
            )
            for fine_index, fine_text in enumerate(fine_texts):
                part_index = _best_section_part_index(fine_text, section_texts)
                if len(section_texts) > 1:
                    fine_parent_id = f"{parent_local_id}::{part_index}"
                else:
                    fine_parent_id = parent_local_id
                meta = dict(section.source_metadata)
                meta.update(
                    {
                        "chunk_level": "fine",
                        "parent_local_id": fine_parent_id,
                        "section_local_id": parent_local_id,
                        "section_title": section.title,
                        "section_path": " > ".join(section.path),
                        "section_index": section_index,
                        "section_part_index": part_index,
                        "section_part_count": len(section_texts),
                        "fine_index": fine_index,
                    }
                )
                chunks.append(Document(page_content=fine_text, metadata=meta))

    return chunks


def _best_section_part_index(fine_text: str, section_texts: list[str]) -> int:
    """把 fine 归属到内容重叠最高的 section_part（根治一律绑 ::0）。"""
    if not section_texts:
        return 0
    if len(section_texts) == 1:
        return 0
    fine = (fine_text or "").strip()
    if not fine:
        return 0
    for i, part in enumerate(section_texts):
        if fine and fine in (part or ""):
            return i
    best_i = 0
    best_score = -1.0
    for i, part in enumerate(section_texts):
        score = _char_overlap_ratio(fine, part or "")
        if score > best_score:
            best_score = score
            best_i = i
    return best_i


def _char_overlap_ratio(needle: str, haystack: str) -> float:
    """粗粒度字符覆盖率：用 needle 的滑动窗口统计有多少落在 haystack 里。"""
    n = (needle or "").strip()
    h = (haystack or "").strip()
    if not n or not h:
        return 0.0
    if n in h:
        return 1.0
    window = 24 if len(n) >= 24 else max(8, len(n) // 2 or 1)
    if len(n) <= window:
        return 1.0 if n in h else 0.0
    hits = 0
    total = 0
    step = max(1, window // 2)
    for start in range(0, len(n) - window + 1, step):
        total += 1
        if n[start : start + window] in h:
            hits += 1
    return hits / total if total else 0.0


def _section_identity_suffix(meta: dict, doc_index: int) -> str:
    """为 parent_local_id 生成批内唯一后缀，避免多人同文件撞 section-00000。"""
    emp_id = str(meta.get("employee_id") or "").strip()
    if emp_id:
        return f"emp-{emp_id}"
    emp_name = str(meta.get("employee_name") or "").strip()
    if emp_name:
        return f"name-{emp_name}"
    source = str(meta.get("source_file") or meta.get("source") or "").strip()
    if source:
        return f"doc-{doc_index}-{Path(source).name}"
    return f"doc-{doc_index}"


def _extract_sections(doc: Document) -> list[_Section]:
    """按 Markdown 标题和 FAQ 问题行抽取稳定语义 section。"""
    lines = [line.rstrip() for line in doc.page_content.split("\n")]
    sections: list[_Section] = []
    heading_stack: list[tuple[int, str]] = []
    current_title = "全文"
    current_path: list[str] = [current_title]
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        content = "\n".join(line for line in buffer).strip()
        if content:
            sections.append(
                _Section(
                    title=current_title,
                    path=list(current_path),
                    content=content,
                    source_metadata=dict(doc.metadata),
                )
            )
        buffer = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line or _SECTION_SEPARATOR_RE.match(line):
            if buffer and buffer[-1] != "":
                buffer.append("")
            continue

        # L2：section_buffer 入队前剥离装饰线和纯指引行（不破坏行缩进；返回 None 直接丢）
        stripped = strip_l2_noise_line(
            raw_line,
            decor=_L2_STRIP_DECOR_DEFAULT,
            ref_directive=_L2_STRIP_REF_DEFAULT,
        )
        if stripped is None:
            if buffer and buffer[-1] != "":
                buffer.append("")
            continue

        heading_match = _HEADING_RE.match(line)
        faq_match = _FAQ_RE.match(line)
        numbered_match = None if heading_match or faq_match else _NUMBERED_HEADING_RE.match(line)
        if heading_match or faq_match or numbered_match:
            flush()
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
            elif numbered_match:
                # 3.5.3 → 点号个数近似标题层级（最多 6）
                num = numbered_match.group(1)
                level = min(6, num.count(".") + 1)
                title = f"{num} {numbered_match.group(2).strip()}"
            else:
                level = 6
                title = faq_match.group(2).strip() if faq_match else line

            heading_stack[:] = [(lvl, text) for lvl, text in heading_stack if lvl < level]
            heading_stack.append((level, title))
            current_title = title
            current_path = [text for _, text in heading_stack] or [title]
            buffer.append(stripped if stripped else line)
            continue

        buffer.append(stripped)

    flush()
    return sections or [
        _Section(
            title="全文",
            path=["全文"],
            content=doc.page_content,
            source_metadata=dict(doc.metadata),
        )
    ]


def _split_fine_chunks(text: str, *, max_tokens: int, max_chars: int | None = None) -> list[str]:
    blocks = _iter_semantic_blocks(text)
    return _pack_blocks(blocks, max_tokens=max_tokens, max_chars=max_chars)


def _split_section_content_with_table_awareness(
    text: str,
    *,
    max_tokens: int,
    max_chars: int | None,
) -> list[str]:
    """section 粗粒切分：对 Markdown 表格优先按行滚动切片，每个切片自带完整表头+对齐线。

    非表格段落继续走原本的 _split_by_budget（按句子打包，超了再切）。
    """
    result: list[str] = []
    for block in _extract_markdown_tables(text):
        if not block.strip():
            continue
        lines = block.split("\n")
        if len(lines) >= 3 and _looks_like_md_table_header_candidate(lines, 0):
            # Markdown 表格：表头副本 + N 行滚动切
            result.extend(
                _split_huge_markdown_table_by_rows(
                    block,
                    max_chars=max_chars if max_chars is not None else 0,
                    max_tokens=max_tokens,
                    use_chars=(max_chars is not None),
                )
            )
        else:
            result.extend(_split_by_budget(block, max_tokens=max_tokens, max_chars=max_chars))
    return [t for t in result if t.strip()]


def _split_fine_chunks_with_table_awareness(
    text: str,
    *,
    max_tokens: int,
    max_chars: int | None,
) -> list[str]:
    """fine 细粒切分：对 Markdown 表格优先按行滚动切片，每个切片自带完整表头+对齐线。

    非表格段落继续走原本的 _split_fine_chunks（按段落/行/句子打包，超了再切）。
    """
    result: list[str] = []
    for block in _extract_markdown_tables(text):
        if not block.strip():
            continue
        lines = block.split("\n")
        if len(lines) >= 3 and _looks_like_md_table_header_candidate(lines, 0):
            result.extend(
                _split_huge_markdown_table_by_rows(
                    block,
                    max_chars=max_chars if max_chars is not None else 0,
                    max_tokens=max_tokens,
                    use_chars=(max_chars is not None),
                )
            )
        else:
            result.extend(_split_fine_chunks(block, max_tokens=max_tokens, max_chars=max_chars))
    return [t for t in result if t.strip()]


def _iter_semantic_blocks(text: str) -> Iterable[str]:
    # Step 1：先按「表格 / 普通段落」拆分，Markdown 表格作为一个原子块进入后续流程，
    # 这样 200/900 字打包不会从表格行中间切一刀，避免内容撕裂。
    for block in _extract_markdown_tables(text):
        if not block.strip():
            continue
        # 单块本身就是表格 → 当作一个 block 返回；它如果超预算，_pack_blocks 会按 _split_by_budget 走表格切分，
        # 表格切分会按「表头副本 + N 行数据」滚动切片，不会从中间硬切。
        first_line = block.lstrip().split("\n", 1)[0] if block else ""
        if _looks_like_md_table_header_candidate(block.split("\n"), 0) or _looks_like_md_table_line(first_line) and _count_md_table_pipes(block.split("\n", 2)[0]) >= 3:
            yield block
            continue
        # 普通段落：按「段落 → 行 → 句子」三级拆分（原有逻辑保持不变）
        for paragraph in re.split(r"\n\s*\n", block):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            lines = [line.strip() for line in paragraph.split("\n") if line.strip()]
            if len(lines) > 1:
                for line in lines:
                    yield line
                continue
            for sentence in _split_sentences(paragraph):
                yield sentence


def _split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_BOUNDARY_RE.split(text) if part.strip()]
    return parts or [text.strip()]


def _pack_blocks(blocks: Iterable[str], *, max_tokens: int, max_chars: int | None = None) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []

    def current_text() -> str:
        return "\n".join(current).strip()

    def _over_budget(value: str) -> bool:
        if max_chars is not None:
            return len(value) > max_chars
        return estimate_tokens(value) > max_tokens

    for block in blocks:
        # 单个 block 本身就超预算：先 flush current，然后按预算切分该 block
        if _over_budget(block):
            if current:
                chunks.append(current_text())
                current = []
            chunks.extend(_split_by_budget(block, max_tokens=max_tokens, max_chars=max_chars))
            continue

        candidate = "\n".join([*current, block]).strip()
        if current and _over_budget(candidate):
            chunks.append(current_text())
            current = [block]
        else:
            current.append(block)

    if current:
        chunks.append(current_text())
    return [chunk for chunk in chunks if chunk]


def _split_by_budget(text: str, *, max_tokens: int, max_chars: int | None = None) -> list[str]:
    def _within(text_value: str) -> bool:
        if max_chars is not None:
            return len(text_value) <= max_chars
        return estimate_tokens(text_value) <= max_tokens

    if _within(text):
        return [text.strip()] if text.strip() else []

    # 【Markdown 表格优先】：若文本本身就是一张 Markdown 表格，按「表头副本 + N 行」滚动切片，
    # 绝不从行中间硬切；每个被切出的 chunk 都会自带一份完整的表头和对齐线，保证行列名可读
    lines = text.split("\n")
    if len(lines) >= 3 and _looks_like_md_table_header_candidate(lines, 0):
        return _split_huge_markdown_table_by_rows(
            text,
            max_chars=max_chars if max_chars is not None else 0,
            max_tokens=max_tokens,
            use_chars=(max_chars is not None),
        )

    sentence_parts: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        sentence_parts.extend(_split_sentences(line))

    if len(sentence_parts) == 1:
        if max_chars is not None:
            max_split = max(200, int(max_chars))
        else:
            max_split = max(200, max_tokens * 2)
        return _split_long_text(sentence_parts[0], max_chars=max_split)
    return _pack_blocks(sentence_parts, max_tokens=max_tokens, max_chars=max_chars)


def _split_long_text(text: str, *, max_chars: int) -> list[str]:
    return [text[i : i + max_chars].strip() for i in range(0, len(text), max_chars) if text[i : i + max_chars].strip()]