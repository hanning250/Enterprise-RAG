"""文件处理工具模块"""

from __future__ import annotations

import hashlib
import os
import re

from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader


def _resolve_env_placeholder(value: str) -> str:
    if not value:
        return ""
    import re

    pattern = re.compile(r"\$\{([A-Z0-9_]+)\}")
    return pattern.sub(lambda match: os.environ.get(match.group(1), ""), value)


def get_file_sha256_hex(filepath: str):
    """计算文件 SHA-256 哈希，返回 64 位十六进制字符串，失败返回 None。"""
    if not os.path.exists(filepath):
        logger.error(f"[SHA256计算]文件{filepath}不存在")
        return
    if not os.path.isfile(filepath):
        logger.error(f"[SHA256计算]路径{filepath}不是文件")
        return

    sha256_obj = hashlib.sha256()
    chunk_size = 4096

    try:
        # 二进制模式读取，避免文本模式换行符转换影响哈希结果
        with open(filepath, "rb") as f:
            while chunk := f.read(chunk_size):
                sha256_obj.update(chunk)
        return sha256_obj.hexdigest()
    except Exception as e:
        logger.error(f"计算文件{filepath}sha256失败，{str(e)}")
        return None


def listdir_with_allowed_type(path: str, allowed_types: tuple[str]):
    """列出目录下指定后缀的所有文件完整路径，返回元组。"""
    files = []

    if not os.path.isdir(path):
        logger.error(f"[lisdir_with_allowed_type]{path}不是文件夹")
        return tuple()

    for f in os.listdir(path):
        if f.endswith(allowed_types):
            files.append(os.path.join(path, f))

    return tuple(files)


def txt_loader(filepath: str) -> list[Document]:
    """加载 TXT 文件为 Document 列表。"""
    return TextLoader(filepath, encoding="utf-8").load()


def _md_normalize_newlines(text: str) -> str:
    """Markdown 文本的轻度规范化：换行 + 段落空行，避免 OS 差异导致 heading/表格行识别失败。

    规范范围（**不会**动标题层级/表格语法/代码块围栏/列表）：
    - \\r\\n / \\r → \\n
    - **非代码块围栏内部**：连续 3 个或以上空行 → 两个空行（保证分隔力度但不出现大段空白导致 heading_stack 误以为到了"新文档"）
    - **代码块围栏 ``` / ~~~ 内部**：完全不做规范化（保留前导空格、连续空行、TAB、缩进，防止破坏代码语义）
    - 非代码块：行尾多余空白 / 全角空格 → 去掉（Markdown 行尾两空格是硬换行，保留 ≥2 空格的尾部）
    - BOM(\\uFEFF) 如果还残留在正文开头 → 去掉（MD 大多从 utf-8 读，但某些编辑器保存会带 BOM）
    """
    if not text:
        return ""
    t = text
    if t.startswith("\ufeff"):
        t = t[1:]
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    # 用状态机先把代码块区间用占位行保住，压缩完再还原
    # 这样 "\n\s*\n(?:\s*\n)+" 这个全局正则不会再误杀代码块里的空行
    lines = t.split("\n")
    PLACEHOLDER: list[str] = []  # 保存原始代码块块内行（包含围栏本身）
    placeholder_idx: list[int] = []  # 保存每个占位符在"规范化后 lines 列表"里的索引

    protected_lines: list[str] = []
    in_fenced_code = False
    fence_marker: str = ""
    i = 0
    while i < len(lines):
        ln = lines[i]
        stripped_left = ln.lstrip()
        is_fence = stripped_left.startswith("```") or stripped_left.startswith("~~~")
        if is_fence and not in_fenced_code:
            # 开一个新围栏：把这一行 + 后续所有直到匹配关围栏的行，整块保存到 PLACEHOLDER
            # 并在 protected_lines 里插一个占位标记，占位索引用 len(PLACEHOLDER)
            block_lines: list[str] = [ln]
            fence_prefix_len = len(ln) - len(stripped_left)
            marker = stripped_left[:3]
            in_fenced_code = True
            fence_marker = marker
            j = i + 1
            closed = False
            while j < len(lines):
                nxt = lines[j]
                block_lines.append(nxt)
                nxt_strip = nxt.lstrip()
                # 匹配关闭围栏：同样以 marker 开头且 ≥3 个 marker 字符（允许后面带语言标签等）
                if (
                    nxt_strip.startswith(fence_marker)
                    and len(nxt_strip) >= 3
                    and all(c == fence_marker[0] for c in nxt_strip[:3])
                    and (len(nxt) - len(nxt_strip)) <= fence_prefix_len + 3
                ):
                    # 严格一点：关闭围栏的缩进一般不大于开围栏的缩进 + 3（防止代码里的字符串 fence 被误判）
                    closed = True
                    j += 1
                    break
                j += 1
            if not closed:
                # 遇到没关闭的围栏（文档结尾处），把剩余行全部算进去
                pass
            # 整块作为一个 PLACEHOLDER 条目保护起来
            PLACEHOLDER.append("\n".join(block_lines))
            placeholder_idx.append(len(PLACEHOLDER) - 1)
            # protected_lines 里塞一个唯一"占位行"，它内部不会出现空行，压缩正则对它是安全的
            protected_lines.append(f"\x00PLACEHOLDER_{len(PLACEHOLDER)-1}\x00")
            i = j
            in_fenced_code = False
            fence_marker = ""
            continue
        protected_lines.append(ln)
        i += 1

    # 在 protected_lines 视图里做 3+ 空行压缩为 2 空行
    compressed_protected = "\n".join(protected_lines)
    compressed_protected = re.sub(r"\n\s*\n(?:\s*\n)+", "\n\n", compressed_protected)

    # 把占位符还原回原始块（整段原始块直接替换）
    lines_after = compressed_protected.split("\n")
    final_lines: list[str] = []
    for ln in lines_after:
        m = re.match(r"^\x00PLACEHOLDER_(\d+)\x00$", ln)
        if m:
            raw_block = PLACEHOLDER[int(m.group(1))]
            # 原始块是用 \n join 起来的一整段，直接按 \n 拆行再塞回 final_lines
            final_lines.extend(raw_block.split("\n"))
            continue
        # 非占位行：再做一次"非代码块行尾空白规范化"
        # 保留 Markdown 硬换行（末尾 ≥2 空格）
        if len(ln) >= 2 and ln.endswith("  ") and not ln.endswith("    "):
            head = ln[:-2]
            final_lines.append(head.rstrip(" \t\u3000") + ln[-2:])
        else:
            final_lines.append(ln.rstrip(" \t\u3000"))
    return "\n".join(final_lines).strip() + "\n"


def md_loader(filepath: str) -> list[Document]:
    """加载 Markdown (.md / .markdown) 文件为 Document 列表。

    设计原则（Phase1）：**原样保留 Markdown 结构**，不做 AST 解析/转换，
    保证下游 semantic_chunker 里的 _HEADING_RE / _extract_markdown_tables / 表格滚动切
    能 100% 复用；只做"文本规范化"：
      1. UTF-8-SIG 自动去 BOM（避免 Windows 记事本保存的 MD 开头出 \ufeff）
      2. \\r\\n / \\r → \\n，段落 3+ 空行压成 2 空行
      3. 代码块围栏内字符不做任何处理
      4. 返回 metadata 里写 { source, md_parser, doc_type, md_sha256 }，
         后续 normalize_documents 会据此 text_source='md' 走完全一致的处理链
    """
    # 先读 bytes 判定 BOM：UTF-8 BOM(EF BB BF) 用 utf-8-sig 打开，其他一律 utf-8
    try:
        with open(filepath, "rb") as f:
            head3 = f.read(3)
        encoding = "utf-8-sig" if head3.startswith(b"\xef\xbb\xbf") else "utf-8"
    except Exception as exc:
        logger.warning(f"[MD加载]{filepath} 读取文件头失败，默认 utf-8 兜底：{exc}")
        encoding = "utf-8"

    try:
        with open(filepath, "r", encoding=encoding) as f:
            raw = f.read()
    except UnicodeDecodeError as exc:
        # 尝试 gbk / cp936（Windows 老 MD 偶尔 gbk 存的）
        logger.warning(f"[MD加载]{filepath} utf-8 解码失败，尝试 gbk 兜底：{exc}")
        with open(filepath, "r", encoding="gbk", errors="ignore") as f:
            raw = f.read()
        encoding = "gbk_fallback"

    normalized = _md_normalize_newlines(raw)
    md_sha256 = get_file_sha256_hex(filepath) or ""
    meta = {
        "source": filepath,
        "doc_type": "md",
        "md_parser": "raw-markdown-v1",
        "md_encoding": encoding,
        # MD 没有稳定的"物理页"概念，这里不写 page，后续靠 section_path 面包屑定位
    }
    if md_sha256:
        meta["md_sha256"] = md_sha256
    logger.info(
        f"[MD加载]{filepath} 解析完成，文本长度={len(normalized)} 编码={encoding}"
    )
    return [Document(page_content=normalized, metadata=meta)]


def _open_pdf_with_password(filepath: str, passwd=None):
    """打开 PDF，兼容加密 PDF。调用方负责 close。"""
    import fitz

    pdf_doc = fitz.open(filepath)
    if passwd:
        if not pdf_doc.authenticate(passwd):
            logger.warning(f"[PDF加载]{filepath} 密码错误或 PDF 未加密")
    return pdf_doc


_VLM_SYSTEM_PROMPT_DETAILED = """你是一个专业的文档内容提取助手。请仔细分析这张图片中的所有内容，并按以下要求输出：

1. 完整提取所有可见文字，不要遗漏任何内容
2. 保持原文的阅读顺序和段落结构
3. 标题用 Markdown 标题格式（# 一级标题、## 二级标题等）
4. 表格用 Markdown 表格格式输出，确保行列对齐
5. 列表用 Markdown 列表格式（有序或无序）
6. 图片和图表的说明文字也要提取
7. 不要添加任何解释、说明或评论，只输出文档内容本身
8. 如果是双栏排版，按左栏从上到下、再右栏从上到下的顺序提取
9. 如果一个大表格被当前页面截断（只显示了一部分行），请照常输出表格内容，不要在表格末尾或开头加任何分割线、"接下页"、"接上页"之类的提示文字。我们会在后续自动做跨页表格合并。
10. 如果页面上存在图片/图表/插图/示意图/流程图/架构图/照片/手绘稿/配色样例等视觉元素，且：
    - 这些元素本身几乎没有可见文字（或文字极少、不足以说明图的含义）；
    - 同时元素周围正文/图注（caption）中也没有充分描述它们的标题或说明；
    请在每个该类视觉元素紧邻的正文位置输出一段「图描述段落」，格式严格固定为：
      【图描述-P<PAGE_NO>-<N>】：<图类型（折线图/柱状图/饼图/散点图/热力图/直方图/流程图/架构图/思维导图/实物照片/手绘示意图/配色样例等）>；<1-3句话描述肉眼可见的关键信息：趋势/最值/占比/结构关系/阶段划分/主色/配色对比等，若涉及时间序列请给出大致阶段和数值区间估计，若涉及结构请给出谁指向谁、谁包含谁、分几块、每块主题>。
    其中 <N> 从 1 开始按这一页从上到下、从左到右的视觉顺序编号；不要杜撰不存在的数值或标注，但允许基于视觉的合理估计。
11. 如果某张图正文里已经有了正式 caption（例如「图 3-2 四阶段推理架构」这种），请把你的【图描述-P<PAGE_NO>-<N>】段直接写在该 caption 后面，紧挨着，不要覆盖或丢弃原 caption。"""

_VLM_SYSTEM_PROMPT_CONCISE = """你是一个文档内容提取助手。请提取这张图片中的核心文字内容：

1. 提取主要的正文和标题，跳过页眉页脚
2. 保持基本的段落结构
3. 表格用简洁的文字描述
4. 只输出内容，不要额外解释"""

# 方案 A 的兜底专用 prompt：
# 当某页 VLM 返回的 page_text 太短（说明这页大概率全是"纯无文字的图/照片/手绘"，OCR 风格的 DETAILED prompt 提取不出来），
# 立刻再用这份 prompt 单独跑一次 VLM，强制让 VLM 按「【图描述】」格式输出视觉描述，避免整张图在向量库里"消失"。
_VLM_DESCRIBE_FALLBACK_PROMPT = """你是一个视觉内容描述助手。请只专注于这张图片中"无法通过文字表达的视觉内容"，严格按要求输出：

1. 忽略页面上能直接用文字读懂的正文段落、标题、表格、页眉页脚（它们会被另一路 OCR 提取）；
2. 只输出对以下视觉元素的结构化描述：
     - 图/图表/示意图/流程图/架构图/思维导图；
     - 照片/手绘稿/样机效果图/配色样例/Logo/商标/印章/签字；
     - 任何"里面几乎没有文字、或文字不足以说明其含义"的图像块；
3. 每一个独立的视觉元素写一段，严格使用以下固定格式（P<PAGE_NO> 由调用方替换为真实页码，这里写 1 即可；N 从 1 开始按从上到下、从左到右顺序）：
      【图描述-P1-N】：<图类型>；<1-3句话的关键视觉信息：趋势/最值/占比/结构关系/阶段划分/主色/配色/主体物体/构图方式/明显标注>。
4. 如果整张页面没有任何上述视觉元素（整页纯文字），直接返回空字符串即可，不要编造。
5. 不要输出任何正文里已有的标题、表格、段落文本，也不要输出任何解释性前缀/后缀。"""


def _get_vlm_prompt(detail_level: str = "detailed") -> str:
    if detail_level == "concise":
        return _VLM_SYSTEM_PROMPT_CONCISE
    return _VLM_SYSTEM_PROMPT_DETAILED


def _get_vlm_describe_fallback_prompt() -> str:
    """方案 A 专用：某页 OCR 结果为空/太短时，强制跑"视觉描述"的 prompt。"""
    return _VLM_DESCRIBE_FALLBACK_PROMPT


# ── 跨页表格合并工具（Markdown 表格规则）──────────────────────────────────────
# Markdown 表格的「表格行」正则：| x | y | ... | 或 :---|:---| 这种对齐线
# 旧正则要求「每一段 :--- 块之间必须是 | 分隔且最后有 |? 结尾」，
# 遇到 | :---: | :------ | ... 这种标准 Markdown 对齐线反而会 match 不上（因为 :---: 冒号是写在破折号两侧，
# 会被旧正则的 ":?-{3,}:?" 吃掉，但末尾的空格要求 " " 就和 " | :" 冲突），
# 这里改成更宽松的规则：每行里有 ≥3 个 "| :?--" 这样的单元，就认定为 Markdown 表格对齐线。
_MD_TABLE_ALIGN_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")
# 再做一个"列数兜底"匹配：一行里有 3+ 个 |，且中间内容由 [- :| ] 组成 → 也算对齐线（兼容 VLM 偶尔手写的奇怪对齐线）
_MD_TABLE_ALIGN_LOOSE_RE = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")
_MD_TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")
_MD_TABLE_LEADING_PIPE = re.compile(r"^\s*\|")


def _count_md_table_pipes(line: str) -> int:
    """粗略统计一行里的 Markdown 表格分隔符 '|' 数量（用于判断是否表格行）。"""
    return line.count("|")


def _looks_like_md_table_header_candidate(lines: list[str], idx: int) -> bool:
    """判断 lines[idx] 是否像一个 Markdown 表格的「表头 + 对齐线」组合。"""
    if idx + 1 >= len(lines):
        return False
    first = lines[idx].rstrip()
    second = lines[idx + 1].rstrip()
    if _MD_TABLE_ALIGN_RE.match(second) or _MD_TABLE_ALIGN_LOOSE_RE.match(second):
        return _MD_TABLE_LEADING_PIPE.match(first) is not None or _count_md_table_pipes(first) >= 2
    return False


def _looks_like_md_table_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if _MD_TABLE_ALIGN_RE.match(s) or _MD_TABLE_ALIGN_LOOSE_RE.match(s):
        return True
    if _MD_TABLE_ROW_RE.match(s):
        return True
    # 兜底：一行里有 ≥3 个 |，且前后没其他明显非表格字符 → 认作表格行
    return _count_md_table_pipes(s) >= 3


def _looks_like_incomplete_md_table_end(lines: list[str]) -> bool:
    """判断一组行的尾部是不是一个「表格被截断，未完待续」。"""
    # 过滤空行，只看最后 5 行
    tail = [ln.rstrip() for ln in lines if ln.strip()]
    if not tail:
        return False
    tail = tail[-5:]
    # 最后一行是表格行 → 高度怀疑未完
    if _looks_like_md_table_line(tail[-1]):
        # 如果最后一行的表格分隔符数 = 倒数第 2 行的数（列数一致）且后面没对齐线 → 99% 被截断
        if len(tail) >= 2:
            last_pipes = _count_md_table_pipes(tail[-1])
            prev_pipes = _count_md_table_pipes(tail[-2])
            _is_align_line = (
                _MD_TABLE_ALIGN_RE.match(tail[-1])
                or _MD_TABLE_ALIGN_LOOSE_RE.match(tail[-1])
            )
            if last_pipes >= 3 and abs(last_pipes - prev_pipes) <= 1 and not _is_align_line:
                return True
        return True
    return False


def _looks_like_incomplete_md_table_start(lines: list[str]) -> bool:
    """判断一组行的开头是不是「表格续页」。

    两种情况都算续页：
    1. 没有表头的续页（第 N 页继续第 N-1 页的表格数据，没有重复表头）
    2. **含重复表头的续页**（VLM/OCR 跨页时经常在每页顶部重新打一遍表头和对齐线，
       这种情况必须也识别成续页，否则重复表头不会被去重，120 行表格会变成 6 份表头堆叠）
    """
    head = [ln.rstrip() for ln in lines if ln.strip()]
    if not head:
        return False
    head = head[:8]  # 放宽到前 8 行去扫，兼容 VLM 每页开头先写一句"第 X 页 / 续上页"的提示行

    for i, ln in enumerate(head):
        # 情况 A：在重复表头之后出现了数据行 → 这是典型的"跨页续页且重复打了表头"
        _is_align_line = _MD_TABLE_ALIGN_RE.match(ln) or _MD_TABLE_ALIGN_LOOSE_RE.match(ln)
        if _looks_like_md_table_line(ln) and not _is_align_line:
            # 不管 i 之前有没有重复表头和对齐线，只要 i 之前的 head 里至少存在一个
            # 「表格行或对齐线或表头」，说明这一页开头整体就是在延续上一页的表格。
            table_context_before = any(
                _looks_like_md_table_line(head[j]) for j in range(i)
            )
            # 或者：前面没有表头，但 i 之前全是"续上页"之类的说明文字（说明文字会被 _looks_like_incomplete_md_table_end 过滤掉），
            # 这里只需要确认"本页开头有表格数据行"，并且上一页 _looks_like_incomplete_md_table_end 是 True，
            # 那两个合起来必然是续页。直接返回 True。
            if table_context_before or True:
                return True
            break
    return False


def _merge_md_table_fragments(prev_lines: list[str], curr_lines: list[str]) -> tuple[list[str], bool]:
    """尝试把上一页表格尾巴和当前页表格续页合并。

    返回 (合并后的完整 lines, 是否发生了合并)。

    合并策略（和 semantic_chunker._extract_markdown_tables 的要求强耦合：一张 Markdown 表格必须是
    「表头 + 对齐线 + N 行数据」连续出现，不能在数据行中间插入"续上页"等非表格文本，否则会被
    _extract_markdown_tables 识别成两张独立的表格，导致第二张表格没有表头/对齐线，
    后续切分会按普通文本硬切，破坏"每块必带表头副本"的保证）。

    合并步骤：
    1. 若 curr 开头出现「重复表头 + 对齐线」（跨页时 VLM 经常每页重新打一遍表头），
       自动跳过这份重复表头，只留 prev 的表头一份；
    2. 把 curr 开头的「（第 X 页，续上页）/ 注：...」之类的非表格说明文字，**不**放到 prev
       的表格数据行中间，而是先临时缓存起来；
    3. 把 curr 剩下的真正数据行，**紧接 prev 最后一个表格数据行之后**写入，让 120 行数据
       作为一张完整的 Markdown 表格呈现（一张表格 120 行，中间不被说明文字打断）；
    4. 表格合并完后，把缓存的说明文字作为页间注释，写在**整段合并后表格的开头位置最前面**
       （保证不破坏「表头 + 对齐线 + 数据行」的块结构）。
    """
    if not _looks_like_incomplete_md_table_end(prev_lines):
        return list(curr_lines), False
    if not _looks_like_incomplete_md_table_start(curr_lines):
        return list(curr_lines), False

    # Step 1：在 head_region 里找重复表头 dup_start；如果有，跳过重复表头那两行
    HEAD_SCAN = 12
    curr = list(curr_lines)
    head_end = min(HEAD_SCAN, len(curr))
    head_region = curr[:head_end]
    tail_region = curr[head_end:]

    dup_start = -1
    for i in range(len(head_region) - 1):
        if _looks_like_md_table_header_candidate(head_region, i):
            dup_start = i
            break

    # 切片 A：dup_start 之前的 head_region 部分（通常就是「（第 N 页，续上页）/ 空行 / 说明文字」）
    # 切片 B：如果有 dup_start，跳过 dup_start 和 dup_start+1（跳过重复表头与对齐线）
    #        之后 head_region 里剩余的行（通常是第 N 页表格真正的前若干条数据行）
    if dup_start >= 0:
        preamble_lines = list(head_region[:dup_start])
        after_dup = list(head_region[dup_start + 2 :])
    else:
        preamble_lines = []
        after_dup = list(head_region)

    # Step 2：把 after_dup 里「开头可能还残留的一些空行 / 说明文字」再剥离一次
    # （比如 dup_start 之前没写"续上页"，写在 after_dup 前面的这种边界情况），
    # 最终留下的纯表格数据行，我们叫 curr_table_lines_start
    curr_table_lines_start: list[str] = []
    preamble_extra: list[str] = []
    for ln in after_dup:
        s = ln.strip()
        # 如果 curr_table_lines_start 还没收集到任何表格行，允许前面塞一些"非表格行到 preamble_extra"
        if not curr_table_lines_start and (
            not s or not _looks_like_md_table_line(s)
        ):
            preamble_extra.append(ln)
            continue
        curr_table_lines_start.append(ln)

    # Step 3：合并逻辑——在 prev_lines 的末尾找到最后一个"真正的表格行"的位置，
    # 把 curr_table_lines_start + tail_region 里的表格数据行插进去；
    # 非表格内容（页尾说明文字）保留在表格行之后；
    # preamble_lines + preamble_extra 缓存到 merged 最前面作为"合并后文档的前置说明"，
    # 注意：不能插到表格数据行之间！
    merged_tail = list(prev_lines)
    insert_pos = len(merged_tail)  # 默认追加到末尾
    for j in range(len(merged_tail) - 1, -1, -1):
        if _looks_like_md_table_line(merged_tail[j]):
            insert_pos = j + 1
            break

    # 新的表格行 = curr_table_lines_start + tail_region 里的所有行（其中会包含最后一页"表格结束"的说明文字）
    # 我们把 tail_region 里的内容也合并进来；如果 tail_region 里也有残留的重复表头，
    # 同样跳过，避免 1~20 行之后再接一份表头
    def _strip_dup_headers_in_tail(tail: list[str]) -> list[str]:
        out: list[str] = []
        i = 0
        while i < len(tail):
            if i + 1 < len(tail) and _looks_like_md_table_header_candidate(tail, i):
                i += 2  # 跳过 表头 + 对齐线
                continue
            out.append(tail[i])
            i += 1
        return out

    new_rows_and_trailing = _strip_dup_headers_in_tail(curr_table_lines_start + list(tail_region))

    # 构造最终 merged 结果：
    # [ 0:insert_pos ] prev_lines 的前半部分（包含第 1 页的表头 + 对齐线 + 1~20 行数据）
    #                 + 中间插入 curr 的续表数据行（21~40、41~60 ... 行）
    #                 + prev_lines 的 insert_pos 之后（若还有内容就是页尾说明文字）
    merged = list(merged_tail[:insert_pos]) + new_rows_and_trailing + list(merged_tail[insert_pos:])

    # Step 4：如果有 preamble（每页开头写的"（第 N 页，续上页）"这种），
    # 把它作为"跨页合并前置的分页注记"追加到合并结果的最**前面**一份就好（只保留最后一页那个也行），
    # 不能放在表格数据行之间，否则会让 _extract_markdown_tables 把表拆成两张。
    preamble_stripped = [ln for ln in (preamble_lines + preamble_extra) if ln.strip()]
    if preamble_stripped:
        # 在结果最前面先放一份分隔符和注记（作为普通段落，不影响后面 120 行表格的连续性），
        # 但如果最前面已经是"## 2025 Q3..."这种标题，注记就放到标题之后、表格之前——
        # 这里统一做法：直接加到 merged 之前就够了，_extract_markdown_tables 会自动把它和第一页
        # 标题当成同一段普通文本 block，不会塞进中间表格行里。
        merged = list(preamble_stripped) + [""] + merged

    if len(merged) == len(prev_lines) + len(curr_lines):
        # 没任何行跳过，说明没有发生合并（只是简单拼接），这种情况不打 cross_page_merged 标记
        # 但实际这个函数被调用时 incomplete_end/incomplete_start 都是 True，所以这里依然返回 True
        pass
    return merged, True


def merge_cross_page_markdown_tables(page_documents: list[Document]) -> list[Document]:
    """跨页表格合并：对 PDF 每页的 VLM 输出做表格续页识别并合并。

    思路：逐页比较「上一页末尾是不是未完表格」和「下一页开头是不是续页」；
    是则把两页的表格 fragment 合并成一个 Markdown 大表格，再作为单个 Document 输出。
    其他非表格文本保持原样。
    """
    if not page_documents:
        return []
    result: list[Document] = []
    buf_doc: Document | None = None
    buf_lines: list[str] = []

    for doc in page_documents:
        text = doc.page_content or ""
        lines = text.split("\n")

        if buf_doc is None:
            buf_doc = doc
            buf_lines = list(lines)
            continue

        merged_lines, ok = _merge_md_table_fragments(buf_lines, lines)
        if ok:
            # 合并成功：把当前页并入 buf_doc，扩展 page 元数据的范围
            buf_lines = merged_lines
            prev_meta = dict(buf_doc.metadata or {})
            curr_meta = dict(doc.metadata or {})
            base_page = prev_meta.get("page") or curr_meta.get("page") or 1
            end_page = curr_meta.get("page") or base_page
            merged_meta = dict(prev_meta)
            merged_meta.update(curr_meta)
            merged_meta["page"] = base_page
            merged_meta["page_start"] = base_page
            merged_meta["page_end"] = end_page
            merged_meta["cross_page_merged_pages"] = sorted(
                set(
                    list(prev_meta.get("cross_page_merged_pages") or [])
                    + [int(base_page), int(end_page)]
                )
            )
            merged_meta["cross_page_merged"] = True
            buf_doc = Document(page_content="\n".join(buf_lines), metadata=merged_meta)
        else:
            # 没有可合并的 → 先把 buf 提交，再用当前页开启新 buf
            result.append(
                Document(
                    page_content="\n".join(ln for ln in buf_lines if ln is not None),
                    metadata=dict(buf_doc.metadata or {}),
                )
            )
            buf_doc = doc
            buf_lines = list(lines)

    if buf_doc is not None:
        result.append(
            Document(
                page_content="\n".join(ln for ln in buf_lines if ln is not None),
                metadata=dict(buf_doc.metadata or {}),
            )
        )

    merged_count = sum(1 for d in result if (d.metadata or {}).get("cross_page_merged"))
    if merged_count:
        logger.info(
            f"[跨页表格合并] 输入 {len(page_documents)} 页 → 输出 {len(result)} 份 Document，"
            f"其中 {merged_count} 份是跨页表格合并后的大节"
        )
    return result


def _call_vlm_raw(
    image_base64: str,
    *,
    model_name: str,
    api_key: str,
    base_url: str,
    prompt: str,
    max_tokens: int = 4096,
) -> str:
    """底层通用 VLM 调用：指定 prompt，拿纯文本输出。

    上层可以用它调用 OCR 风格 prompt / describe 回退 prompt / 其他专用 prompt，
    避免重复写 requests 调用代码。
    """
    import json

    import requests

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        },
                    },
                ],
            }
        ],
        "max_tokens": int(max_tokens),
        "temperature": 0,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
    except Exception as exc:
        logger.error(f"[VLM解析]调用失败：{exc}")
        raise


def _call_vlm_for_page(
    image_base64: str,
    *,
    model_name: str,
    api_key: str,
    base_url: str,
    detail_level: str,
) -> str:
    """调用 VLM 模型解析单页图片（OCR/结构提取风格），返回提取的文本。"""
    return _call_vlm_raw(
        image_base64,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        prompt=_get_vlm_prompt(detail_level),
        max_tokens=4096,
    )


# 方案 A：某页 OCR 结果太短时，认为"这页主要是纯无文字图"
#   - 阈值：< 24 个非空白字符（约 6~12 个汉字），说明 VLM 基本没提取出任何有效正文
#   - 同时：检测不到 【图描述】标签（意味着 prompt 第 10/11 条没生效，或确实是纯无文字）
#   触发：再跑一次 describe 专用 prompt，强制输出视觉描述
_PURE_IMAGE_FALLBACK_MIN_TEXT_CHARS = 24
_PURE_IMAGE_FALLBACK_TAG = "【图描述-"


def _should_run_describe_fallback(ocr_text: str) -> bool:
    """判断这一页 OCR 结果是否需要二次跑 describe fallback。"""
    stripped = (ocr_text or "").strip()
    if not stripped:
        return True
    non_ws = "".join(stripped.split())
    if len(non_ws) < _PURE_IMAGE_FALLBACK_MIN_TEXT_CHARS:
        return True
    if _PURE_IMAGE_FALLBACK_TAG not in stripped:
        # 文本量虽然过了 24 字，但可能是页眉/页脚重复字，没有任何图描述标签
        # 这里不强制回退（避免正常文字页多跑一次），只有"过短"才回退
        return False
    return False


def _patch_figure_page_no(describe_text: str, page_no: int) -> str:
    """把 describe fallback 里写死的 P1 占位符替换成真实页码 P<真实页号>。

    VLM describe prompt 里让它写【图描述-P1-N】，是为了让模型格式不容易跑偏；
    实际返回后我们统一用正则把 P1 替换成这一页真实的 page_no，保证跨多页大图时【图描述-Px-N】的页号是对的。
    """
    if not describe_text:
        return ""
    import re

    # 允许【图描述-P1-1】 / 【图描述-p1-1】 / 【图描述- P1 -1】 这类小偏差
    return re.sub(
        r"【图描述[-\s]*[Pp]\s*1[-\s]*-",
        f"【图描述-P{page_no}-",
        describe_text,
    )



def pdf_loader(filepath: str, passwd=None) -> list[Document]:
    """用 VLM 视觉大模型解析 PDF，每页返回一个 Document。

    注意：返回前会跑一次「跨页表格合并」merge_cross_page_markdown_tables，
    因此跨 5~6 页的大表格不会被拆成 5~6 个碎片，而会被合并成一份带完整行列的 Markdown 大表格 Document。
    跨页合并后的文档会在 metadata 里写 cross_page_merged=True / page_start / page_end / cross_page_merged_pages。
    """
    import base64
    import os

    from utils.config_handler import chroma_conf

    vlm_config = chroma_conf.get("vlm", {})
    model_name = vlm_config.get("model_name", "qwen-vl-plus")
    detail_level = vlm_config.get("detail_level", "detailed")
    base_url = _resolve_env_placeholder(
        str(vlm_config.get("base_url") or os.environ.get("DASHSCOPE_BASE_URL", ""))
    ).strip()
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")

    if not api_key:
        raise RuntimeError(
            "VLM 解析需要 DASHSCOPE_API_KEY 环境变量，请在 .env 中配置"
        )

    pdf_doc = _open_pdf_with_password(filepath, passwd)
    documents: list[Document] = []

    try:
        total_pages = len(pdf_doc)
        for page_idx in range(total_pages):
            page = pdf_doc[page_idx]
            page_no = page_idx + 1
            pix = page.get_pixmap(dpi=200)

            img_bytes = pix.tobytes("png")
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")

            logger.info(
                f"[VLM解析]正在处理 {filepath} 第 {page_no}/{total_pages} 页"
            )

            # 第一次调用：OCR + 结构提取（双栏顺序 / 表格 / 列表 / 已带caption图 的图描述）
            page_text = _call_vlm_for_page(
                img_b64,
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
                detail_level=detail_level,
            )

            # 方案 A 兜底：当某页 OCR 结果为空/过短（说明大概率是纯无文字图/照片/手绘），
            # 强制再跑一次 describe 专用 prompt，单独输出【图描述-P<页>-N】段落
            merged_text = page_text or ""
            describe_fallback_used = False
            describe_text_raw = ""
            if detail_level == "detailed" and _should_run_describe_fallback(page_text):
                logger.info(
                    f"[VLM解析]{filepath} 第 {page_no} 页 OCR 结果过短"
                    f"(非空白字符<{_PURE_IMAGE_FALLBACK_MIN_TEXT_CHARS})，"
                    f"触发纯无文字图 describe 兜底"
                )
                try:
                    describe_text_raw = _call_vlm_raw(
                        img_b64,
                        model_name=model_name,
                        api_key=api_key,
                        base_url=base_url,
                        prompt=_get_vlm_describe_fallback_prompt(),
                        max_tokens=2048,
                    )
                except Exception as exc:
                    logger.warning(
                        f"[VLM解析]{filepath} 第 {page_no} 页 describe fallback 调用失败：{exc}，"
                        f"降级使用 OCR 原文"
                    )
                    describe_text_raw = ""
                describe_text_patched = _patch_figure_page_no(describe_text_raw, page_no=page_no)
                if describe_text_patched.strip():
                    describe_fallback_used = True
                    # 合并策略：把 describe 得到的【图描述...】段，写在这页 OCR 内容的最前面（空一行分隔），
                    # 保证后续 chunker 不会因为 page_text 为空而整页跳过入库
                    if merged_text.strip():
                        merged_text = describe_text_patched.strip() + "\n\n" + merged_text.strip()
                    else:
                        merged_text = describe_text_patched.strip()

            meta = {
                "source": filepath,
                "page": page_no,
                "vlm_ocr": True,
                "vlm_model": model_name,
            }
            if describe_fallback_used:
                meta["vlm_describe_fallback"] = True

            documents.append(
                Document(
                    page_content=merged_text,
                    metadata=meta,
                )
            )
    finally:
        pdf_doc.close()

    # 对每页输出做「跨页大表格合并」：
    # 一张跨 5~6 页的大表格会被合并为单个 Document，不再是 5~6 个独立碎片
    return merge_cross_page_markdown_tables(documents)
