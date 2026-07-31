import { useEffect, useId, useRef, useState } from "react";
import type { ReactNode } from "react";
import type { RetrievedDoc } from "./types";

const CITATION_RE = /【(\d+)】/g;

function sourceLabel(doc: RetrievedDoc): string {
  const m = doc.metadata || {};
  const name =
    (m.file_name as string) ||
    (m.source_file as string) ||
    (m.source as string) ||
    (m.employee_name as string) ||
    "未命名片段";
  const page = m.page ?? m.page_start;
  const base = String(name).split(/[/\\]/).pop() || "片段";
  return page != null ? `${base} · p${page}` : base;
}

type CitationChipProps = {
  n: number;
  doc: RetrievedDoc | undefined;
};

function CitationChip({ n, doc }: CitationChipProps) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLSpanElement>(null);
  const panelId = useId();

  useEffect(() => {
    if (!open) return;
    const onDocClick = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDocClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDocClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  if (!doc) {
    return <span className="cite-miss">【{n}】</span>;
  }

  return (
    <span className="cite-wrap" ref={rootRef}>
      <button
        type="button"
        className={`cite-btn${open ? " open" : ""}`}
        aria-expanded={open}
        aria-controls={panelId}
        title={sourceLabel(doc)}
        onClick={() => setOpen((v) => !v)}
      >
        【{n}】
      </button>
      {open && (
        <span className="cite-pop" id={panelId} role="dialog" aria-label={`引用 ${n}`}>
          <strong>{sourceLabel(doc)}</strong>
          <pre>{doc.page_content || "（无正文）"}</pre>
        </span>
      )}
    </span>
  );
}

type Props = {
  text: string;
  evidence?: RetrievedDoc[];
};

/** 把回答中的【N】渲染为可点击引用；越界编号原样保留。 */
export function AnswerContent({ text, evidence = [] }: Props) {
  const parts: ReactNode[] = [];
  let last = 0;
  let match: RegExpExecArray | null;
  const re = new RegExp(CITATION_RE.source, "g");
  let key = 0;

  while ((match = re.exec(text)) !== null) {
    if (match.index > last) {
      parts.push(text.slice(last, match.index));
    }
    const n = Number(match[1]);
    const doc = Number.isFinite(n) && n >= 1 ? evidence[n - 1] : undefined;
    parts.push(<CitationChip key={`c-${key++}-${n}`} n={n} doc={doc} />);
    last = match.index + match[0].length;
  }
  if (last < text.length) {
    parts.push(text.slice(last));
  }

  return <div className="answer-body">{parts.length ? parts : text}</div>;
}
