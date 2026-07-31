import { useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent } from "react";
import { AnswerContent } from "./AnswerContent";
import { getInternalAuth, healthCheck, queryRag, setInternalAuth } from "./api/rag";
import type { ChatMessage, DataScope, Identity, QueryTrace } from "./types";
import "./index.css";

const STORAGE_KEY = "hanning-kb-identity-v1";

/** 与 data/员工工资表_*.xlsx 中「部门」列对齐 */
const DEPARTMENT_OPTIONS = [
  "技术部",
  "产品部",
  "运营部",
  "市场部",
  "销售部",
  "人事部",
  "财务部",
  "行政部",
] as const;

/** 与工资表「岗位」列对齐 */
const POSITION_OPTIONS = [
  "架构师",
  "高级工程师",
  "中级工程师",
  "产品总监",
  "产品经理",
  "产品专员",
  "运营经理",
  "运营专员",
  "内容运营",
  "市场经理",
  "市场专员",
  "品牌策划",
  "销售经理",
  "销售主管",
  "销售代表",
  "HR经理",
  "招聘主管",
  "HR专员",
  "财务经理",
  "会计",
  "出纳",
  "行政经理",
  "行政专员",
  "前台",
] as const;

const SCOPE_OPTIONS: { value: DataScope; label: string }[] = [
  { value: "self", label: "本人" },
  { value: "department", label: "本部门" },
  { value: "company", label: "全公司" },
];

const SCOPE_LABEL: Record<string, string> = Object.fromEntries(
  SCOPE_OPTIONS.map((s) => [s.value, s.label])
);

const ROLE_LABEL: Record<string, string> = {
  employee: "员工",
  manager: "经理",
  hr_admin: "HR 管理员",
  finance_admin: "财务管理员",
  admin: "系统管理员",
};

/** 按工资表岗位推断 ACL 角色，便于联调权限 */
function inferAclRoles(position: string): string[] {
  const p = position.trim();
  // 财务类必须先于「经理」：否则「财务经理」会被误判成 manager
  if (/HR|人事|招聘/.test(p)) return ["hr_admin"];
  if (/财务|会计|出纳/.test(p)) return ["finance_admin"];
  if (/经理|总监|主管|架构师/.test(p)) return ["manager"];
  return ["employee"];
}

/** 把中文角色展示名 / 历史脏数据规范成后端 ACL 认识的 code */
function normalizeAclRoles(roles: string[] | undefined, position: string): string[] {
  const mapped = (roles || [])
    .map((r) => {
      const s = (r || "").trim();
      if (!s) return "";
      if (s === "财务管理员" || s === "finance_admin") return "finance_admin";
      if (s === "HR 管理员" || s === "HR管理员" || s === "hr_admin") return "hr_admin";
      if (s === "系统管理员" || s === "admin") return "admin";
      if (s === "经理" || s === "manager") return "manager";
      if (s === "员工" || s === "employee") return "employee";
      return s;
    })
    .filter(Boolean);
  const known = new Set([
    "employee",
    "manager",
    "hr_admin",
    "finance_admin",
    "admin",
  ]);
  const valid = mapped.filter((r) => known.has(r));
  // 无效/空 roles 时按岗位重推，避免侧栏看着像财务管理员、请求却按 employee 过滤
  if (!valid.length) return inferAclRoles(position);
  return valid;
}

function roleLabel(code: string): string {
  return ROLE_LABEL[code] || code;
}

const defaultIdentity: Identity = {
  user_id: "EMP009",
  user_name: "王刚",
  department: "技术部",
  position: "架构师",
  roles: ["manager"],
  data_scope: "self",
};

function loadIdentity(): Identity {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return defaultIdentity;
    const parsed = { ...defaultIdentity, ...JSON.parse(raw) } as Identity;
    if (!parsed.position) {
      parsed.position = defaultIdentity.position;
    }
    // 旧版把 user_id 存成 u001 时，纠正为工号默认
    if (!/^EMP/i.test(parsed.user_id)) {
      parsed.user_id = defaultIdentity.user_id;
    }
    if (!parsed.roles?.length) {
      parsed.roles = inferAclRoles(parsed.position);
    } else {
      parsed.roles = normalizeAclRoles(parsed.roles, parsed.position);
    }
    return parsed;
  } catch {
    return defaultIdentity;
  }
}

export default function App() {
  const [identity, setIdentity] = useState<Identity>(loadIdentity);
  const [authSecret, setAuthSecret] = useState(getInternalAuth());
  const [health, setHealth] = useState<"unknown" | "ok" | "down">("unknown");
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [lastTrace, setLastTrace] = useState<QueryTrace | null>(null);
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setInternalAuth(authSecret);
  }, [authSecret]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(identity));
  }, [identity]);

  useEffect(() => {
    healthCheck()
      .then((r) => setHealth(r.status === "ok" ? "ok" : "down"))
      .catch(() => setHealth("down"));
  }, []);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, busy]);

  const identitySummary = useMemo(() => {
    const acl = roleLabel(identity.roles[0] || "employee");
    const scope = SCOPE_LABEL[identity.data_scope] || identity.data_scope;
    return `${identity.user_name || identity.user_id} · ${identity.department} · ${identity.position} · ${acl} · ${scope}`;
  }, [identity]);

  function patchIdentity(patch: Partial<Identity>) {
    setIdentity((prev) => {
      const next = { ...prev, ...patch };
      if (patch.position != null) {
        next.roles = inferAclRoles(next.position);
      }
      return next;
    });
  }

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    const q = input.trim();
    if (!q || busy) return;

    // 每次提问都按岗位重算 roles，避免 localStorage 脏角色导致「界面像财务、ACL 当员工」
    const nextIdentity: Identity = {
      ...identity,
      roles: inferAclRoles(identity.position),
    };
    setIdentity(nextIdentity);

    const userMsg: ChatMessage = {
      id: `u-${Date.now()}`,
      role: "user",
      content: q,
    };
    setMessages((m) => [...m, userMsg]);
    setInput("");
    setBusy(true);

    try {
      const res = await queryRag(q, nextIdentity);
      const assistant: ChatMessage = {
        id: `a-${Date.now()}`,
        role: "assistant",
        content: res.answer || "（空回答）",
        evidence: res.evidence_docs,
        trace: res.trace,
      };
      setMessages((m) => [...m, assistant]);
      setLastTrace(res.trace);
    } catch (err) {
      const detail = err instanceof Error ? err.message : "请求失败";
      setMessages((m) => [
        ...m,
        {
          id: `e-${Date.now()}`,
          role: "assistant",
          content: detail,
          error: detail,
        },
      ]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">韩宁知识助手</div>
          <div className="brand-sub">企业内部知识库 · 工资 / 制度 / 财报问答</div>
        </div>
        <div className="top-actions">
          <span className={`pill ${health === "ok" ? "ok" : ""}`}>
            <span className="dot" />
            {health === "ok" ? "服务在线" : health === "down" ? "服务不可达" : "检测中"}
          </span>
          <span className="pill">{identitySummary}</span>
        </div>
      </header>

      <div className="shell">
        <section className="main">
          <div className="hero-chat">
            <div className="hero-line">
              <h1>问公司里的事，答有据可查。</h1>
              <p>基于当前身份做权限过滤；回答中的【N】可点击核对原文来源。</p>
            </div>

            <div className="messages" ref={listRef}>
              {messages.length === 0 && (
                <div className="empty">
                  <strong>从一条问题开始</strong>
                  例如：「我2026年3月个税扣多少？」或「年假怎么规定？」
                </div>
              )}
              {messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`bubble ${msg.role}${msg.error ? " error" : ""}`}
                >
                  {msg.role === "assistant" && !msg.error ? (
                    <AnswerContent text={msg.content} evidence={msg.evidence} />
                  ) : (
                    msg.content
                  )}
                  {msg.role === "assistant" && msg.trace && (
                    <div className="meta-row">
                      <span className="chip">
                        权限过滤 {msg.trace.acl_filtered_count} ·{" "}
                        {msg.trace.retriever_used === "hybrid"
                          ? "混合检索"
                          : msg.trace.retriever_used}
                        {msg.trace.cache_hit ? " · 缓存命中" : ""}
                      </span>
                    </div>
                  )}
                </div>
              ))}
              {busy && (
                <div className="bubble assistant">
                  正在检索与生成（检索通常 &lt;2s，生成可能需数十秒）…
                </div>
              )}
            </div>
          </div>

          <form className="composer" onSubmit={onSubmit}>
            <div className="composer-box">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="输入问题，Enter 发送（Shift+Enter 换行）"
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    void onSubmit(e);
                  }
                }}
              />
              <button className="primary-btn" type="submit" disabled={busy || !input.trim()}>
                发送
              </button>
            </div>
          </form>
        </section>

        <aside className="side">
          <div className="side-section">
            <h2>身份（开发联调）</h2>
            <p className="hint">
              工号 / 部门 / 岗位与工资表字段对齐；权限角色由岗位自动推断。
            </p>
            <div className="field">
              <label>内部鉴权密钥</label>
              <input
                value={authSecret}
                onChange={(e) => setAuthSecret(e.target.value)}
                placeholder="与后端 AUTH_TRUSTED_IDENTITY_SECRET 一致"
                type="password"
                autoComplete="off"
              />
            </div>
            <div className="field">
              <label>工号</label>
              <input
                value={identity.user_id}
                onChange={(e) => patchIdentity({ user_id: e.target.value.trim() })}
                placeholder="如 EMP009"
              />
            </div>
            <div className="field">
              <label>姓名</label>
              <input
                value={identity.user_name}
                onChange={(e) => patchIdentity({ user_name: e.target.value })}
              />
            </div>
            <div className="field">
              <label>部门</label>
              <select
                value={
                  DEPARTMENT_OPTIONS.includes(
                    identity.department as (typeof DEPARTMENT_OPTIONS)[number]
                  )
                    ? identity.department
                    : DEPARTMENT_OPTIONS[0]
                }
                onChange={(e) => patchIdentity({ department: e.target.value })}
              >
                {DEPARTMENT_OPTIONS.map((d) => (
                  <option key={d} value={d}>
                    {d}
                  </option>
                ))}
              </select>
            </div>
            <div className="field">
              <label>岗位</label>
              <select
                value={
                  POSITION_OPTIONS.includes(
                    identity.position as (typeof POSITION_OPTIONS)[number]
                  )
                    ? identity.position
                    : POSITION_OPTIONS[0]
                }
                onChange={(e) => patchIdentity({ position: e.target.value })}
              >
                {POSITION_OPTIONS.map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            </div>
            <div className="field">
              <label>权限角色（自动）</label>
              <input
                value={roleLabel(identity.roles[0] || "employee")}
                readOnly
                title="由岗位自动推断，用于 ACL"
              />
            </div>
            <div className="field">
              <label>数据范围</label>
              <select
                value={identity.data_scope}
                onChange={(e) =>
                  patchIdentity({ data_scope: e.target.value as DataScope })
                }
              >
                {SCOPE_OPTIONS.map((s) => (
                  <option key={s.value} value={s.value}>
                    {s.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {lastTrace && (
            <div className="side-section">
              <h2>调用追踪</h2>
              <div className="trace-grid">
                <div className="trace-item">
                  <span>检索方式</span>
                  <strong>
                    {lastTrace.retriever_used === "hybrid"
                      ? "混合检索"
                      : lastTrace.retriever_used}
                  </strong>
                </div>
                <div className="trace-item">
                  <span>权限过滤</span>
                  <strong>{lastTrace.acl_filtered_count}</strong>
                </div>
                <div className="trace-item">
                  <span>上下文用量</span>
                  <strong>{lastTrace.context_used_tokens}</strong>
                </div>
                <div className="trace-item">
                  <span>检索耗时</span>
                  <strong>{lastTrace.retrieval_ms ?? "-"} ms</strong>
                </div>
              </div>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}
