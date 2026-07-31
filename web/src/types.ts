export type DataScope = "self" | "department" | "company";

export type Identity = {
  /** 工号，对应工资表 employee_id */
  user_id: string;
  user_name: string;
  /** 对应工资表「部门」 */
  department: string;
  /** 对应工资表「岗位」 */
  position: string;
  /** ACL 权限角色（可由岗位推断） */
  roles: string[];
  data_scope: DataScope;
};

export type RetrievedDoc = {
  page_content: string;
  metadata: Record<string, unknown>;
};

export type QueryTrace = {
  retriever_used: string;
  rerank_used: boolean;
  acl_filtered_count: number;
  context_used_tokens: number;
  docs_included_count: number;
  retrieval_ms?: number | null;
  answer_ms?: number | null;
  fallback_reason?: string | null;
  cache_hit?: boolean;
};

export type QueryResponseV2 = {
  answer: string;
  evidence_docs: RetrievedDoc[];
  trace: QueryTrace;
  identity_snapshot: Identity;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  evidence?: RetrievedDoc[];
  trace?: QueryTrace;
  error?: string;
};
