import { apiFetch, onRequest } from "./client";
import type { Identity, QueryResponseV2 } from "../types";

let internalAuth = (import.meta.env.VITE_INTERNAL_AUTH as string | undefined)?.trim() ?? "";

export function setInternalAuth(secret: string) {
  internalAuth = secret.trim();
}

export function getInternalAuth() {
  return internalAuth;
}

onRequest((init) => {
  const headers = new Headers(init.headers);
  if (internalAuth) {
    headers.set("X-Internal-Auth", internalAuth);
  }
  const requestId =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : `web-${Date.now()}`;
  headers.set("X-Request-Id", requestId);
  return { ...init, headers };
});

export async function queryRag(query: string, identity: Identity): Promise<QueryResponseV2> {
  // 身份走 V2 body；仅附带内部鉴权与 request-id 请求头
  return apiFetch<QueryResponseV2>("/api/v2/rag/query", {
    method: "POST",
    body: JSON.stringify({
      query,
      summarize: true,
      identity: {
        ...identity,
        client_ip: "",
        request_id: "",
      },
    }),
  });
}

export async function healthCheck(): Promise<{ status: string }> {
  return apiFetch<{ status: string }>("/api/health");
}
