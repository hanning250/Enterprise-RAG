export type ApiError = Error & { status?: number; detail?: string };

type RequestInterceptor = (init: RequestInit) => RequestInit | Promise<RequestInit>;

const requestHooks: RequestInterceptor[] = [];

export function onRequest(hook: RequestInterceptor) {
  requestHooks.push(hook);
}

function apiBase(): string {
  const raw = (import.meta.env.VITE_API_BASE as string | undefined)?.trim();
  return raw?.replace(/\/$/, "") ?? "";
}

export async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const timeoutMs = Number(
    (import.meta.env.VITE_API_TIMEOUT_MS as string | undefined)?.trim() || 120000
  );
  const controller = new AbortController();
  const timer =
    Number.isFinite(timeoutMs) && timeoutMs > 0
      ? setTimeout(() => controller.abort(), timeoutMs)
      : null;

  let nextInit: RequestInit = {
    ...init,
    signal: init.signal ?? controller.signal,
    headers: {
      Accept: "application/json",
      ...(init.body ? { "Content-Type": "application/json" } : {}),
      ...(init.headers ?? {}),
    },
  };

  try {
    for (const hook of requestHooks) {
      nextInit = await hook(nextInit);
    }

    const res = await fetch(`${apiBase()}${path}`, nextInit);

    if (!res.ok) {
      let detail = res.statusText;
      try {
        const data = (await res.json()) as { detail?: unknown };
        if (typeof data.detail === "string") detail = data.detail;
        else if (data.detail != null) detail = JSON.stringify(data.detail);
      } catch {
        /* ignore */
      }
      const err = new Error(detail) as ApiError;
      err.status = res.status;
      err.detail = detail;
      throw err;
    }

    if (res.status === 204) return undefined as T;
    return (await res.json()) as T;
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      const timeoutErr = new Error(
        `请求超时（>${Math.round(timeoutMs / 1000)}s）。多半卡在大模型生成，请重试或检查模型网关。`
      ) as ApiError;
      timeoutErr.status = 408;
      throw timeoutErr;
    }
    throw err;
  } finally {
    if (timer) clearTimeout(timer);
  }
}
