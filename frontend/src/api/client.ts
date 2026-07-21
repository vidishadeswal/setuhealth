import type { AskResponse, DocumentOut, EvalReport, Role, TokenResponse } from './types';

const BASE_URL = import.meta.env.VITE_API_BASE_URL as string;
export const AUTH_STORAGE_KEY = 'setuhealth_auth';

export interface StoredAuth {
  token: string;
  role: Role;
  email: string;
}

export function readStoredAuth(): StoredAuth | null {
  const raw = localStorage.getItem(AUTH_STORAGE_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as StoredAuth;
  } catch {
    return null;
  }
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

// Read synchronously at module load, not from a React effect: an effect only runs
// after the initial render commits, and React fires a descendant's effects before
// its ancestor's — so a page's own data-fetching effect (e.g. SourcesPage) would
// otherwise run before AuthContext's effect ever set this, sending the very first
// request after a page load out with no Authorization header.
let authToken: string | null = readStoredAuth()?.token ?? null;

export function setAuthToken(token: string | null): void {
  authToken = token;
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers = new Headers(options.headers);
  if (authToken) headers.set('Authorization', `Bearer ${authToken}`);
  if (options.body && !(options.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }

  const res = await fetch(`${BASE_URL}${path}`, { ...options, headers });

  if (!res.ok) {
    let detail: unknown = res.statusText;
    try {
      const data = await res.json();
      detail = data.detail ?? data;
    } catch {
      // response wasn't JSON — keep statusText
    }
    throw new ApiError(res.status, typeof detail === 'string' ? detail : JSON.stringify(detail));
  }

  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export function login(email: string, password: string): Promise<TokenResponse> {
  const body = new URLSearchParams({ username: email, password });
  return request<TokenResponse>('/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body,
  });
}

export function askQuestion(query: string): Promise<AskResponse> {
  return request<AskResponse>('/ask', { method: 'POST', body: JSON.stringify({ query }) });
}

export function getSources(): Promise<DocumentOut[]> {
  return request<DocumentOut[]>('/sources');
}

export function getEvalReport(): Promise<EvalReport> {
  return request<EvalReport>('/admin/eval-report');
}

export function uploadDocument(form: FormData): Promise<DocumentOut> {
  return request<DocumentOut>('/admin/documents', { method: 'POST', body: form });
}

export function deleteDocument(id: string): Promise<void> {
  return request<void>(`/admin/documents/${id}`, { method: 'DELETE' });
}
