export type Role = 'agent' | 'admin';

export interface TokenResponse {
  access_token: string;
  token_type: string;
  role: Role;
}

export interface Citation {
  document_title: string;
  page_number: number | null;
  chunk_id: string;
}

export interface EmergencyResource {
  label: string;
  value: string;
}

export type AskStatus = 'answered' | 'refused' | 'emergency';

export interface AskResponse {
  status: AskStatus;
  answer: string | null;
  citations: Citation[];
  confidence_score: number | null;
  refusal_reason: string | null;
  emergency_resources: EmergencyResource[];
}

export interface DocumentOut {
  id: string;
  source: string;
  title: string;
  url: string | null;
  published_date: string | null;
  ingested_at: string;
}

export interface EvalReport {
  total_queries: number;
  answered_count: number;
  refused_count: number;
  refusal_rate: number | null;
  refusal_reason_breakdown: Record<string, number>;
  avg_confidence_score_answered: number | null;
  total_ungrounded_claims_flagged: number;
  static_eval_command: string;
}
