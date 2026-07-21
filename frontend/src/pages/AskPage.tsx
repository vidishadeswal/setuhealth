import { useState, type FormEvent } from 'react';
import { askQuestion, ApiError } from '../api/client';
import type { AskResponse } from '../api/types';
import './AskPage.css';

interface HistoryEntry {
  id: string;
  query: string;
  response: AskResponse;
}

const REFUSAL_COPY: Record<string, string> = {
  low_confidence: 'Retrieval confidence was below the safety threshold for this question — SetuHealth would rather say nothing than guess.',
  out_of_scope: "This question doesn't match anything in the current drug-interaction corpus.",
};

export function AskPage() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const trimmed = query.trim();
    if (!trimmed || loading) return;

    setLoading(true);
    setError(null);
    try {
      const response = await askQuestion(trimmed);
      setHistory((prev) => [{ id: crypto.randomUUID(), query: trimmed, response }, ...prev]);
      setQuery('');
    } catch (err) {
      if (err instanceof ApiError && err.status === 429) {
        setError('Rate limit reached — wait a moment before asking again.');
      } else {
        setError(err instanceof ApiError ? err.message : 'Could not reach SetuHealth — check the API server.');
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div className="page-head">
        <h1>Ask a drug-interaction question</h1>
        <p>Answers are grounded only in the ingested corpus below, with a citation for every claim.</p>
      </div>

      <form onSubmit={handleSubmit} className="card ask-form">
        <div className="field" style={{ marginBottom: 12 }}>
          <label htmlFor="query">Question</label>
          <textarea
            id="query"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g. Can I take ibuprofen while on warfarin?"
            rows={3}
          />
        </div>
        <button type="submit" className="btn btn-primary" disabled={loading || !query.trim()}>
          {loading ? <span className="spinner" /> : 'Ask'}
        </button>
      </form>

      {error && <div className="error-banner" style={{ marginTop: 16 }}>{error}</div>}

      <div className="ask-history">
        {history.map((entry) => (
          <ResultCard key={entry.id} query={entry.query} response={entry.response} />
        ))}
      </div>
    </div>
  );
}

function ResultCard({ query, response }: { query: string; response: AskResponse }) {
  return (
    <div className="card result-card">
      <p className="result-query">{query}</p>

      {response.status === 'emergency' && (
        <div className="result-emergency">
          <div className="badge badge-danger" style={{ marginBottom: 10 }}>
            Emergency detected — not answered
          </div>
          <p className="result-emergency-note">
            This looks like it may describe a medical emergency. SetuHealth does not answer these —
            it refuses and shows resources instead.
          </p>
          <ul className="emergency-resources">
            {response.emergency_resources.map((r) => (
              <li key={r.label}>
                <strong>{r.label}:</strong> {r.value}
              </li>
            ))}
          </ul>
        </div>
      )}

      {response.status === 'refused' && (
        <div className="result-refused">
          <div className="badge badge-caution" style={{ marginBottom: 10 }}>
            Refused
          </div>
          <p>
            {(response.refusal_reason && REFUSAL_COPY[response.refusal_reason]) ??
              'SetuHealth cannot answer this safely.'}
          </p>
          {response.confidence_score !== null && (
            <ConfidenceBadge score={response.confidence_score} />
          )}
        </div>
      )}

      {response.status === 'answered' && (
        <div className="result-answered">
          {response.confidence_score !== null && (
            <ConfidenceBadge score={response.confidence_score} />
          )}
          <p className="result-answer">{response.answer}</p>
          {response.citations.length > 0 && (
            <div className="citations">
              <div className="citations-label">Sources</div>
              <ul>
                {response.citations.map((c) => (
                  <li key={c.chunk_id}>
                    {c.document_title}
                    {c.page_number !== null && <>, p.{c.page_number}</>}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function ConfidenceBadge({ score }: { score: number }) {
  return (
    <div className="badge badge-accent" style={{ marginBottom: 10 }}>
      Confidence {Math.round(score * 100)}%
    </div>
  );
}
