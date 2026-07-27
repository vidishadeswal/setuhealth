import { useEffect, useState, type FormEvent } from 'react';
import { askQuestion, getSources, ApiError } from '../api/client';
import type { AskResponse, Citation } from '../api/types';
import { parsePassageContent } from '../utils/parsePassage';
import './AskPage.css';

interface HistoryEntry {
  id: string;
  query: string;
  response: AskResponse;
}

const REFUSAL_COPY: Record<string, string> = {
  low_confidence:
    'Retrieval confidence was below the safety threshold for this question — SetuHealth would rather say nothing than guess. This usually means the drug isn\'t in the current corpus (see below), or the interaction isn\'t addressed in that drug\'s label.',
  out_of_scope: "This question doesn't match anything in the current drug-interaction corpus.",
};

function titleToDrugName(title: string): string {
  return title.replace(/\s+Interactions$/i, '');
}

export function AskPage() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [coveredDrugs, setCoveredDrugs] = useState<string[] | null>(null);

  useEffect(() => {
    getSources()
      .then((docs) => setCoveredDrugs(docs.map((d) => titleToDrugName(d.title)).sort()))
      .catch(() => setCoveredDrugs(null)); // non-critical — the page works without this note
  }, []);

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
        {coveredDrugs && coveredDrugs.length > 0 && (
          <p className="corpus-scope-note">
            This demo currently covers: <strong>{coveredDrugs.join(', ')}</strong>. A question about
            anything else will be correctly refused as out of scope, not answered incorrectly — see{' '}
            <a href="/sources">Sources</a> for the exact documents.
          </p>
        )}
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
                  <CitationItem key={c.chunk_id} citation={c} />
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function CitationItem({ citation }: { citation: Citation }) {
  const [expanded, setExpanded] = useState(false);
  const parsed = parsePassageContent(citation.content);

  return (
    <li className="citation-item">
      <button type="button" className="citation-toggle" onClick={() => setExpanded((v) => !v)}>
        <span className="citation-caret">{expanded ? '▾' : '▸'}</span>
        {citation.document_title}
        {citation.page_number !== null && <>, p.{citation.page_number}</>}
      </button>
      {expanded && (
        <div className="citation-passage">
          {parsed.prefix && <p>{parsed.prefix}</p>}
          {parsed.table && <PassageTable table={parsed.table} />}
          {parsed.suffix && <p>{parsed.suffix}</p>}
        </div>
      )}
    </li>
  );
}

function PassageTable({ table }: { table: { headers: string[]; rows: string[][] } }) {
  return (
    <div className="passage-table-wrap">
      <table className="passage-table">
        <thead>
          <tr>
            {table.headers.map((h, i) => (
              <th key={i}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {table.rows.map((row, i) => (
            <tr key={i}>
              {row.map((cell, j) => (
                <td key={j}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
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
