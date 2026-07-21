import { useEffect, useRef, useState, type FormEvent } from 'react';
import { deleteDocument, getEvalReport, getSources, uploadDocument, ApiError } from '../api/client';
import type { DocumentOut, EvalReport } from '../api/types';
import './AdminPage.css';

export function AdminPage() {
  const [documents, setDocuments] = useState<DocumentOut[] | null>(null);
  const [report, setReport] = useState<EvalReport | null>(null);
  const [sourcesError, setSourcesError] = useState<string | null>(null);
  const [reportError, setReportError] = useState<string | null>(null);

  function reloadAll() {
    getSources()
      .then((docs) => {
        setDocuments(docs);
        setSourcesError(null);
      })
      .catch((err) => setSourcesError(err instanceof ApiError ? err.message : 'Could not load sources.'));
    getEvalReport()
      .then((r) => {
        setReport(r);
        setReportError(null);
      })
      .catch((err) => setReportError(err instanceof ApiError ? err.message : 'Could not load eval report.'));
  }

  useEffect(reloadAll, []);

  return (
    <div>
      <div className="page-head">
        <h1>Admin</h1>
        <p>Manage the corpus and watch the live operational metrics it produces.</p>
      </div>

      {sourcesError && <div className="error-banner">{sourcesError}</div>}
      {reportError && <div className="error-banner">{reportError}</div>}

      <EvalReportPanel report={report} />
      <UploadPanel onUploaded={reloadAll} />
      <DocumentsPanel documents={documents} onChanged={reloadAll} />
    </div>
  );
}

function EvalReportPanel({ report }: { report: EvalReport | null }) {
  return (
    <section className="admin-section">
      <h2>Live eval report</h2>
      {!report && <p>Loading…</p>}
      {report && (
        <div className="stat-grid">
          <Stat label="Total queries" value={report.total_queries} />
          <Stat label="Answered" value={report.answered_count} />
          <Stat label="Refused" value={report.refused_count} />
          <Stat
            label="Refusal rate"
            value={report.refusal_rate !== null ? `${Math.round(report.refusal_rate * 100)}%` : '—'}
          />
          <Stat
            label="Avg. confidence (answered)"
            value={
              report.avg_confidence_score_answered !== null
                ? `${Math.round(report.avg_confidence_score_answered * 100)}%`
                : '—'
            }
          />
          <Stat label="Ungrounded claims flagged" value={report.total_ungrounded_claims_flagged} />
        </div>
      )}
      {report && (
        <div className="refusal-breakdown">
          {Object.entries(report.refusal_reason_breakdown).map(([reason, count]) => (
            <span key={reason} className="badge badge-accent">
              {reason.replace(/_/g, ' ')}: {count}
            </span>
          ))}
        </div>
      )}
      <p className="eval-note">
        For retrieval precision/recall, measured hallucination rate, and red-team false-negative rate,
        run <code>{report?.static_eval_command ?? 'python -m eval.run_eval'}</code> — those need known-correct
        answers, which live traffic doesn't have.
      </p>
    </section>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="stat-tile">
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
}

function UploadPanel({ onUploaded }: { onUploaded: () => void }) {
  const [source, setSource] = useState('');
  const [title, setTitle] = useState('');
  const [url, setUrl] = useState('');
  const [publishedDate, setPublishedDate] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!file) return;
    setSubmitting(true);
    setError(null);
    setSuccess(null);
    try {
      const form = new FormData();
      form.append('file', file);
      form.append('source', source);
      form.append('title', title);
      if (url) form.append('url', url);
      if (publishedDate) form.append('published_date', publishedDate);

      const doc = await uploadDocument(form);
      setSuccess(`Ingested "${doc.title}".`);
      setSource('');
      setTitle('');
      setUrl('');
      setPublishedDate('');
      setFile(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
      onUploaded();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Upload failed — check the API server.');
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <section className="admin-section">
      <h2>Upload a document</h2>
      <p className="section-hint">Plain-text extraction only — .pdf or .txt, no OCR for scanned files.</p>
      {error && <div className="error-banner">{error}</div>}
      {success && <div className="success-banner">{success}</div>}
      <form onSubmit={handleSubmit} className="card upload-form">
        <div className="field">
          <label htmlFor="upload-file">File</label>
          <input
            id="upload-file"
            ref={fileInputRef}
            type="file"
            accept=".pdf,.txt"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            required
          />
        </div>
        <div className="upload-grid">
          <div className="field">
            <label htmlFor="upload-title">Title</label>
            <input id="upload-title" value={title} onChange={(e) => setTitle(e.target.value)} required />
          </div>
          <div className="field">
            <label htmlFor="upload-source">Source / publisher</label>
            <input id="upload-source" value={source} onChange={(e) => setSource(e.target.value)} required />
          </div>
          <div className="field">
            <label htmlFor="upload-url">URL (optional)</label>
            <input id="upload-url" type="url" value={url} onChange={(e) => setUrl(e.target.value)} />
          </div>
          <div className="field">
            <label htmlFor="upload-date">Published date (optional)</label>
            <input
              id="upload-date"
              type="date"
              value={publishedDate}
              onChange={(e) => setPublishedDate(e.target.value)}
            />
          </div>
        </div>
        <button type="submit" className="btn btn-primary" disabled={submitting || !file}>
          {submitting ? <span className="spinner" /> : 'Upload & ingest'}
        </button>
      </form>
    </section>
  );
}

function DocumentsPanel({
  documents,
  onChanged,
}: {
  documents: DocumentOut[] | null;
  onChanged: () => void;
}) {
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleDelete(doc: DocumentOut) {
    if (!window.confirm(`Delete "${doc.title}"? This removes it from retrieval immediately.`)) return;
    setDeletingId(doc.id);
    setError(null);
    try {
      await deleteDocument(doc.id);
      onChanged();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Delete failed.');
    } finally {
      setDeletingId(null);
    }
  }

  return (
    <section className="admin-section">
      <h2>Ingested documents</h2>
      {error && <div className="error-banner">{error}</div>}
      {documents === null && <p>Loading…</p>}
      {documents !== null && documents.length === 0 && <p>No documents ingested yet.</p>}
      {documents !== null && documents.length > 0 && (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Title</th>
                <th>Source</th>
                <th>Ingested</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {documents.map((doc) => (
                <tr key={doc.id}>
                  <td>{doc.title}</td>
                  <td>{doc.source}</td>
                  <td>{new Date(doc.ingested_at).toLocaleDateString()}</td>
                  <td>
                    <button
                      type="button"
                      className="btn btn-danger"
                      disabled={deletingId === doc.id}
                      onClick={() => handleDelete(doc)}
                    >
                      {deletingId === doc.id ? <span className="spinner" /> : 'Delete'}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
