import { useEffect, useState } from 'react';
import { getSources, ApiError } from '../api/client';
import type { DocumentOut } from '../api/types';

export function SourcesPage() {
  const [documents, setDocuments] = useState<DocumentOut[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getSources()
      .then((docs) => {
        setDocuments(docs);
        setError(null);
      })
      .catch((err) => setError(err instanceof ApiError ? err.message : 'Could not load sources.'));
  }, []);

  return (
    <div>
      <div className="page-head">
        <h1>Corpus sources</h1>
        <p>Every document currently indexed — what SetuHealth's answers are allowed to draw from.</p>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {documents === null && !error && <p>Loading…</p>}

      {documents !== null && documents.length === 0 && <p>No documents ingested yet.</p>}

      {documents !== null && documents.length > 0 && (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Title</th>
                <th>Source</th>
                <th>Published</th>
                <th>Ingested</th>
              </tr>
            </thead>
            <tbody>
              {documents.map((doc) => (
                <tr key={doc.id}>
                  <td>{doc.url ? <a href={doc.url}>{doc.title}</a> : doc.title}</td>
                  <td>{doc.source}</td>
                  <td>{doc.published_date ?? '—'}</td>
                  <td>{new Date(doc.ingested_at).toLocaleDateString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
