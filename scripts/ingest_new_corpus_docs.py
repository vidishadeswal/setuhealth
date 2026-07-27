"""Idempotently ingests any corpus/sample/*.txt file not already in the database, by
title. Unlike backend/app/seed.py (which is a from-scratch bootstrap and would
duplicate every already-ingested document if re-run), this only adds what's missing —
safe to run after dropping new files into the corpus without touching existing users,
documents, or index entries.

Usage: python -m scripts.ingest_new_corpus_docs
"""

from backend.app import models  # noqa: F401
from backend.app.config import get_settings
from backend.app.db.session import SessionLocal
from backend.app.ingestion.pipeline import ingest_document
from backend.app.models.document import Document


def main() -> None:
    settings = get_settings()
    db = SessionLocal()
    try:
        existing_titles = {title for (title,) in db.query(Document.title).all()}
        for path in sorted(settings.corpus_dir.glob("*.txt")):
            title = path.stem.replace("_", " ").title()
            if title in existing_titles:
                print(f"  skip (already ingested): {title}")
                continue
            print(f"  ingesting: {path.name} -> {title}")
            ingest_document(db, path, source="FDA-approved drug labeling via openFDA (api.fda.gov)", title=title)
        print("Done.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
