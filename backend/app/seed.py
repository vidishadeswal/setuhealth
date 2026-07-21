"""Bootstraps a fresh install: creates tables, one admin user, one agent user, and
ingests the sample corpus. Run once with `python -m backend.app.seed`.
"""

import secrets

from backend.app import models  # noqa: F401
from backend.app.config import get_settings
from backend.app.core.security import hash_password
from backend.app.db.base import Base
from backend.app.db.session import SessionLocal, engine
from backend.app.ingestion.pipeline import ingest_document
from backend.app.models.user import Role, User


def _create_user(db, email: str, role: Role) -> str:
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        print(f"  {email} already exists, skipping")
        return ""
    password = secrets.token_urlsafe(12)
    db.add(User(email=email, password_hash=hash_password(password), role=role))
    db.commit()
    return password


def main() -> None:
    settings = get_settings()
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        print("Creating users...")
        admin_password = _create_user(db, "admin@setuhealth.local", Role.admin)
        agent_password = _create_user(db, "agent@setuhealth.local", Role.agent)
        if admin_password:
            print(f"  admin@setuhealth.local / {admin_password}")
        if agent_password:
            print(f"  agent@setuhealth.local / {agent_password}")
        print("  (save these now — they are not stored or shown again)")

        print("\nIngesting sample corpus...")
        for path in sorted(settings.corpus_dir.glob("*.txt")):
            title = path.stem.replace("_", " ").title()
            print(f"  {path.name} -> {title}")
            ingest_document(db, path, source="FDA-approved drug labeling via openFDA (api.fda.gov)", title=title)

        print("\nDone. Start the API with: uvicorn backend.app.main:app --reload")
    finally:
        db.close()


if __name__ == "__main__":
    main()
