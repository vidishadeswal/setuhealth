from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app import models  # noqa: F401 — ensures all tables are registered before create_all
from backend.app.api import routes_admin, routes_ask, routes_auth, routes_sources
from backend.app.config import get_settings
from backend.app.db.base import Base
from backend.app.db.session import SessionLocal, engine
from backend.app.retrieval.registry import registry


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        registry.refresh(db)
    finally:
        db.close()

    yield


app = FastAPI(
    title="SetuHealth",
    description="PROTOTYPE — professional-assist drug-interaction lookup. Not for clinical use.",
    version="0.1.0",
    lifespan=lifespan,
)

# The frontend (a separate Vite dev server / static build) is a different origin from
# this API — allowed origins are explicit, not "*", since requests carry a bearer token.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_auth.router)
app.include_router(routes_ask.router)
app.include_router(routes_sources.router)
app.include_router(routes_admin.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "prototype": True}
