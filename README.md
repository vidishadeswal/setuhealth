# SetuHealth

> **PROTOTYPE — not a medical device, not for clinical use, not for real health decisions.**
> This is an engineering prototype demonstrating a trustworthy retrieval architecture. Its corpus (see
> [`corpus/sample/`](corpus/sample/)) is real FDA-approved drug labeling, pulled from the
> [openFDA](https://open.fda.gov/) API — a U.S. government work, not subject to copyright
> (17 U.S.C. § 105) — covering five drugs as a demo-scale subset, not a comprehensive or current
> reference. Do not use any answer from this system for an actual medical decision.

SetuHealth is a retrieval system for a professional-assist tool — a pharmacist or health-line agent
looking up drug-interaction information — that answers **only** from a fixed, verified document corpus,
cites the source and page for every claim, and refuses when retrieval confidence is low or the query
looks like a medical emergency, before generation ever runs.

The full design rationale (architecture, chunking strategy, embedding model choice, hybrid retrieval,
confidence scoring, the emergency-refusal engine, and evaluation methodology) is written up as a design
document — ask in your SetuHealth conversation to regenerate it, or see `docs/design.md` if present.

## Why this isn't "just ChatGPT"

- Every answer is grounded in retrieved passages only — a context-restricted prompt plus a post-generation
  groundedness check, not a hope that the model doesn't improvise.
- A deterministic, auditable emergency heuristic runs *before* retrieval and generation — not a prompt
  instruction that could be jailbroken.
- Confidence is scored from retrieval signals (reranker score, top-1/top-2 margin, BM25/vector agreement)
  and calibrated against a real eval set, not guessed.
- `GET /admin/eval-report` reports actual retrieval precision/recall, hallucination rate, and red-team
  false-negative rate — numbers, not claims.

## Stack

Fully local — no external API keys required:

- **Generation**: local LLM via [Ollama](https://ollama.com)
- **Embeddings**: `BAAI/bge-small-en-v1.5` (sentence-transformers, CPU)
- **Reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Vector search**: FAISS (`IndexFlatIP` over normalized embeddings)
- **Keyword search**: BM25 (`rank_bm25`)
- **Backend**: FastAPI + SQLAlchemy
- **DB**: SQLite by default for zero-setup local dev; swap `DATABASE_URL` for Postgres (see
  `docker-compose.yml`) to match the "real" architecture.

## Setup

1. Install [Ollama](https://ollama.com) and pull a model:
   ```
   ollama pull llama3.2:3b
   ```
2. Create a virtualenv and install dependencies:
   ```
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Copy the env template and generate your own admin secret:
   ```
   cp .env.example .env
   python -c "import secrets; print(secrets.token_hex(32))"   # paste into ADMIN_SECRET_KEY
   ```
4. Ingest the sample corpus and create the first admin user:
   ```
   python -m backend.app.seed
   ```
5. Run the API:
   ```
   uvicorn backend.app.main:app --reload
   ```
6. Try it — log in and ask a question:
   ```
   TOKEN=$(curl -s -X POST localhost:8000/auth/login \
     -d "username=agent@setuhealth.local&password=<from step 4>" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

   curl -s -X POST localhost:8000/ask -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"query": "Can I take ibuprofen while on warfarin?"}'
   ```
   Or open `localhost:8000/docs` for interactive Swagger UI.

### Docker / Postgres path

`docker-compose.yml` runs the "real" architecture (Postgres instead of the SQLite dev
default) — Ollama still runs on the host for direct GPU/Metal access:

```
ollama serve                                    # on the host, if not already running
docker compose up -d
docker compose exec backend python -m backend.app.seed
docker compose restart backend                  # see note below
```

**Why the restart**: the retrieval indexes (FAISS + BM25) live in the running server
process's memory, not just on disk — `registry.refresh()` runs in-process on every
ingestion so the already-serving process picks up new documents immediately. But
`docker compose exec` runs the seed script as a *separate* process inside the container;
it writes correctly to Postgres and the shared `data/` volume, but the already-running
API process never hears about it until it reloads that state itself. Restarting is the
correct fix here, not a workaround — it's the same reason `POST /admin/documents`
(which runs inside the live request-handling process) doesn't need one.

## Evaluation

```
python -m eval.run_eval
```

Runs the retrieval eval, groundedness eval, and red-team eval against the running pipeline and prints
precision/recall, hallucination rate, and emergency-heuristic false-negative rate.

## Project layout

```
backend/app/
  ingestion/     text extraction -> chunking -> embedding -> indexing (offline/batch)
  retrieval/     embeddings, FAISS, BM25, hybrid fusion, cross-encoder reranking
  safety/        emergency heuristic, confidence scoring, groundedness check
  generation/    Ollama client + context-restricted prompt builder
  api/           /ask, /sources, /admin/* routes
  models/        SQLAlchemy: User, Document, Chunk, QueryLog
corpus/sample/   real FDA drug-labeling excerpts (openFDA API), 5 drugs, demo-scale subset
eval/            retrieval + red-team eval sets and the eval runner
```
