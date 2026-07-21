"""Offline evaluation (design doc Section 13). Three independent eval sets, each
answering a different question — see the design doc for why they're kept separate
rather than averaged into one accuracy number.

The retrieval/groundedness eval calls api.routes_ask._run_pipeline directly — the same
function the live /ask endpoint runs — rather than re-implementing the retrieve/rerank
sequence here. Two eval-only copies of that logic could silently drift (e.g. one of them
forgetting query expansion), which would make the eval numbers a measurement of a
different system than the one actually serving traffic.

Run with: python -m eval.run_eval
Requires the corpus to already be ingested (`python -m backend.app.seed`) and, for the
groundedness eval only, Ollama running locally.
"""

import asyncio
import json
from pathlib import Path

import httpx

from backend.app.api.routes_ask import _run_pipeline
from backend.app.config import get_settings
from backend.app.db.session import SessionLocal
from backend.app.models.chunk import Chunk
from backend.app.models.document import Document
from backend.app.retrieval.registry import registry
from backend.app.safety.emergency import check_emergency

EVAL_DIR = Path(__file__).parent


def _load_jsonl(name: str) -> list[dict]:
    with open(EVAL_DIR / name) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_retrieval_and_groundedness_eval(db, settings) -> None:
    items = _load_jsonl("retrieval_eval.jsonl")
    hit_at_1 = hit_at_k = 0
    total_sentences = total_ungrounded = 0
    ollama_reachable = True

    print(f"\n=== Retrieval + groundedness eval ({len(items)} questions) ===")
    for item in items:
        try:
            result = asyncio.run(_run_pipeline(db, settings, item["query"]))
        except httpx.ConnectError:
            if ollama_reachable:
                print("  (Ollama unreachable — remaining items skip retrieval/groundedness metrics)")
            ollama_reachable = False
            continue

        chunk_rows = (
            db.query(Chunk, Document.title)
            .join(Document, Chunk.doc_id == Document.id)
            .filter(Chunk.id.in_(result.retrieved_chunk_ids))
            .all()
        )
        lookup = {c.id: (title, c.page_number) for c, title in chunk_rows}
        # retrieved_chunk_ids preserves reranked order even when the pipeline went on
        # to refuse — confidence is a downstream decision, not a retrieval-quality one.
        ordered = [lookup[cid] for cid in result.retrieved_chunk_ids if cid in lookup]

        hit_positions = [
            i for i, (title, page) in enumerate(ordered)
            if title == item["expected_document_title"] and page == item["expected_page"]
        ]
        if hit_positions:
            hit_at_k += 1
            if hit_positions[0] == 0:
                hit_at_1 += 1
        else:
            print(f"  MISS: \"{item['query']}\" — expected {item['expected_document_title']} p.{item['expected_page']}")

        total_sentences += result.total_sentences
        total_ungrounded += result.ungrounded_count

    n = len(items)
    print(f"\nRetrieval recall@top_k_reranked: {hit_at_k}/{n} = {hit_at_k / n:.0%}")
    print(f"Retrieval hit@1 (top reranked chunk is the expected one): {hit_at_1}/{n} = {hit_at_1 / n:.0%}")
    if ollama_reachable and total_sentences:
        print(f"Hallucination rate (ungrounded sentences / total): {total_ungrounded}/{total_sentences} = {total_ungrounded / total_sentences:.1%}")


def run_redteam_eval() -> None:
    items = _load_jsonl("redteam_eval.jsonl")
    print(f"\n=== Red-team emergency-heuristic eval ({len(items)} phrasings) ===")

    false_negatives, false_positives, expected_positive, expected_negative = [], [], 0, 0

    for item in items:
        result = check_emergency(item["query"])
        if item["expect_flagged"]:
            expected_positive += 1
            if not result.flagged:
                false_negatives.append(item)
        else:
            expected_negative += 1
            if result.flagged:
                false_positives.append(item)

    fn_rate = len(false_negatives) / expected_positive if expected_positive else 0.0
    fp_rate = len(false_positives) / expected_negative if expected_negative else 0.0

    print(f"False-negative rate (missed real emergencies — the critical metric): {len(false_negatives)}/{expected_positive} = {fn_rate:.0%}")
    for item in false_negatives:
        print(f"  MISSED [{item['category']}]: \"{item['query']}\"")

    print(f"False-positive rate (over-refusal on non-emergencies): {len(false_positives)}/{expected_negative} = {fp_rate:.0%}")
    for item in false_positives:
        print(f"  OVER-FLAGGED [{item['category']}]: \"{item['query']}\"")


def main() -> None:
    settings = get_settings()
    db = SessionLocal()
    try:
        registry.refresh(db)
        run_retrieval_and_groundedness_eval(db, settings)
        run_redteam_eval()
    finally:
        db.close()


if __name__ == "__main__":
    main()
