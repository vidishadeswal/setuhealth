"""Threshold calibration (design doc Section 13): sweep CONFIDENCE_THRESHOLD against a
positive set (questions the corpus should answer) and a negative set (genuinely
out-of-scope questions it must refuse), and report precision/refusal-rate at each
candidate threshold — the actual curve behind the chosen operating point, not a guess.

Run with: python -m eval.threshold_sweep
"""

import asyncio
import json
from pathlib import Path

from backend.app.api.routes_ask import _run_pipeline
from backend.app.config import get_settings
from backend.app.db.session import SessionLocal
from backend.app.retrieval.registry import registry

EVAL_DIR = Path(__file__).parent

# Genuinely out-of-scope questions — not drug-interaction questions at all, or about a
# drug entirely outside the corpus. The system must refuse these at every threshold
# candidate worth considering.
NEGATIVE_SET = [
    "What is the capital of France?",
    "What vaccines should a 6-month-old baby receive?",
    "What are the symptoms of dengue?",
    "How do I renew my passport?",
    "What's a good recipe for banana bread?",
]

CANDIDATE_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def _load_positive_set() -> list[str]:
    with open(EVAL_DIR / "retrieval_eval.jsonl") as f:
        return [json.loads(line)["query"] for line in f if line.strip()]


def main() -> None:
    settings = get_settings()
    db = SessionLocal()
    try:
        registry.refresh(db)

        positive_queries = _load_positive_set()
        print(f"Scoring {len(positive_queries)} positive (should-answer) and {len(NEGATIVE_SET)} negative (should-refuse) queries...\n")

        positive_scores = [asyncio.run(_run_pipeline(db, settings, q)).confidence_score or 0.0 for q in positive_queries]
        negative_scores = [asyncio.run(_run_pipeline(db, settings, q)).confidence_score or 0.0 for q in NEGATIVE_SET]

        print("Positive set confidence scores:")
        for q, s in zip(positive_queries, positive_scores):
            print(f"  {s:.2f}  {q}")
        print("\nNegative set confidence scores:")
        for q, s in zip(NEGATIVE_SET, negative_scores):
            print(f"  {s:.2f}  {q}")

        print(f"\n{'threshold':>9}  {'answer-rate (positives)':>24}  {'leak-rate (negatives)':>22}")
        for t in CANDIDATE_THRESHOLDS:
            answered = sum(1 for s in positive_scores if s >= t)
            leaked = sum(1 for s in negative_scores if s >= t)
            print(f"{t:>9.2f}  {answered:>3}/{len(positive_scores)} = {answered/len(positive_scores):>6.0%}          {leaked:>3}/{len(negative_scores)} = {leaked/len(negative_scores):>6.0%}")

        current = settings.confidence_threshold
        print(f"\nCurrent CONFIDENCE_THRESHOLD = {current}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
