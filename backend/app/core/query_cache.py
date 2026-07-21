"""Exact-match query cache (design doc Section 15): identical question text is common
for FAQ-style queries ("can I take X with Y") — cheap and correctness-safe to cache,
unlike a semantic cache, which would reintroduce the same confidence-calibration problem
retrieval already has to solve once. In-memory only; a single-process prototype has no
need for a shared cache backend.

Deliberately keyed and populated *after* the emergency heuristic has already run fresh
on the request — the cache must never let a query skip that check. See api/routes_ask.py.
"""

from collections import OrderedDict
from typing import Any

MAX_ENTRIES = 256


class QueryCache:
    def __init__(self, max_entries: int = MAX_ENTRIES):
        self._store: OrderedDict[str, Any] = OrderedDict()
        self.max_entries = max_entries

    @staticmethod
    def normalize(query: str) -> str:
        return " ".join(query.strip().lower().split())

    def get(self, query: str) -> Any | None:
        key = self.normalize(query)
        if key not in self._store:
            return None
        self._store.move_to_end(key)  # LRU touch
        return self._store[key]

    def set(self, query: str, value: Any) -> None:
        key = self.normalize(query)
        self._store[key] = value
        self._store.move_to_end(key)
        if len(self._store) > self.max_entries:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


query_cache = QueryCache()
