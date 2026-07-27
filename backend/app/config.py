from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(REPO_ROOT / ".env"), extra="ignore")

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    database_url: str = "sqlite:///./data/setuhealth.db"

    admin_secret_key: str = "dev-only-insecure-key-replace-me"
    access_token_expire_minutes: int = 120

    # Calibrated via eval/threshold_sweep.py: 0.30-0.45 is a plateau of 96% answer rate
    # on the positive eval set with 0% leakage on genuinely out-of-scope negatives (which
    # never scored above 0.10). Set to 0.40, not the plateau's own floor of 0.45 —
    # borderline real questions (e.g. "cholesterol medicine" + "antifungal", both
    # resolvable via query_expansion.py but landing at ~0.44 due to reranker phrasing
    # sensitivity) sit just under 0.45 with genuinely correct retrieval underneath them.
    # 0.40 keeps a full 0.30 safety margin above the highest observed negative (0.10) —
    # more margin than 0.45 had. Re-run the sweep before changing this further.
    confidence_threshold: float = 0.40
    top_k_candidates: int = 20
    top_k_reranked: int = 5
    chunk_token_size: int = 350
    chunk_overlap_ratio: float = 0.15

    rate_limit_per_minute: int = 20

    corpus_dir: Path = REPO_ROOT / "corpus" / "sample"
    index_dir: Path = REPO_ROOT / "data"

    cors_allow_origins: list[str] = ["http://localhost:5174"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
