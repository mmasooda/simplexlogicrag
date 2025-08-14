"""
Configuration management for the Simplex RAG engine.

This module centralizes runtime settings to simplify tuning and
integration.  Values can be overridden via environment variables or by
providing a custom ``Settings`` instance.  A ``.env`` file may be
loaded automatically if present in the working directory using
``python‑dotenv`` (optional dependency).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Helper to fetch an environment variable or return a default."""
    return os.environ.get(name, default)


@dataclass
class Settings:
    """Centralised configuration values for the RAG engine."""

    # ------------------------------------------------------------------
    # Data and storage
    # ------------------------------------------------------------------

    # Path to the SQLite database storing component definitions.  A
    # relative path will be interpreted relative to the working
    # directory.
    db_path: str = field(default_factory=lambda: _env("SIMPLEX_DB_PATH", "simplex_components.db"))

    # Path to persist the in‑memory graph as JSON for quick loading on
    # subsequent runs.  If None, the graph will not be persisted.
    graph_persistence_path: Optional[str] = field(default_factory=lambda: _env("SIMPLEX_GRAPH_PATH"))

    # Path to persist the vector index.  If None, the index is built
    # from scratch on each run.
    vector_index_path: Optional[str] = field(default_factory=lambda: _env("SIMPLEX_VECTOR_INDEX_PATH"))

    # Maximum depth for graph traversal when expanding entity
    # relationships.  Lower values reduce computation but may miss
    # distant connections.
    graph_max_depth: int = int(_env("SIMPLEX_GRAPH_DEPTH", "2"))

    # Number of results to return from vector retrieval before
    # reranking.  A higher value increases recall but can slow
    # reranking and overwhelm the LLM context window.
    vector_top_k: int = int(_env("SIMPLEX_VECTOR_TOP_K", "20"))

    # The embedding model to use for text vectorization.  If using
    # OpenAI embeddings, set ``embedding_model = 'openai'`` and
    # provide your API key via ``OPENAI_API_KEY``.  Otherwise a
    # fallback TF‑IDF model is used.
    embedding_model: str = field(default_factory=lambda: _env("SIMPLEX_EMBED_MODEL", "tfidf"))

    # ------------------------------------------------------------------
    # LLM integration
    # ------------------------------------------------------------------

    # Model name for chat completions.  Examples: ``gpt-3.5-turbo``,
    # ``gemini-pro``.  If ``None``, only the regex fallback is used.
    llm_model: Optional[str] = field(default_factory=lambda: _env("SIMPLEX_LLM_MODEL"))

    # Use the OpenAI batch API for large numbers of extraction calls.
    use_batch_api: bool = _env("SIMPLEX_USE_BATCH_API", "false").lower() == "true"

    # Path to store JSONL tasks for batch API submissions.
    batch_tasks_path: str = field(default_factory=lambda: _env("SIMPLEX_BATCH_TASKS_PATH", "./batch_tasks.jsonl"))

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    # Path to a ground truth dataset for evaluation.  If None,
    # evaluation metrics are not computed.
    eval_dataset_path: Optional[str] = field(default_factory=lambda: _env("SIMPLEX_EVAL_DATASET"))

    # Use advanced evaluation frameworks if installed (e.g. ragas,
    # quotient).  If False, a simple precision/recall calculation is
    # used.
    use_advanced_eval: bool = _env("SIMPLEX_USE_ADVANCED_EVAL", "false").lower() == "true"

    # ------------------------------------------------------------------
    # Miscellaneous
    # ------------------------------------------------------------------

    # Enable verbose debug logging.  Useful for tracing the execution
    # during development.
    debug: bool = _env("SIMPLEX_DEBUG", "false").lower() == "true"


settings = Settings()
