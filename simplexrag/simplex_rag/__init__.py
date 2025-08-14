"""
Simplex Fire Alarm RAG System
=============================

This package contains a modular implementation of a retrieval‑augmented
generation (RAG) engine tailored for designing and quoting Simplex fire
alarm systems.  The code has been refactored into discrete modules
following best practices suggested by an external review.  Key
enhancements include:

* A hybrid data layer that uses SQLite for persistence and in‑memory
  graphs for fast traversal of product relationships.
* Robust document ingestion capable of parsing PDF, Word and Excel
  datasheets, with optional LLM batch extraction for complex tables.
* Vector and graph retrieval engines that can be combined to
  maximize recall and relevance of context.
* Deterministic constraint logic encapsulated in a solver module to
  compute panel configurations based on BOQ requirements.
* A pluggable LLM interface with support for OpenAI's batch API
  requests and a regex fallback when keys are unavailable.

Usage of this package typically involves building the database and
graph via the ingestion utilities, then instantiating the
``SimplexRAGOrchestrator`` to process BOQs and specifications.
"""

from .orchestrator import SimplexRAGOrchestrator  # noqa: F401
from .config import Settings  # noqa: F401