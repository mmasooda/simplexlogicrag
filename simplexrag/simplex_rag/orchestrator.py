"""
Orchestrator for the Simplex RAG system.

This module coordinates the ingestion, retrieval, reasoning and
generation components of the engine.  It exposes high‑level methods
for answering questions, processing BOQs and producing reports.

The orchestrator uses the ``SimplexDatabase`` for persistent storage
and retrieval, the ``LLMInterface`` for language model interactions,
and the ``ConstraintEngine`` for deterministic sizing.  It also
supports evaluation via the ``evaluation`` module.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Tuple

from .config import settings
from .database import SimplexDatabase
from .ingestion.datasheet_ingestor import ingest_datasheets
from pathlib import Path
from .llm_interface import LLMInterface
from .solver import ConstraintEngine

try:
    import openai  # type: ignore
except ImportError:
    openai = None

logger = logging.getLogger(__name__)


class SimplexRAGOrchestrator:
    """High‑level coordinator for the Simplex RAG engine."""

    def __init__(self, db: Optional[SimplexDatabase] = None, llm: Optional[LLMInterface] = None) -> None:
        self.db = db or SimplexDatabase()
        self.llm = llm or LLMInterface()
        self.solver = ConstraintEngine(self.db)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def ingest_directory(self, directory: str) -> None:
        """Ingest all supported files in the given directory into the database."""
        ingest_datasheets(Path(directory), self.db, self.llm)
        # Rebuild vector index after ingestion
        self.db.rebuild_vector_index()
        # Persist graph/vector index
        self.db.persist_graph()

    # ------------------------------------------------------------------
    # Question answering
    # ------------------------------------------------------------------
    def answer_question(self, question: str) -> Tuple[str, List[Dict]]:
        """Retrieve context and generate an answer for the user's question.

        :param question: natural language query
        :return: tuple of (answer, contexts) where contexts is a list of
            component dicts with part_number and description.
        """
        # Perform hybrid search for relevant components
        ranked = self.db.hybrid_search(question)
        # Take top N components for context
        top_contexts = []
        for part_number, score in ranked[: settings.vector_top_k]:
            data = self.db.graph.nodes.get(part_number)
            if data:
                top_contexts.append({"part_number": part_number, "description": data.get("description", "")})
        logger.debug(f"Retrieved {len(top_contexts)} context components for query: {question}")
        # Construct a prompt for the LLM summarising the context
        if self.llm.is_available():
            prompt = "Answer the following question about Simplex fire alarm products.\n"
            prompt += f"Question: {question}\n"
            prompt += "Here are some relevant components with descriptions:\n"
            for ctx in top_contexts[:10]:
                prompt += f"- {ctx['part_number']}: {ctx['description']}\n"
            prompt += "Provide a concise answer using the information above."
            # Use available LLM provider with modern APIs
            try:
                if self.llm._openai_available:
                    # Use modern OpenAI client API
                    completion = self.llm.openai_client.chat.completions.create(
                        model=self.llm.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                    )
                    answer = completion.choices[0].message.content
                elif self.llm._gemini_available:
                    # Use Gemini API
                    response = self.llm.gemini_model.generate_content(prompt)
                    answer = response.text
                else:
                    answer = "I'm sorry, no LLM is configured."
            except Exception as e:
                logger.warning(f"LLM query failed: {e}")
                answer = "I'm sorry, I encountered an error while generating the answer."
        else:
            # Fallback: simple summarisation of components
            if top_contexts:
                answer = "Relevant components:\n" + "\n".join(
                    [f"- {c['part_number']}: {c['description']}" for c in top_contexts[:5]]
                )
            else:
                answer = "I'm sorry, I couldn't find relevant information in the knowledge graph."
        return answer, top_contexts

    # ------------------------------------------------------------------
    # BOQ processing
    # ------------------------------------------------------------------
    def process_boq(self, boq_text: str) -> Tuple[Dict, Dict, str]:
        """Process a Bill of Quantities and return configuration, validation and report.

        :param boq_text: free‑form text describing system requirements
        :return: (configuration, validation, report)
        """
        requirements = self.llm.parse_boq(boq_text)
        configuration, validation = self.solver.size_system(requirements)
        report = self.llm.generate_report(configuration, validation)
        return configuration, validation, report
