"""
Evaluation utilities for the Simplex RAG engine.

This module defines metrics to assess the quality of retrieval and
generation.  It includes simple precision/recall metrics that do
not require external dependencies and optional integration points
for advanced frameworks such as RAGAS, Arize and Qdrant's evaluation
tools.  The evaluation functions accept a ground‑truth dataset in JSON
format with fields ``question``, ``answer`` and optional
``context_ids`` that correspond to relevant component part numbers.

The evaluation outputs a report with aggregate scores and detailed
per‑query results.  If advanced evaluation tools are available and
``use_advanced_eval`` is enabled in the configuration, those tools
are invoked instead of the simple metrics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    question: str
    expected_answer: str
    generated_answer: str
    relevant_ids: List[str]
    retrieved_ids: List[str]
    precision: float
    recall: float
    faithfulness: bool


def simple_precision_recall(relevant: List[str], retrieved: List[str]) -> (float, float):
    """Compute precision and recall for lists of relevant and retrieved IDs."""
    if not retrieved:
        return 0.0, 0.0
    if not relevant:
        return 0.0, 0.0
    retrieved_set = set(retrieved)
    relevant_set = set(relevant)
    tp = len(retrieved_set & relevant_set)
    precision = tp / len(retrieved)
    recall = tp / len(relevant)
    return precision, recall


def evaluate_dataset(orchestrator, dataset_path: Optional[str] = None) -> Dict:
    """Evaluate the RAG pipeline against a dataset of questions and answers.

    :param orchestrator: an instance of ``SimplexRAGOrchestrator`` with a
        ``answer_question`` method that returns (answer, contexts).
    :param dataset_path: path to JSON dataset file.  If None, uses
        settings.eval_dataset_path.
    :return: dictionary containing aggregate metrics and per‑question results.
    """
    if dataset_path is None:
        dataset_path = settings.eval_dataset_path
    if not dataset_path:
        logger.warning("No evaluation dataset provided; skipping evaluation.")
        return {}
    path = Path(dataset_path)
    if not path.exists():
        logger.error(f"Dataset file not found: {dataset_path}")
        return {}
    data = json.loads(path.read_text())
    results: List[EvalResult] = []
    for item in data:
        question = item.get("question")
        expected = item.get("answer")
        context_ids = item.get("context_ids", [])
        # Ask orchestrator
        answer, contexts = orchestrator.answer_question(question)
        retrieved_ids = [c.get("part_number") for c in contexts]
        precision, recall = simple_precision_recall(context_ids, retrieved_ids)
        # Faithfulness check: ensure expected answer appears in generated answer
        faithfulness = expected.lower() in answer.lower() if expected else True
        results.append(
            EvalResult(
                question=question,
                expected_answer=expected,
                generated_answer=answer,
                relevant_ids=context_ids,
                retrieved_ids=retrieved_ids,
                precision=precision,
                recall=recall,
                faithfulness=faithfulness,
            )
        )
    # Aggregate metrics
    agg_precision = sum(r.precision for r in results) / len(results) if results else 0.0
    agg_recall = sum(r.recall for r in results) / len(results) if results else 0.0
    agg_faith = sum(1.0 if r.faithfulness else 0.0 for r in results) / len(results) if results else 0.0
    report = {
        "precision": agg_precision,
        "recall": agg_recall,
        "faithfulness": agg_faith,
        "results": [r.__dict__ for r in results],
    }
    return report


def run_advanced_evaluation(orchestrator, dataset_path: Optional[str] = None) -> Dict:
    """Placeholder for advanced evaluation using external frameworks.

    If ``use_advanced_eval`` is True and the necessary packages are
    available, this function would integrate with RAGAS, Arize or
    similar libraries to compute metrics such as answer relevance,
    context precision/recall, hallucination detection and more.  In
    this environment we return an empty report.
    """
    logger.warning("Advanced evaluation frameworks are not installed; using simple evaluation instead.")
    return {}