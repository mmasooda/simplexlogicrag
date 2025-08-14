"""
Command‑line utility to ingest datasheets into the Simplex RAG database.

Usage::

    python scripts/run_ingestion.py --dir path/to/datasheets

This script initialises the Simplex database, loads any existing
components, ingests all supported files in the specified directory
using the advanced ingestion pipeline, rebuilds the vector index, and
persists the graph and vector index to disk.  This allows the rest
of the RAG engine to operate without requiring repeated ingestion.
"""

import argparse
import logging
from pathlib import Path
import os

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv('/root/.env')

from simplex_rag.orchestrator import SimplexRAGOrchestrator

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Simplex datasheets into the RAG database")
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Directory containing datasheets (PDF, Word, Excel) to ingest",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    orch = SimplexRAGOrchestrator()
    dir_path = Path(args.dir)
    orch.ingest_directory(str(dir_path))
    print(f"Ingested datasheets from {dir_path}")

if __name__ == "__main__":
    main()