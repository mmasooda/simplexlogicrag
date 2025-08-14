"""
Document ingestion utilities.

This subpackage provides functions to parse datasheets in PDF,
Word and Excel formats.  It extracts product part numbers,
compatibility relationships, capacities and other metadata from
unstructured text.  For complex tables or where heuristics fail,
an optional LLM interface can be used via the batch API.

The entry point is the ``ingest_datasheets`` function which
iterates over files in a directory and populates the database.
"""

from .datasheet_ingestor import ingest_datasheets  # noqa: F401