"""
Hybrid database layer for the Simplex RAG engine.

This module provides classes to persist component data in SQLite while
mirroring that information into an in‑memory graph for fast traversal.
It also exposes utilities to build and query a vector index over
component descriptions and datasheet text.  A hybrid retrieval
function leverages both the vector index and graph to surface
semantically relevant and structurally related information.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any
from enum import Enum

import networkx as nx

try:
    # FAISS is optional; fall back to sklearn if unavailable
    import faiss
except ImportError:
    faiss = None  # type: ignore

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .config import settings
from .data_models import Component, ComponentType, CertificationType, Protocol

logger = logging.getLogger(__name__)


def _component_to_dict(comp: Component) -> Dict[str, Any]:
    """Convert a Component to a dictionary with enum values converted to strings."""
    result = {}
    comp_dict = asdict(comp)
    
    for key, value in comp_dict.items():
        if isinstance(value, Enum):
            # Convert enum to its string value
            result[key] = value.value
        elif isinstance(value, list):
            # Handle lists that might contain enums
            result[key] = []
            for item in value:
                if isinstance(item, Enum):
                    result[key].append(item.value)
                else:
                    result[key].append(item)
        else:
            result[key] = value
    
    return result


class SimplexDatabase:
    """Persistent database coupled with an in‑memory graph and vector index."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or settings.db_path
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.graph = nx.MultiDiGraph()
        # Vector index structures
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._vector_matrix = None  # type: ignore
        self._nearest_neighbors = None  # type: ignore
        # Build schema if empty
        self._initialize_schema()
        # Load data into graph
        self._load_graph_from_db()
        # Load vector index if available
        self._load_vector_index()

    # ------------------------------------------------------------------
    # Database initialisation
    # ------------------------------------------------------------------

    def _initialize_schema(self) -> None:
        """Create database tables if they do not exist."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS components (
                part_number TEXT PRIMARY KEY,
                sku_type TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT NOT NULL,
                manufacturer TEXT,
                document_reference TEXT,
                revision TEXT,
                voltage_nominal REAL,
                voltage_min REAL,
                voltage_max REAL,
                current_supervisory_mA REAL,
                current_alarm_mA REAL,
                power_output_W REAL,
                capacity_points INTEGER,
                capacity_devices INTEGER,
                capacity_circuits INTEGER,
                max_wire_distance_ft INTEGER,
                max_wire_resistance_ohms REAL,
                max_wire_capacitance_uF REAL,
                slot_type TEXT,
                slot_size INTEGER,
                weight_kg REAL,
                dimensions_mm TEXT,
                mounting_location TEXT,
                compatible_panels TEXT,
                compatible_with TEXT,
                requires TEXT,
                excludes TEXT,
                max_per_system INTEGER,
                max_per_panel INTEGER,
                max_per_loop INTEGER,
                max_per_bay INTEGER,
                certifications TEXT,
                unit_cost REAL,
                lead_time_days INTEGER,
                date_created TEXT,
                date_modified TEXT
            )
            """
        )
        self.conn.commit()

    def init_db(self) -> None:
        """
        Initialize the database schema.  This method is provided for
        compatibility with older code that expected an explicit init
        call.  It simply invokes the internal ``_initialize_schema``.
        """
        self._initialize_schema()

    def add_component(self, comp: Component) -> None:
        """Insert or update a component definition in the database."""
        # Convert complex fields to serialised JSON strings
        def serialise(val: Optional[Iterable]):
            if val is None:
                return None
            if isinstance(val, (list, tuple, set)):
                return json.dumps(list(val))
            if isinstance(val, dict):
                return json.dumps(val)
            return val

        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO components (
                    part_number, sku_type, category, description,
                    manufacturer, document_reference, revision,
                    voltage_nominal, voltage_min, voltage_max,
                    current_supervisory_mA, current_alarm_mA,
                    power_output_W, capacity_points, capacity_devices,
                    capacity_circuits, max_wire_distance_ft,
                    max_wire_resistance_ohms, max_wire_capacitance_uF,
                    slot_type, slot_size, weight_kg, dimensions_mm,
                    mounting_location, compatible_panels, compatible_with,
                    requires, excludes, max_per_system, max_per_panel,
                    max_per_loop, max_per_bay, certifications,
                    unit_cost, lead_time_days, date_created, date_modified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    comp.part_number,
                    comp.sku_type,
                    comp.category.value,
                    comp.description,
                    comp.manufacturer,
                    comp.document_reference,
                    comp.revision,
                    comp.voltage_nominal,
                    comp.voltage_min,
                    comp.voltage_max,
                    comp.current_supervisory_mA,
                    comp.current_alarm_mA,
                    comp.power_output_W,
                    comp.capacity_points,
                    comp.capacity_devices,
                    comp.capacity_circuits,
                    comp.max_wire_distance_ft,
                    comp.max_wire_resistance_ohms,
                    comp.max_wire_capacitance_uF,
                    comp.slot_type,
                    comp.slot_size,
                    comp.weight_kg,
                    serialise(comp.dimensions_mm),
                    serialise(comp.mounting_location),
                    serialise(comp.compatible_panels),
                    serialise(comp.compatible_with),
                    serialise(comp.requires),
                    serialise(comp.excludes),
                    comp.max_per_system,
                    comp.max_per_panel,
                    comp.max_per_loop,
                    comp.max_per_bay,
                    serialise([c.value for c in comp.certifications]),
                    comp.unit_cost,
                    comp.lead_time_days,
                    comp.date_created,
                    comp.date_modified,
                ),
            )
        # Update graph
        self.graph.add_node(comp.part_number, **_component_to_dict(comp))
        for compat in comp.compatible_with:
            self.graph.add_edge(comp.part_number, compat, relation="compatible")
        for req in comp.requires:
            self.graph.add_edge(comp.part_number, req, relation="requires")
        for excl in comp.excludes:
            self.graph.add_edge(comp.part_number, excl, relation="excludes")

    def _load_graph_from_db(self) -> None:
        """Load components from the database into the in‑memory graph."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM components")
        rows = cursor.fetchall()
        for row in rows:
            # Parse JSON fields back into Python structures
            def deserialise(value: Optional[str]):
                if value is None:
                    return []
                try:
                    return json.loads(value)
                except Exception:
                    return []

            comp = Component(
                part_number=row["part_number"],
                sku_type=row["sku_type"],
                category=ComponentType(row["category"]),
                description=row["description"],
                manufacturer=row["manufacturer"],
                document_reference=row["document_reference"],
                revision=row["revision"],
                protocols=[Protocol[p] if p in Protocol.__members__ else Protocol.IDNET for p in deserialise(None)],
                voltage_nominal=row["voltage_nominal"],
                voltage_min=row["voltage_min"],
                voltage_max=row["voltage_max"],
                current_supervisory_mA=row["current_supervisory_mA"],
                current_alarm_mA=row["current_alarm_mA"],
                power_output_W=row["power_output_W"],
                capacity_points=row["capacity_points"],
                capacity_devices=row["capacity_devices"],
                capacity_circuits=row["capacity_circuits"],
                max_wire_distance_ft=row["max_wire_distance_ft"],
                max_wire_resistance_ohms=row["max_wire_resistance_ohms"],
                max_wire_capacitance_uF=row["max_wire_capacitance_uF"],
                slot_type=row["slot_type"],
                slot_size=row["slot_size"],
                weight_kg=row["weight_kg"],
                dimensions_mm=json.loads(row["dimensions_mm"]) if row["dimensions_mm"] else None,
                mounting_location=deserialise(row["mounting_location"]),
                compatible_panels=deserialise(row["compatible_panels"]),
                compatible_with=deserialise(row["compatible_with"]),
                requires=deserialise(row["requires"]),
                excludes=deserialise(row["excludes"]),
                max_per_system=row["max_per_system"],
                max_per_panel=row["max_per_panel"],
                max_per_loop=row["max_per_loop"],
                max_per_bay=row["max_per_bay"],
                certifications=[CertificationType[c] for c in deserialise(row["certifications"]) if c in CertificationType.__members__],
                unit_cost=row["unit_cost"],
                lead_time_days=row["lead_time_days"],
                date_created=row["date_created"],
                date_modified=row["date_modified"],
            )
            # Add to graph
            self.graph.add_node(comp.part_number, **_component_to_dict(comp))
            # Add edges
            for compat in comp.compatible_with:
                self.graph.add_edge(comp.part_number, compat, relation="compatible")
            for req in comp.requires:
                self.graph.add_edge(comp.part_number, req, relation="requires")
            for excl in comp.excludes:
                self.graph.add_edge(comp.part_number, excl, relation="excludes")

    # ------------------------------------------------------------------
    # Vector index
    # ------------------------------------------------------------------

    def _load_vector_index(self) -> None:
        """Load a pre‑built vector index from disk if available."""
        if settings.vector_index_path and Path(settings.vector_index_path).exists():
            try:
                # Handle both JSON and pickle files
                if settings.vector_index_path.endswith('.pkl'):
                    import pickle
                    with open(settings.vector_index_path, "rb") as f:
                        data = pickle.load(f)
                else:
                    with open(settings.vector_index_path, "r") as f:
                        data = json.load(f)
                # Build vectorizer and index
                self._vectorizer = TfidfVectorizer().fit(data["documents"])
                self._vector_matrix = self._vectorizer.transform(data["documents"])
                self._nearest_neighbors = NearestNeighbors(metric="cosine")
                self._nearest_neighbors.fit(self._vector_matrix)
                logger.info(
                    f"Loaded vector index from {settings.vector_index_path} with {len(data['documents'])} documents"
                )
            except Exception as e:
                logger.warning(f"Failed to load vector index: {e}. Rebuilding from scratch.")
        else:
            # Rebuild index if not present
            self.rebuild_vector_index()

    def rebuild_vector_index(self) -> None:
        """Rebuild the vector index from component descriptions and graph nodes."""
        documents = []
        part_numbers = []
        for node, data in self.graph.nodes(data=True):
            text = data.get("description", "")
            # Include compatibility lists and requirements to enrich text
            if data.get("compatible_with"):
                text += "\n" + " ".join(data["compatible_with"])
            if data.get("requires"):
                text += "\n" + " ".join(data["requires"])
            documents.append(text)
            part_numbers.append(node)
        if not documents:
            logger.warning("No documents found for vector index. Skipping build.")
            # Initialize empty vectorizer and index to prevent errors
            self._vectorizer = TfidfVectorizer()
            self._vector_matrix = None
            self._nearest_neighbors = None
            return
        # Fit vectorizer
        self._vectorizer = TfidfVectorizer()
        matrix = self._vectorizer.fit_transform(documents)
        self._vector_matrix = matrix
        self._nearest_neighbors = NearestNeighbors(metric="cosine")
        self._nearest_neighbors.fit(matrix)
        # Persist if path provided
        if settings.vector_index_path:
            with open(settings.vector_index_path, "w") as f:
                json.dump({"documents": documents}, f)
            logger.info(f"Persisted vector index to {settings.vector_index_path}")

    # ------------------------------------------------------------------
    # Retrieval functions
    # ------------------------------------------------------------------

    def vector_search(self, query: str, top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """Return a list of part numbers ranked by TF‑IDF similarity to the query."""
        if self._vectorizer is None or self._nearest_neighbors is None:
            logger.warning("Vector index not built; rebuilding now")
            self.rebuild_vector_index()
        # Check if index is still empty after rebuild
        if self._vectorizer is None or self._vector_matrix is None:
            logger.warning("No documents available for vector search")
            return []
        top_k = top_k or settings.vector_top_k
        query_vec = self._vectorizer.transform([query])
        distances, indices = self._nearest_neighbors.kneighbors(query_vec, n_neighbors=min(top_k, len(self.graph)))
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            # Map vector index back to part number
            node_id = list(self.graph.nodes())[idx]
            similarity = 1.0 - dist
            results.append((node_id, similarity))
        return results

    def graph_search(self, part_numbers: Iterable[str], max_depth: Optional[int] = None) -> List[str]:
        """Perform a breadth‑first search from given part numbers up to a specified depth."""
        max_depth = max_depth or settings.graph_max_depth
        visited = set()
        queue: List[Tuple[str, int]] = []
        for pn in part_numbers:
            queue.append((pn, 0))
            visited.add(pn)
        results = list(part_numbers)
        while queue:
            node, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            for neighbor in self.graph.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    results.append(neighbor)
                    queue.append((neighbor, depth + 1))
        return results

    def hybrid_search(self, query: str) -> List[Tuple[str, float]]:
        """Combine vector and graph retrieval to return a ranked list of part numbers."""
        # First perform vector search
        vector_results = self.vector_search(query)
        # Expand via graph
        seeds = [pn for pn, _ in vector_results]
        expanded = self.graph_search(seeds)
        # Score expanded nodes by presence in vector results (1) or adjacency (0.5)
        scores: Dict[str, float] = {}
        for pn, sim in vector_results:
            scores[pn] = max(scores.get(pn, 0.0), sim)
        for pn in expanded:
            if pn not in scores:
                scores[pn] = 0.5  # base score for adjacency
        # Sort by score descending
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        # Return list of (part_number, score)
        return sorted_results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist_graph(self) -> None:
        """Persist the in‑memory graph to disk as JSON."""
        if not settings.graph_persistence_path:
            return
        data = {
            "nodes": [
                {"part_number": n, **d} for n, d in self.graph.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, **d} for u, v, d in self.graph.edges(data=True)
            ],
        }
        with open(settings.graph_persistence_path, "w") as f:
            json.dump(data, f)
        logger.info(f"Persisted graph to {settings.graph_persistence_path}")

    def close(self) -> None:
        """Close the database connection and persist caches."""
        if settings.graph_persistence_path:
            self.persist_graph()
        self.conn.close()
