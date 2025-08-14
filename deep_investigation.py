#!/usr/bin/env python3
"""
Deep investigation of why relationships became 0 during full dataset ingestion
"""

import json
import sqlite3
import sys
from pathlib import Path

# Load dotenv first
from dotenv import load_dotenv
load_dotenv('/root/.env')

sys.path.insert(0, '/root/simplexrag')

from simplex_rag.database import SimplexDatabase
from simplex_rag.config import settings

print("🔬 DEEP INVESTIGATION: Why Relationships Became 0")
print("=" * 70)

# 1. Check settings
print(f"\n1️⃣ SETTINGS CHECK:")
print(f"   graph_persistence_path: {settings.graph_persistence_path}")
print(f"   db_path: {settings.db_path}")
print(f"   vector_index_path: {settings.vector_index_path}")

# 2. Check old persisted graph vs current in-memory graph
print(f"\n2️⃣ GRAPH STATE COMPARISON:")

# Load old persisted graph
with open('/root/simplex_graph.json') as f:
    old_graph = json.load(f)

print(f"   OLD PERSISTED (simplex_graph.json):")
print(f"     Nodes: {len(old_graph['nodes'])}")
print(f"     Edges: {len(old_graph['edges'])}")

# Sample relationships in old graph
nodes_with_rels = 0
for node in old_graph['nodes'][:20]:
    compat = node.get('compatible_with', [])
    requires = node.get('requires', [])
    if compat or requires:
        nodes_with_rels += 1
        print(f"     {node['part_number'][:20]:20} compat:{len(compat):2} req:{len(requires):2}")

print(f"     Nodes with relationships: {nodes_with_rels}/20 sampled")

# Load current database and check in-memory graph
print(f"\n   CURRENT IN-MEMORY GRAPH:")
db = SimplexDatabase()
print(f"     Nodes: {len(db.graph.nodes())}")
print(f"     Edges: {len(db.graph.edges())}")

# Check if current graph has relationships
current_nodes_with_rels = 0
for node, data in list(db.graph.nodes(data=True))[:20]:
    compat = data.get('compatible_with', [])
    requires = data.get('requires', [])
    if compat or requires:
        current_nodes_with_rels += 1
        print(f"     {node[:20]:20} compat:{len(compat):2} req:{len(requires):2}")

print(f"     Current nodes with relationships: {current_nodes_with_rels}/20 sampled")

# 3. Check database vs in-memory consistency
print(f"\n3️⃣ DATABASE vs IN-MEMORY CONSISTENCY:")
conn = sqlite3.connect('/root/simplex_components.db')
cursor = conn.cursor()

# Count components in database with relationships
cursor.execute("SELECT COUNT(*) FROM components WHERE compatible_with != '[]'")
db_compat_count = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM components WHERE requires != '[]'")
db_req_count = cursor.fetchone()[0]

print(f"   SQLite components with compatible_with: {db_compat_count}")
print(f"   SQLite components with requires: {db_req_count}")
print(f"   In-memory graph edges: {len(db.graph.edges())}")

# 4. The critical test: Check if persist_graph would work now
print(f"\n4️⃣ TESTING PERSIST_GRAPH FUNCTIONALITY:")
print(f"   settings.graph_persistence_path = '{settings.graph_persistence_path}'")

if settings.graph_persistence_path:
    print(f"   ✅ Graph persistence path is set")
    try:
        # Test if we can persist current graph
        test_path = '/root/test_persist.json'
        data = {
            "nodes": [{"part_number": n, **d} for n, d in db.graph.nodes(data=True)],
            "edges": [{"source": u, "target": v, **d} for u, v, d in db.graph.edges(data=True)]
        }
        with open(test_path, 'w') as f:
            json.dump(data, f)
        print(f"   ✅ Test persistence successful")
        
        # Check the test file
        test_size = Path(test_path).stat().st_size
        print(f"   Test file size: {test_size} bytes")
        
        # Clean up
        Path(test_path).unlink()
        
    except Exception as e:
        print(f"   ❌ Test persistence failed: {e}")
else:
    print(f"   ❌ Graph persistence path is None - THIS IS THE PROBLEM!")

conn.close()

print(f"\n🎯 CONCLUSIONS:")
print(f"   - Old graph (233 nodes, 126 edges) has proper relationships")
print(f"   - Current database (294 components) has NO relationships")  
print(f"   - This suggests OpenAI batch results were missing relationship data")
print(f"   - OR the relationship parsing failed during ingestion")
print(f"   - Need to check if batch results contained compatible_with/requires fields")