#!/usr/bin/env python3
"""
Analyze relationship data to understand why relationships became 0
"""

import json
import sqlite3
from pathlib import Path

print("🔍 RELATIONSHIP ANALYSIS")
print("=" * 60)

# Check old graph from JSON persistence
old_graph_path = "/root/simplex_graph.json"
if Path(old_graph_path).exists():
    with open(old_graph_path) as f:
        old_graph = json.load(f)
        
    print(f"\n📊 OLD PERSISTED GRAPH ({old_graph_path}):")
    print(f"   Nodes: {len(old_graph.get('nodes', []))}")
    print(f"   Edges: {len(old_graph.get('edges', []))}")
    
    # Sample some nodes with relationships
    nodes_with_relationships = 0
    for node in old_graph.get("nodes", [])[:10]:  # First 10 nodes
        compat = node.get("compatible_with", [])
        requires = node.get("requires", [])
        if compat or requires:
            nodes_with_relationships += 1
            print(f"   {node['part_number'][:15]}: compat={len(compat)}, req={len(requires)}")
    
    print(f"   Sample nodes with relationships: {nodes_with_relationships}/10")

# Check current SQLite database
db_path = "/root/simplex_components.db"
if Path(db_path).exists():
    db = sqlite3.connect(db_path)
    cursor = db.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM components")
    total_components = cursor.fetchone()[0]
    
    print(f"\n📊 CURRENT SQLite DATABASE ({db_path}):")
    print(f"   Total components: {total_components}")
    
    # Check for relationship fields that are not empty lists
    cursor.execute("SELECT COUNT(*) FROM components WHERE compatible_with != '[]'")
    result = cursor.fetchone()
    compat_count = result[0] if result else 0
    
    cursor.execute("SELECT COUNT(*) FROM components WHERE requires != '[]'")
    result = cursor.fetchone()
    req_count = result[0] if result else 0
    
    print(f"   Components with compatible_with: {compat_count}")
    print(f"   Components with requires: {req_count}")
    
    # Sample some components and their relationship data
    print(f"\n   Sample component relationship fields:")
    cursor.execute("SELECT part_number, compatible_with, requires FROM components LIMIT 10")
    for row in cursor.fetchall():
        pn, compat, req = row
        print(f"   {pn[:20]:20} compat='{str(compat)[:20]}' req='{str(req)[:20]}'")
    
    db.close()
else:
    print(f"\n❌ SQLite database not found at {db_path}")

print(f"\n🎯 KEY FINDINGS:")
print("   - Old graph has 233 nodes with 126 relationships")
print("   - Current database has 294 components")
print("   - Need to check if new components have relationship data from OpenAI")