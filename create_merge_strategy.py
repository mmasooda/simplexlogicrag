#!/usr/bin/env python3
"""
Create merge strategy to combine:
1. Old graph relationships (233 components with 126 proper relationships)
2. New components from full ingestion (61 additional components)

This will restore relationship data while keeping the expanded component set.
"""

import json
import sqlite3
import sys
from pathlib import Path
from datetime import datetime

# Load environment
from dotenv import load_dotenv
load_dotenv('/root/.env')

sys.path.insert(0, '/root/simplexrag')

from simplex_rag.database import SimplexDatabase

print("🔀 MERGE STRATEGY CREATION")
print("=" * 60)
print(f"Timestamp: {datetime.now()}")

# Load old graph with relationships
with open('/root/simplex_graph.json') as f:
    old_graph = json.load(f)

# Load current database
db = SimplexDatabase()

print(f"\n📊 DATA INVENTORY:")
print(f"   Old graph nodes: {len(old_graph['nodes'])}")  
print(f"   Old graph edges: {len(old_graph['edges'])}")
print(f"   Current database components: {len(db.graph.nodes())}")
print(f"   Current database edges: {len(db.graph.edges())}")

# Analyze overlap and differences
old_part_numbers = {node['part_number'] for node in old_graph['nodes']}
current_part_numbers = set(db.graph.nodes())

common_parts = old_part_numbers.intersection(current_part_numbers)
old_only = old_part_numbers - current_part_numbers
new_only = current_part_numbers - old_part_numbers

print(f"\n🔍 COMPONENT ANALYSIS:")
print(f"   Common components (in both): {len(common_parts)}")
print(f"   Old-only components (missing in new): {len(old_only)}")  
print(f"   New-only components (added in full ingestion): {len(new_only)}")

if old_only:
    print(f"\n   Components missing from new database:")
    for part in sorted(list(old_only)[:10]):
        print(f"     - {part}")
    if len(old_only) > 10:
        print(f"     ... and {len(old_only) - 10} more")

if new_only:
    print(f"\n   New components from full ingestion:")
    for part in sorted(list(new_only)[:10]):
        print(f"     - {part}")
    if len(new_only) > 10:
        print(f"     ... and {len(new_only) - 10} more")

# Create mapping of old graph relationships
print(f"\n🔗 RELATIONSHIP MAPPING:")
old_relationships = {}
for node in old_graph['nodes']:
    part_number = node['part_number']
    compat = node.get('compatible_with', [])
    requires = node.get('requires', [])
    
    if compat or requires:
        old_relationships[part_number] = {
            'compatible_with': compat,
            'requires': requires
        }

print(f"   Components with relationships in old graph: {len(old_relationships)}")

# Check how many of these components still exist
restorable_relationships = 0
for part_number in old_relationships:
    if part_number in current_part_numbers:
        restorable_relationships += 1

print(f"   Relationships restorable (component still exists): {restorable_relationships}")

# Create merge plan
print(f"\n📋 MERGE STRATEGY:")
print(f"   1. Keep all {len(current_part_numbers)} components from current database")
print(f"   2. Restore relationships for {restorable_relationships} components from old graph")
print(f"   3. New components ({len(new_only)}) will have empty relationships (no data available)")
print(f"   4. Components lost in transition ({len(old_only)}) cannot be restored without re-ingestion")

# Validate merge feasibility
print(f"\n✅ MERGE FEASIBILITY CHECK:")

# Check if old relationships reference components that still exist
invalid_references = 0
total_references = 0

for part_number, rels in old_relationships.items():
    if part_number in current_part_numbers:  # Only check relationships for components we can restore
        for target in rels['compatible_with'] + rels['requires']:
            total_references += 1
            if target not in current_part_numbers:
                invalid_references += 1

print(f"   Total relationship references: {total_references}")
print(f"   Invalid references (target component missing): {invalid_references}")  
print(f"   Valid references: {total_references - invalid_references}")
print(f"   Reference validity: {(total_references - invalid_references) / total_references * 100:.1f}%" if total_references > 0 else "N/A")

if invalid_references / total_references < 0.1 if total_references > 0 else True:  # Less than 10% invalid
    print(f"   🟢 MERGE IS FEASIBLE - Low invalid reference rate")
else:
    print(f"   🟡 MERGE CAUTION - High invalid reference rate")

print(f"\n🚀 RECOMMENDED ACTION:")
print(f"   Execute merge script to:")
print(f"   1. Backup current database")
print(f"   2. Restore {restorable_relationships} component relationships from old graph")
print(f"   3. Rebuild NetworkX graph with restored relationships")
print(f"   4. Persist updated graph to simplex_graph.json")
print(f"   5. Verify web interface shows correct statistics")

# Generate merge script
merge_script_path = "/root/execute_merge.py"
print(f"\n📄 Next step: Run {merge_script_path} to execute the merge")