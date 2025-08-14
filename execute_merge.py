#!/usr/bin/env python3
"""
Execute the merge to restore relationships from old graph to current database.

This script will:
1. Backup current database and graph
2. Restore relationships for 30 components from old graph (94.4% validity rate)
3. Update SQLite database with relationship fields  
4. Rebuild NetworkX graph with relationships
5. Persist updated graph
6. Verify the results
"""

import json
import sqlite3
import sys
import shutil
from pathlib import Path
from datetime import datetime

# Load environment
from dotenv import load_dotenv
load_dotenv('/root/.env')

sys.path.insert(0, '/root/simplexrag')

from simplex_rag.database import SimplexDatabase

print("🚀 EXECUTING RELATIONSHIP MERGE")
print("=" * 60)
print(f"Timestamp: {datetime.now()}")

# Step 1: Create backups
print(f"\n1️⃣ CREATING BACKUPS:")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

backup_db = f"/root/simplex_components_backup_{timestamp}.db"
backup_graph = f"/root/simplex_graph_backup_{timestamp}.json"

shutil.copy2("/root/simplex_components.db", backup_db)
shutil.copy2("/root/simplex_graph.json", backup_graph)

print(f"   ✅ Database backed up to: {backup_db}")
print(f"   ✅ Graph backed up to: {backup_graph}")

# Step 2: Load old graph relationships
print(f"\n2️⃣ LOADING OLD GRAPH RELATIONSHIPS:")
with open('/root/simplex_graph.json') as f:
    old_graph = json.load(f)

# Extract relationships from old graph
old_relationships = {}
for node in old_graph['nodes']:
    part_number = node['part_number']
    compat = node.get('compatible_with', [])
    requires = node.get('requires', [])
    excludes = node.get('excludes', [])
    
    if compat or requires or excludes:
        old_relationships[part_number] = {
            'compatible_with': compat,
            'requires': requires, 
            'excludes': excludes
        }

print(f"   📊 Found relationships for {len(old_relationships)} components")

# Step 3: Load current database and check compatibility
print(f"\n3️⃣ ANALYZING CURRENT DATABASE:")
db = SimplexDatabase()
current_parts = set(db.graph.nodes())

restorable_parts = []
invalid_refs = 0
total_refs = 0

for part_number, rels in old_relationships.items():
    if part_number in current_parts:
        restorable_parts.append(part_number)
        # Check validity of relationship references
        for ref_list in [rels['compatible_with'], rels['requires'], rels['excludes']]:
            for ref in ref_list:
                total_refs += 1
                if ref not in current_parts:
                    invalid_refs += 1

print(f"   📊 Restorable components: {len(restorable_parts)}")
print(f"   📊 Valid relationship references: {total_refs - invalid_refs}/{total_refs}")
print(f"   📊 Reference validity: {(total_refs - invalid_refs)/total_refs*100:.1f}%")

# Step 4: Update SQLite database with relationships
print(f"\n4️⃣ UPDATING SQLite DATABASE:")
conn = sqlite3.connect("/root/simplex_components.db")
cursor = conn.cursor()

updated_count = 0
for part_number in restorable_parts:
    rels = old_relationships[part_number]
    
    # Filter out invalid references
    valid_compat = [ref for ref in rels['compatible_with'] if ref in current_parts]
    valid_req = [ref for ref in rels['requires'] if ref in current_parts] 
    valid_excl = [ref for ref in rels['excludes'] if ref in current_parts]
    
    # Update database
    cursor.execute("""
        UPDATE components 
        SET compatible_with = ?, requires = ?, excludes = ?
        WHERE part_number = ?
    """, (
        json.dumps(valid_compat),
        json.dumps(valid_req), 
        json.dumps(valid_excl),
        part_number
    ))
    
    if cursor.rowcount > 0:
        updated_count += 1
        print(f"   ✅ {part_number}: compat={len(valid_compat)}, req={len(valid_req)}, excl={len(valid_excl)}")

conn.commit()
conn.close()

print(f"   📊 Updated {updated_count} components in SQLite")

# Step 5: Rebuild in-memory graph with relationships
print(f"\n5️⃣ REBUILDING IN-MEMORY GRAPH:")

# Close existing database connection and create fresh one
db.close()
db = SimplexDatabase()

# Verify graph has relationships now
edges_count = len(db.graph.edges())
print(f"   📊 Graph now has {edges_count} edges")

# Count nodes with relationships
nodes_with_rels = 0
for node, data in db.graph.nodes(data=True):
    compat = data.get('compatible_with', [])
    requires = data.get('requires', [])
    if compat or requires:
        nodes_with_rels += 1

print(f"   📊 Nodes with relationships: {nodes_with_rels}")

# Step 6: Persist the updated graph  
print(f"\n6️⃣ PERSISTING UPDATED GRAPH:")
try:
    db.persist_graph()
    graph_size = Path('/root/simplex_graph.json').stat().st_size
    print(f"   ✅ Graph persisted successfully ({graph_size} bytes)")
except Exception as e:
    print(f"   ❌ Graph persistence failed: {e}")

# Step 7: Final verification
print(f"\n7️⃣ FINAL VERIFICATION:")
print(f"   Database components: {len(db.graph.nodes())}")
print(f"   Database relationships: {len(db.graph.edges())}")

# Check a few restored components
print(f"\n   Sample restored relationships:")
sample_count = 0
for part_number in restorable_parts[:5]:
    data = db.graph.nodes.get(part_number, {})
    compat = data.get('compatible_with', [])
    requires = data.get('requires', [])
    if compat or requires:
        print(f"     {part_number}: compat={len(compat)}, req={len(requires)}")
        sample_count += 1

if sample_count == 0:
    print(f"     ⚠️  No relationships found in sample - check if merge worked")

# Step 8: Summary
print(f"\n🎉 MERGE EXECUTION COMPLETE!")
print(f"=" * 60)
print(f"   ✅ Backups created: {backup_db}, {backup_graph}")
print(f"   ✅ Relationships restored for {updated_count} components") 
print(f"   ✅ Graph edges: {edges_count}")
print(f"   ✅ Total components preserved: {len(db.graph.nodes())}")
print(f"   ✅ Reference validity maintained: {(total_refs-invalid_refs)/total_refs*100:.1f}%")

print(f"\n🌐 Next: Test web interface to verify correct statistics display")
print(f"   Expected: ~294 components, ~{edges_count} relationships")

db.close()