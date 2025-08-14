#!/usr/bin/env python3
"""
Ingest the complete dataset directory and monitor progress.
No code changes - just run existing system and report results.
"""

from dotenv import load_dotenv
load_dotenv('/root/.env')

import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '/root/simplexrag')

from simplex_rag.orchestrator import SimplexRAGOrchestrator

print("🚀 FULL DATASET INGESTION MONITORING")
print("=" * 80)
print(f"Start time: {datetime.now()}")

# Check initial state
print("\n📊 Initial Database State:")
orch = SimplexRAGOrchestrator()
initial_components = len(orch.db.graph.nodes())
initial_relationships = len(orch.db.graph.edges())

print(f"   Starting components: {initial_components}")
print(f"   Starting relationships: {initial_relationships}")

# Count files to be processed
dataset_path = Path('/root/dataset')
pdf_files = list(dataset_path.glob('**/*.pdf'))
doc_files = list(dataset_path.glob('**/*.doc*'))
xls_files = list(dataset_path.glob('**/*.xls*'))

total_files = len(pdf_files) + len(doc_files) + len(xls_files)

print(f"\n📁 Files to Process:")
print(f"   PDF files: {len(pdf_files)}")
print(f"   Word files: {len(doc_files)}")
print(f"   Excel files: {len(xls_files)}")
print(f"   Total files: {total_files}")

# Start ingestion
print(f"\n🔄 Starting full dataset ingestion...")
print(f"   Dataset path: {dataset_path}")
print(f"   This may take 30-60 minutes depending on batch processing...")

start_time = time.time()

try:
    # Run the existing ingestion system as-is
    orch.ingest_directory(str(dataset_path))
    
    ingestion_time = time.time() - start_time
    print(f"\n✅ Ingestion completed successfully!")
    print(f"   Total time: {ingestion_time/60:.1f} minutes")
    
except Exception as e:
    ingestion_time = time.time() - start_time
    print(f"\n❌ Ingestion failed after {ingestion_time/60:.1f} minutes")
    print(f"   Error: {type(e).__name__}: {e}")
    print(f"\n   Will still check what was processed...")

# Check final state
print(f"\n📊 Final Database State:")
final_components = len(orch.db.graph.nodes())
final_relationships = len(orch.db.graph.edges())

print(f"   Final components: {final_components}")
print(f"   Final relationships: {final_relationships}")
print(f"   Growth: +{final_components - initial_components} components")
print(f"   Growth: +{final_relationships - initial_relationships} relationships")

print(f"\n🔍 ANALYSIS OF COMPONENT QUALITY:")

# Analyze component part numbers vs descriptive names
valid_part_numbers = 0
descriptive_names = 0
none_slot_size = 0
zero_slot_size = 0

descriptive_examples = []
none_slot_examples = []

for part_number, data in orch.db.graph.nodes(data=True):
    # Check if part number looks like a model number vs descriptive text
    if (len(part_number) <= 20 and 
        not ' ' in part_number and 
        any(c.isdigit() for c in part_number) and
        not part_number.startswith(('compatible', 'requires', 'SmartSync', 'flush', 'surface', 'ceiling'))):
        valid_part_numbers += 1
    else:
        descriptive_names += 1
        if len(descriptive_examples) < 10:
            descriptive_examples.append(part_number[:60])
    
    # Check slot_size
    slot_size = data.get('slot_size')
    if slot_size is None:
        none_slot_size += 1
        if len(none_slot_examples) < 5:
            none_slot_examples.append(part_number[:40])
    elif slot_size == 0:
        zero_slot_size += 1

print(f"\n📈 Component Analysis Results:")
print(f"   Valid part numbers: {valid_part_numbers}")
print(f"   Descriptive names: {descriptive_names}")
print(f"   Components with None slot_size: {none_slot_size}")
print(f"   Components with zero slot_size: {zero_slot_size}")

if descriptive_examples:
    print(f"\n   Examples of descriptive names:")
    for example in descriptive_examples:
        print(f"     - {example}")

if none_slot_examples:
    print(f"\n   Examples with None slot_size:")
    for example in none_slot_examples:
        print(f"     - {example}")

# Summary
print(f"\n" + "=" * 80)
print(f"📋 INGESTION SUMMARY")
print(f"=" * 80)
print(f"Files processed: {total_files}")
print(f"Database growth: {initial_components} → {final_components} components")
print(f"Quality issues:")
print(f"  - Descriptive names: {descriptive_names} components")
print(f"  - None slot_size: {none_slot_size} components")
print(f"  - Zero slot_size: {zero_slot_size} components")
print(f"Processing time: {(time.time() - start_time)/60:.1f} minutes")
print(f"End time: {datetime.now()}")