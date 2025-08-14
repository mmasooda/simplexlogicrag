#!/usr/bin/env python3
"""
Monitor batch job completion and properly complete the ingestion.
This addresses the root cause: batch timeout is too short.
"""

from dotenv import load_dotenv
load_dotenv('/root/.env')

import sys
import time
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '/root/simplexrag')

from openai import OpenAI
from simplex_rag.database import SimplexDatabase
from simplex_rag.ingestion.datasheet_ingestor import ingest_file
from simplex_rag.data_models import Component, ComponentType, Protocol, CertificationType

print("🔍 MONITORING BATCH JOB COMPLETION")
print("=" * 60)

# Find the in-progress batch
client = OpenAI()
batches = client.batches.list(limit=10)

active_batch = None
for batch in batches.data:
    if batch.status in ['validating', 'in_progress', 'finalizing']:
        active_batch = batch
        break

if not active_batch:
    print("❌ No active batch job found")
    # Check for recently completed batches
    for batch in batches.data[:3]:
        age = (time.time() - batch.created_at) / 60
        if age < 30:  # Within last 30 minutes
            print(f"\nRecent batch: {batch.id}")
            print(f"  Status: {batch.status}")
            print(f"  Age: {age:.1f} minutes")
            if batch.status == 'completed' and batch.output_file_id:
                active_batch = batch
                print("  ✅ Using this completed batch")
                break
    
    if not active_batch:
        sys.exit(1)

print(f"\n📋 Batch job: {active_batch.id}")
print(f"   Status: {active_batch.status}")
print(f"   Created: {datetime.fromtimestamp(active_batch.created_at)}")

# If still processing, wait for it
if active_batch.status in ['validating', 'in_progress', 'finalizing']:
    print(f"\n⏳ Waiting for batch to complete...")
    print("   (This is normal - batch jobs can take 10-30 minutes)")
    
    start_wait = time.time()
    while active_batch.status not in ['completed', 'failed', 'expired', 'cancelled']:
        time.sleep(10)
        active_batch = client.batches.retrieve(active_batch.id)
        elapsed = (time.time() - start_wait) / 60
        print(f"   Status: {active_batch.status} ({elapsed:.1f} minutes waiting)")
        
        # Safety limit - 60 minutes
        if elapsed > 60:
            print("   ⚠️  Exceeded 60 minute wait limit")
            break

print(f"\n📊 Final batch status: {active_batch.status}")

if active_batch.status != 'completed':
    print(f"❌ Batch did not complete successfully")
    sys.exit(1)

# Download and process results
print(f"\n📥 Downloading batch results...")
result_file = client.files.content(active_batch.output_file_id)
result_text = result_file.read().decode()

# Parse results
responses = {}
for line in result_text.splitlines():
    obj = json.loads(line)
    custom_id = obj.get("custom_id")
    # Navigate the correct structure
    response = obj.get("response", {})
    body = response.get("body", {})
    choices = body.get("choices", [])
    content = ""
    if choices:
        content = choices[0].get("message", {}).get("content", "")
    responses[custom_id] = content

print(f"   ✅ Downloaded {len(responses)} responses")

# Process the files in test directory
test_dir = Path('/root/test_20_more_files')
if not test_dir.exists():
    print(f"\n❌ Test directory not found")
    sys.exit(1)

print(f"\n🔄 Processing components with batch results...")

# Initialize database
db = SimplexDatabase()
initial_count = len(db.graph.nodes())

# Process each file and update with batch results
components_added = 0
for pdf_file in test_dir.glob('*.pdf'):
    print(f"\n📄 Processing {pdf_file.name}...")
    
    # Extract components from the file (without LLM since we have batch results)
    components = ingest_file(pdf_file, llm=None)
    
    # Update components with LLM results
    for comp in components:
        result = responses.get(comp.part_number, "")
        if result:
            try:
                data = json.loads(result)
                
                # Update component with LLM data
                if data.get("category") in ComponentType.__members__:
                    comp.category = ComponentType[data["category"]]
                if data.get("description"):
                    comp.description = data["description"]
                if data.get("protocols"):
                    comp.protocols = [Protocol[p] for p in data.get("protocols", []) 
                                    if p in Protocol.__members__]
                comp.slot_size = data.get("slot_size", comp.slot_size)
                comp.compatible_with = data.get("compatible_with", [])
                comp.requires = data.get("requires", [])
                comp.excludes = data.get("excludes", [])
                if data.get("max_devices"):
                    comp.capacity_devices = data["max_devices"]
                if data.get("certifications"):
                    comp.certifications = [CertificationType[c] for c in data.get("certifications", []) 
                                         if c in CertificationType.__members__]
                
                print(f"   ✅ Enhanced {comp.part_number}: {comp.description[:50]}...")
            except Exception as e:
                print(f"   ⚠️  Failed to parse result for {comp.part_number}: {e}")
        
        # Add to database
        try:
            db.add_component(comp)
            components_added += 1
        except Exception as e:
            print(f"   ❌ Failed to add {comp.part_number}: {e}")

# Rebuild vector index and persist
print(f"\n🔨 Rebuilding vector index...")
db.rebuild_vector_index()

print(f"\n💾 Persisting graph...")
db.persist_graph()

# Final stats
final_count = len(db.graph.nodes())
print(f"\n✅ BATCH PROCESSING COMPLETE")
print(f"   Components: {initial_count} → {final_count} (+{final_count - initial_count})")
print(f"   Components processed: {components_added}")

# Clean up
import shutil
if test_dir.exists():
    shutil.rmtree(test_dir)
    print(f"\n🧹 Cleaned up test directory")