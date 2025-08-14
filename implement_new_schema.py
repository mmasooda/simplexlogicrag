#!/usr/bin/env python3
"""
Implement the new graph schema with correct compatibility data
Based on ChatGPT-5 recommendations
"""

import sys
from pathlib import Path
from datetime import datetime

# Load environment
from dotenv import load_dotenv
load_dotenv('/root/.env')

sys.path.insert(0, '/root/simplexrag')

from simplex_rag.database import SimplexDatabase
from simplex_rag.data_models import Component, ComponentType, Protocol, CertificationType

print("🔄 IMPLEMENTING NEW GRAPH SCHEMA WITH CORRECT COMPATIBILITY DATA")
print("=" * 80)

# Initialize fresh database
db = SimplexDatabase()

# Clear existing data
print("1️⃣ Clearing existing incorrect data...")
db.graph.clear()

# Define correct compatibility data based on ChatGPT-5 analysis
COMPATIBILITY_DATA = {
    "4098-9714": {  # Photoelectric Smoke Detector Head
        "description": "TrueAlarm Photoelectric Smoke Detector Head",
        "category": ComponentType.SENSOR,
        "compatible_bases": [
            {"model": "4098-9792", "function": "standard", "description": "Standard sensor base (white)"},
            {"model": "4098-9789", "function": "relay", "description": "Base with connections for remote LED or unsupervised relay"},
            {"model": "4098-9791", "function": "relay_supervised", "description": "4-wire base for supervised remote relay", "caveats": ["not 2120 CDT"]},
            {"model": "4098-9780", "function": "relay_supervised", "description": "2-wire base for supervised remote relay", "caveats": ["not 2120 CDT"]},
            {"model": "4098-9794", "function": "sounder", "description": "Integrated piezo sounder base"},
            {"model": "4098-9793", "function": "isolator", "description": "IDNet isolator base (provides SLC isolation)"},
            {"model": "4098-9766", "function": "isolator", "description": "Isolator2 base (newer IDNet isolator version)"},
            {"model": "4098-9777", "function": "isolator", "description": "Isolator2 base (newer IDNet isolator version)"},
            {"model": "4098-9770", "function": "co_base", "description": "CO base (standard)"},
            {"model": "4098-9771", "function": "co_sounder", "description": "CO sounder base"}
        ]
    },
    "4098-9754": {  # Photo/Heat Multi-Sensor Head
        "description": "TrueAlarm Photo/Heat Multi-Sensor Detector Head", 
        "category": ComponentType.SENSOR,
        "compatible_bases": [
            {"model": "4098-9796", "function": "multi_sensor", "description": "Multi-Sensor standard base that supports two sequential addresses"},
            {"model": "4098-9795", "function": "multi_sensor_sounder", "description": "Multi-Sensor base with built-in piezoelectric sounder (88 dBA)"},
            {"model": "4098-9792", "function": "standard", "description": "Standard sensor base"},
            {"model": "4098-9789", "function": "relay", "description": "Relay base (connections for remote LED or unsupervised relay)"},
            {"model": "4098-9794", "function": "sounder", "description": "Sounder base"},
            {"model": "4098-9791", "function": "relay_supervised", "description": "4-wire relay base (supervised)"},
            {"model": "4098-9793", "function": "isolator", "description": "IDNet isolator base"}
        ]
    },
    "4098-9733": {  # Heat Detector Head
        "description": "TrueAlarm Heat Detector Head",
        "category": ComponentType.SENSOR, 
        "compatible_bases": [
            {"model": "4098-9792", "function": "standard", "description": "Standard sensor base"},
            {"model": "4098-9789", "function": "relay", "description": "Sensor base with terminal connections for a remote LED or unsupervised relay"},
            {"model": "4098-9791", "function": "relay_supervised", "description": "Four-wire base for supervised relay", "caveats": ["not 2120 CDT"]},
            {"model": "4098-9780", "function": "relay_supervised", "description": "Two-wire base for supervised relay", "caveats": ["not 2120 CDT"]},
            {"model": "4098-9793", "function": "isolator", "description": "IDNet isolator base"},
            {"model": "4098-9794", "function": "sounder", "description": "Sounder base (integrated audible notification)"},
            {"model": "4098-9770", "function": "co_base", "description": "CO sensor base (standard)"},
            {"model": "4098-9771", "function": "co_sounder", "description": "CO sensor base with sounder"}
        ]
    }
}

print("2️⃣ Creating detector heads with correct data...")

# Create all unique bases first
all_bases = {}
for head_data in COMPATIBILITY_DATA.values():
    for base_info in head_data["compatible_bases"]:
        model = base_info["model"]
        if model not in all_bases:
            all_bases[model] = {
                "model": model,
                "function": base_info["function"],
                "description": base_info["description"],
                "caveats": base_info.get("caveats", [])
            }

print(f"   Creating {len(all_bases)} unique base components...")
for base_model, base_info in all_bases.items():
    base_component = Component(
        part_number=base_model,
        sku_type="product",
        category=ComponentType.BASE,
        description=base_info["description"],
        protocols=[Protocol.IDNET],
        certifications=[CertificationType.UL]
    )
    
    # Add function as metadata
    base_component.description = f"{base_info['description']} [{base_info['function']}]"
    
    db.add_component(base_component)
    print(f"   ✅ Added base: {base_model} - {base_info['function']}")

print("3️⃣ Creating detector heads and compatibility relationships...")

for head_model, head_data in COMPATIBILITY_DATA.items():
    # Create head component
    head_component = Component(
        part_number=head_model,
        sku_type="product", 
        category=ComponentType.SENSOR,
        description=head_data["description"],
        protocols=[Protocol.IDNET],
        certifications=[CertificationType.UL]
    )
    
    # Set correct compatibility relationships
    compatible_bases = [base["model"] for base in head_data["compatible_bases"]]
    head_component.compatible_with = compatible_bases
    
    db.add_component(head_component)
    print(f"   ✅ Added head: {head_model} -> {len(compatible_bases)} bases")

print("4️⃣ Graph relationships built automatically during component addition...")

print("5️⃣ Persisting updated graph...")
db.persist_graph()

print("6️⃣ Rebuilding vector index...")  
db.rebuild_vector_index()

# Final verification
final_components = len(db.graph.nodes())
final_relationships = len(db.graph.edges())

print(f"\n🎉 NEW SCHEMA IMPLEMENTATION COMPLETE!")
print("=" * 80)
print(f"   Components: {final_components}")
print(f"   Relationships: {final_relationships}")

# Test the new data with the original failing query
print(f"\n🧪 TESTING WITH ORIGINAL FAILING QUERY:")
print("Query: 'what all bases compatible with Simplex 4098-9714 Head'")

# Get compatibility data for 4098-9714
if "4098-9714" in db.graph.nodes():
    node_data = db.graph.nodes["4098-9714"]
    compatible_bases = node_data.get("compatible_with", [])
    
    print(f"New answer: {len(compatible_bases)} bases found:")
    for base in sorted(compatible_bases):
        if base in db.graph.nodes():
            base_data = db.graph.nodes[base]
            description = base_data.get("description", "")
            print(f"   ✅ {base}: {description}")
        else:
            print(f"   ⚠️  {base}: (base not found in graph)")
else:
    print("   ❌ Head 4098-9714 not found in graph")

print(f"\n💡 COMPARISON:")
print("   Old wrong answer: 4098-9788, 4098-9684, 4098-9615, 4098-9685, 4098-9407, 4098-9408, 2098-9201, 2098-9202, 2098-9203")
print("   New correct bases: " + ", ".join(sorted(compatible_bases)) if '4098-9714' in db.graph.nodes() else "None")

db.close()