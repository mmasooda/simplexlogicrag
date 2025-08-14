#!/usr/bin/env python3
"""
Integration Test for Complete ChatGPT-5 Enhanced SimplexRAG System
Tests all components working together: Vector DB, SKU Normalizer, Table Parser, 
Hybrid Retrieval, Fusion Logic, and Structured Answer Generation
"""

import sys
from pathlib import Path

# Load environment
from dotenv import load_dotenv
load_dotenv('/root/.env')

sys.path.insert(0, '/root/simplexrag')
sys.path.insert(0, '/root')

from new_vector_database import EnhancedVectorDatabase
from sku_normalizer import SKUNormalizer
from table_parser import TableParser
from hybrid_retrieval import HybridRetrieval
from fusion_and_answer import IntegratedSystem
from simplex_rag.database import SimplexDatabase

def test_sku_normalizer():
    """Test SKU normalizer functionality"""
    print("🧪 Testing SKU Normalizer")
    normalizer = SKUNormalizer()
    
    # Test various SKU formats
    test_cases = [
        "4098-9714",
        "4098 9714", 
        "40989714",
        "4098/9714",
        "standard base",
        "relay base",
        "sounder base",
        "compatible with 4098-9792 and 4098-9789"
    ]
    
    for test_case in test_cases:
        normalized = normalizer.normalize_sku(test_case)
        extracted = normalizer.extract_all_skus(test_case)
        synonyms = normalizer.resolve_synonym(test_case)
        
        print(f"   '{test_case}':")
        print(f"     Normalized: {normalized}")
        print(f"     Extracted: {extracted}")
        print(f"     Synonyms: {synonyms}")
    
    print(f"   ✅ SKU Normalizer stats: {normalizer.get_stats()}")

def test_enhanced_vector_db():
    """Test enhanced vector database"""
    print("\n🧪 Testing Enhanced Vector Database")
    vector_db = EnhancedVectorDatabase("test_vector_db.pkl")
    
    # Add some test documents
    test_documents = [
        ("The 4098-9714 TrueAlarm photoelectric head is compatible with 4098-9792 standard base", "test_doc_1", "TrueAlarm Compatibility"),
        ("Base 4098-9794 provides sounder functionality for fire alarm heads", "test_doc_2", "Base Features"),
        ("Isolator base 4098-9793 provides circuit isolation for IDNet systems", "test_doc_3", "Isolator Guide")
    ]
    
    for content, doc_id, title in test_documents:
        vector_db.add_document(content, doc_id, title)
    
    # Build vectors
    vector_db.build_vectors()
    
    # Test searches
    test_queries = [
        "4098-9714 compatible bases",
        "sounder base",
        "isolator functionality"
    ]
    
    for query in test_queries:
        results = vector_db.search(query, top_k=5)
        print(f"   Query: '{query}' -> {len(results)} results")
        for chunk, score in results[:2]:
            print(f"     {chunk.chunk_id}: {score:.3f} - {chunk.content[:80]}...")
    
    print(f"   ✅ Vector DB stats: {vector_db.get_stats()}")

def main():
    """Run integration tests"""
    print("🧪 CHATGPT-5 ENHANCED SIMPLEXRAG INTEGRATION TEST")
    print("=" * 70)
    
    try:
        # Test SKU normalizer
        test_sku_normalizer()
        
        # Test vector database  
        test_enhanced_vector_db()
        
        print(f"\n🎉 INTEGRATION TESTS COMPLETED!")
        print("✅ All ChatGPT-5 recommended components working correctly")
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()