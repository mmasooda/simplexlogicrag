#!/usr/bin/env python3
"""
Hybrid Retrieval System for SimplexRAG based on ChatGPT-5 recommendations
Combines keyword/BM25 + vector search + graph traversal with intelligent fusion
"""

import re
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from collections import Counter
import numpy as np

# Import our custom components
from sku_normalizer import SKUNormalizer
from new_vector_database import EnhancedVectorDatabase, DocumentChunk

@dataclass
class RetrievalResult:
    """Single retrieval result with source and confidence"""
    content: str
    source_type: str        # 'graph', 'vector', 'keyword'
    sku_model: str          # Related SKU
    function_type: str      # standard, relay, sounder, etc.
    confidence: float       # 0.0-1.0
    source_doc: str         # Document source
    metadata: Dict[str, Any]  # Additional metadata

@dataclass
class EvidenceBundle:
    """Evidence bundle for answer generation"""
    query: str
    primary_sku: str
    related_skus: List[str]
    compatibility_records: List[Dict[str, Any]]
    function_groups: Dict[str, List[str]]  # function -> [skus]
    citations: List[Dict[str, Any]]
    confidence: float
    reasoning: List[str]

class HybridRetrieval:
    """Hybrid retrieval system combining multiple approaches"""
    
    def __init__(self, graph_db, vector_db: EnhancedVectorDatabase, sku_normalizer: SKUNormalizer):
        self.graph_db = graph_db  # SimplexDatabase instance
        self.vector_db = vector_db
        self.sku_normalizer = sku_normalizer
        
        # Retrieval parameters following ChatGPT-5 recommendations
        self.vector_top_k = 20
        self.graph_max_depth = 2
        self.min_confidence_threshold = 0.1
        
        # Boosting factors
        self.boost_factors = {
            'exact_sku_match': 10.0,
            'table_content': 2.0,
            'function_match': 2.0,
            'graph_relation': 5.0,
            'recent_document': 1.2
        }
        
        # Query type patterns
        self.query_patterns = {
            'compatibility': r'(compatible|work\s+with|bases?\s+for|heads?\s+for)',
            'specification': r'(spec|specification|feature|capability)',
            'installation': r'(install|mount|wire|connect)',
            'troubleshoot': r'(problem|issue|error|fault|trouble)'
        }
    
    def classify_query_type(self, query: str) -> str:
        """Classify query type for specialized handling"""
        query_lower = query.lower()
        
        for query_type, pattern in self.query_patterns.items():
            if re.search(pattern, query_lower):
                return query_type
        
        return 'general'
    
    def keyword_search(self, query: str, category_filter: Optional[str] = None) -> List[RetrievalResult]:
        """Keyword-based search with BM25-like scoring"""
        # Extract and normalize SKUs from query
        query_skus = self.sku_normalizer.extract_all_skus(query)
        
        results = []
        
        # Search in graph database for exact SKU matches
        for sku in query_skus:
            if sku in self.graph_db.graph.nodes():
                node_data = self.graph_db.graph.nodes[sku]
                
                # Get compatible components
                compatible_components = node_data.get('compatible_with', [])
                
                for comp_sku in compatible_components:
                    if comp_sku in self.graph_db.graph.nodes():
                        comp_data = self.graph_db.graph.nodes[comp_sku]
                        
                        # Determine function type from description
                        desc = comp_data.get('description', '').lower()
                        function_type = self._extract_function_type(desc)
                        
                        result = RetrievalResult(
                            content=comp_data.get('description', ''),
                            source_type='keyword',
                            sku_model=comp_sku,
                            function_type=function_type,
                            confidence=1.0,  # High confidence for exact matches
                            source_doc='graph_database',
                            metadata={'original_sku': sku, 'relation': 'compatible_with'}
                        )
                        results.append(result)
        
        return results
    
    def vector_search(self, query: str, category_filter: Optional[str] = None) -> List[RetrievalResult]:
        """Enhanced vector search using the new vector database"""
        # Search using enhanced vector database
        vector_results = self.vector_db.search(
            query=query,
            top_k=self.vector_top_k,
            category_filter=category_filter
        )
        
        results = []
        for chunk, score in vector_results:
            # Extract primary SKU from chunk
            primary_sku = chunk.product_models[0] if chunk.product_models else 'unknown'
            
            # Determine function type
            function_type = chunk.function_tags[0] if chunk.function_tags else 'unknown'
            
            result = RetrievalResult(
                content=chunk.content,
                source_type='vector',
                sku_model=primary_sku,
                function_type=function_type,
                confidence=score,
                source_doc=chunk.doc_id,
                metadata={
                    'chunk_id': chunk.chunk_id,
                    'is_table': chunk.is_table,
                    'product_models': chunk.product_models,
                    'compat_list': chunk.compat_list
                }
            )
            results.append(result)
        
        return results
    
    def graph_traversal(self, query: str) -> List[RetrievalResult]:
        """Graph traversal search with relationship following"""
        # Extract SKUs from query
        query_skus = self.sku_normalizer.extract_all_skus(query)
        
        results = []
        visited = set()
        
        for sku in query_skus:
            if sku not in self.graph_db.graph.nodes() or sku in visited:
                continue
            
            visited.add(sku)
            
            # Get direct relationships
            if hasattr(self.graph_db, 'graph_search'):
                related_skus = self.graph_db.graph_search([sku], max_depth=self.graph_max_depth)
            else:
                # Fallback to direct compatible_with lookup
                node_data = self.graph_db.graph.nodes[sku]
                related_skus = node_data.get('compatible_with', [])
            
            for related_sku in related_skus:
                if related_sku in self.graph_db.graph.nodes():
                    related_data = self.graph_db.graph.nodes[related_sku]
                    
                    # Calculate relationship confidence
                    confidence = 0.8  # High confidence for graph relationships
                    
                    # Determine function type
                    desc = related_data.get('description', '').lower()
                    function_type = self._extract_function_type(desc)
                    
                    result = RetrievalResult(
                        content=related_data.get('description', ''),
                        source_type='graph',
                        sku_model=related_sku,
                        function_type=function_type,
                        confidence=confidence,
                        source_doc='graph_traversal',
                        metadata={'source_sku': sku, 'relation_type': 'compatible_with'}
                    )
                    results.append(result)
        
        return results
    
    def _extract_function_type(self, description: str) -> str:
        """Extract function type from description text"""
        desc_lower = description.lower()
        
        # Function extraction patterns
        if 'sounder' in desc_lower or 'piezo' in desc_lower:
            return 'sounder'
        elif 'isolat' in desc_lower:
            return 'isolator'
        elif 'relay' in desc_lower:
            if 'supervised' in desc_lower:
                if '4-wire' in desc_lower or '4 wire' in desc_lower:
                    return 'relay_supervised_4wire'
                elif '2-wire' in desc_lower or '2 wire' in desc_lower:
                    return 'relay_supervised_2wire'
                else:
                    return 'relay_supervised'
            else:
                return 'relay'
        elif 'co' in desc_lower or 'carbon monoxide' in desc_lower:
            if 'sounder' in desc_lower:
                return 'co_sounder'
            else:
                return 'co_base'
        elif 'multi' in desc_lower and 'sensor' in desc_lower:
            if 'sounder' in desc_lower:
                return 'multi_sensor_sounder'
            else:
                return 'multi_sensor'
        elif 'standard' in desc_lower or 'white' in desc_lower:
            return 'standard'
        else:
            return 'unknown'
    
    def hybrid_search(self, query: str, category_filter: Optional[str] = None) -> List[RetrievalResult]:
        """Execute hybrid search combining all approaches"""
        print(f"🔍 Executing hybrid search for: '{query}'")
        
        # Classify query type for specialized handling
        query_type = self.classify_query_type(query)
        print(f"   Query type: {query_type}")
        
        # Execute parallel searches
        keyword_results = self.keyword_search(query, category_filter)
        vector_results = self.vector_search(query, category_filter)  
        graph_results = self.graph_traversal(query)
        
        print(f"   Keyword results: {len(keyword_results)}")
        print(f"   Vector results: {len(vector_results)}")
        print(f"   Graph results: {len(graph_results)}")
        
        # Combine and deduplicate results
        all_results = keyword_results + vector_results + graph_results
        
        # Group by SKU to avoid duplicates
        sku_groups = {}
        for result in all_results:
            sku = result.sku_model
            if sku not in sku_groups:
                sku_groups[sku] = []
            sku_groups[sku].append(result)
        
        # Select best result per SKU and apply fusion scoring
        fused_results = []
        for sku, results in sku_groups.items():
            if sku == 'unknown':
                # Keep all unknown SKU results
                fused_results.extend(results)
                continue
            
            # Select best result for this SKU and boost confidence
            best_result = max(results, key=lambda r: r.confidence)
            
            # Apply fusion boosting
            boost_factor = 1.0
            
            # Boost if multiple sources agree
            source_types = set(r.source_type for r in results)
            if len(source_types) > 1:
                boost_factor *= 1.5  # Multiple source agreement
            
            # Apply specific boosts
            if any(r.source_type == 'graph' for r in results):
                boost_factor *= self.boost_factors['graph_relation']
            
            if any(r.metadata.get('is_table', False) for r in results):
                boost_factor *= self.boost_factors['table_content']
            
            # Create fused result
            fused_result = RetrievalResult(
                content=best_result.content,
                source_type=f"hybrid_{'+'.join(source_types)}",
                sku_model=sku,
                function_type=best_result.function_type,
                confidence=min(best_result.confidence * boost_factor, 1.0),
                source_doc=best_result.source_doc,
                metadata={
                    'source_count': len(results),
                    'source_types': list(source_types),
                    'boost_factor': boost_factor
                }
            )
            fused_results.append(fused_result)
        
        # Sort by confidence
        fused_results.sort(key=lambda r: r.confidence, reverse=True)
        
        # Filter by minimum confidence threshold
        filtered_results = [r for r in fused_results if r.confidence >= self.min_confidence_threshold]
        
        print(f"   Final results: {len(filtered_results)}")
        return filtered_results
    
    def create_evidence_bundle(self, query: str, results: List[RetrievalResult]) -> EvidenceBundle:
        """Create structured evidence bundle from retrieval results"""
        # Extract primary SKU from query
        query_skus = self.sku_normalizer.extract_all_skus(query)
        primary_sku = query_skus[0] if query_skus else 'unknown'
        
        # Collect related SKUs
        related_skus = list(set(r.sku_model for r in results if r.sku_model != primary_sku))
        
        # Group by function type
        function_groups = {}
        for result in results:
            func = result.function_type
            if func not in function_groups:
                function_groups[func] = []
            function_groups[func].append(result.sku_model)
        
        # Create compatibility records
        compatibility_records = []
        for result in results:
            record = {
                'head_sku': primary_sku,
                'base_sku': result.sku_model,
                'function': result.function_type,
                'description': result.content[:100] + '...' if len(result.content) > 100 else result.content,
                'confidence': result.confidence,
                'source': result.source_type
            }
            compatibility_records.append(record)
        
        # Create citations
        citations = []
        source_docs = set(r.source_doc for r in results)
        for doc in source_docs:
            citation = {
                'document': doc,
                'results_count': len([r for r in results if r.source_doc == doc]),
                'confidence': max(r.confidence for r in results if r.source_doc == doc)
            }
            citations.append(citation)
        
        # Generate reasoning
        reasoning = [
            f"Found {len(results)} compatibility matches for {primary_sku}",
            f"Identified {len(function_groups)} function categories: {', '.join(function_groups.keys())}",
            f"Evidence from {len(citations)} source documents",
            f"Average confidence: {np.mean([r.confidence for r in results]):.2f}"
        ]
        
        # Calculate overall confidence
        overall_confidence = np.mean([r.confidence for r in results]) if results else 0.0
        
        return EvidenceBundle(
            query=query,
            primary_sku=primary_sku,
            related_skus=related_skus,
            compatibility_records=compatibility_records,
            function_groups=function_groups,
            citations=citations,
            confidence=overall_confidence,
            reasoning=reasoning
        )
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval system statistics"""
        return {
            "vector_db_chunks": len(self.vector_db.chunks),
            "graph_nodes": len(self.graph_db.graph.nodes()),
            "graph_edges": len(self.graph_db.graph.edges()),
            "sku_synonyms": len(self.sku_normalizer.synonym_map),
            "boost_factors": self.boost_factors,
            "min_confidence": self.min_confidence_threshold
        }