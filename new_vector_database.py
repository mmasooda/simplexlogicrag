#!/usr/bin/env python3
"""
New Vector Database Implementation based on ChatGPT-5 recommendations
- Proper metadata fields for fire alarm components
- Table-aware chunking with preference for compatibility tables
- Function-based indexing and retrieval
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import re

@dataclass
class DocumentChunk:
    """Enhanced document chunk with fire alarm specific metadata"""
    chunk_id: str
    content: str
    doc_id: str
    title: str
    page_from: int
    page_to: int
    product_models: List[str]  # SKUs mentioned in chunk
    product_category: str      # Head/Base/Panel/Module
    compat_list: List[str]     # Models found in compatibility tables
    function_tags: List[str]   # sounder, isolator, relay, co_base, multi_sensor
    protocols: List[str]       # IDNet, MAPNET II
    is_table: bool            # Prefer table chunks
    is_bullet: bool           # Bullet point format
    confidence_source: float  # 0.0-1.0 based on source quality

class EnhancedVectorDatabase:
    """Enhanced vector database following ChatGPT-5 recommendations"""
    
    def __init__(self, db_path: str = "enhanced_vector_db.pkl"):
        self.db_path = db_path
        self.chunks: List[DocumentChunk] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.vectors: Optional[np.ndarray] = None
        
        # Fire alarm specific configurations
        self.max_chars = 1200
        self.overlap = 150
        
        # Enhanced stop words for fire alarm domain
        self.custom_stop_words = [
            'simplex', 'fire', 'alarm', 'system', 'device', 'unit', 
            'model', 'part', 'number', 'see', 'page', 'refer', 'contact'
        ]
        
        # SKU patterns for better extraction
        self.sku_patterns = [
            r'\b\d{4}-\d{4}\b',    # 4098-9714 format
            r'\b\d{4}\s+\d{4}\b',  # 4098 9714 format  
            r'\b\d{4}/\d{4}\b',    # 4098/9714 format
        ]
        
        self._load_database()
    
    def _load_database(self):
        """Load existing database or initialize empty"""
        if Path(self.db_path).exists():
            try:
                with open(self.db_path, 'rb') as f:
                    data = pickle.load(f)
                    self.chunks = data.get('chunks', [])
                    self.vectorizer = data.get('vectorizer')
                    self.vectors = data.get('vectors')
                print(f"✅ Loaded {len(self.chunks)} chunks from database")
            except Exception as e:
                print(f"⚠️ Failed to load database: {e}. Starting fresh.")
                self._initialize_empty()
        else:
            self._initialize_empty()
    
    def _initialize_empty(self):
        """Initialize empty database"""
        self.chunks = []
        self.vectorizer = None
        self.vectors = None
    
    def extract_sku_numbers(self, text: str) -> List[str]:
        """Extract SKU/part numbers using enhanced patterns"""
        skus = []
        for pattern in self.sku_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            skus.extend(matches)
        
        # Normalize SKUs to standard format (XXXX-XXXX)
        normalized_skus = []
        for sku in skus:
            # Convert various formats to standard XXXX-XXXX
            clean_sku = re.sub(r'[\s/]', '-', sku)
            if re.match(r'\d{4}-\d{4}', clean_sku):
                normalized_skus.append(clean_sku)
        
        return list(set(normalized_skus))  # Remove duplicates
    
    def extract_function_tags(self, text: str) -> List[str]:
        """Extract fire alarm function tags from text"""
        functions = []
        function_patterns = {
            'sounder': r'\b(sounder|audible|piezo|alarm)\b',
            'isolator': r'\b(isolat\w*|isolation|circuit\s+protection)\b',
            'relay': r'\b(relay|supervised|unsupervised)\b',
            'co_base': r'\b(carbon\s+monoxide|co\s+sensor|co\s+base)\b',
            'multi_sensor': r'\b(multi[\-\s]?sensor|combination|photo[\-\s]?heat)\b',
            'standard': r'\b(standard|basic|conventional)\b',
            'remote_led': r'\b(remote\s+led|led\s+indicator)\b'
        }
        
        text_lower = text.lower()
        for func, pattern in function_patterns.items():
            if re.search(pattern, text_lower):
                functions.append(func)
        
        return functions
    
    def detect_table_content(self, text: str) -> Tuple[bool, float]:
        """Detect if content is from a table and confidence level"""
        table_indicators = [
            r'\|\s*\w+\s*\|',  # Pipe-separated columns
            r'\t\w+\t',        # Tab-separated columns
            r'Compatible\s+with',
            r'Base\s+Type',
            r'Function',
            r'Model\s+Number',
            r'Part\s+Number'
        ]
        
        is_table = False
        confidence = 0.0
        
        for pattern in table_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                is_table = True
                confidence += 0.2
        
        # Higher confidence for structured content
        if '\t' in text or '|' in text:
            confidence += 0.3
        
        return is_table, min(confidence, 1.0)
    
    def chunk_document(self, content: str, doc_id: str, title: str, page: int = 1) -> List[DocumentChunk]:
        """Chunk document with enhanced metadata extraction"""
        chunks = []
        
        # Split into overlapping chunks
        start = 0
        chunk_id = 0
        
        while start < len(content):
            end = start + self.max_chars
            
            # Try to break at sentence boundary
            if end < len(content):
                last_period = content.rfind('.', start, end)
                last_newline = content.rfind('\n', start, end)
                boundary = max(last_period, last_newline)
                if boundary > start:
                    end = boundary + 1
            
            chunk_text = content[start:end].strip()
            if not chunk_text:
                break
            
            # Extract metadata
            product_models = self.extract_sku_numbers(chunk_text)
            function_tags = self.extract_function_tags(chunk_text)
            is_table, table_confidence = self.detect_table_content(chunk_text)
            
            # Determine product category
            category = "unknown"
            if any(word in chunk_text.lower() for word in ['head', 'detector', 'sensor']):
                category = "Head"
            elif any(word in chunk_text.lower() for word in ['base', 'mounting']):
                category = "Base"
            elif any(word in chunk_text.lower() for word in ['panel', 'control']):
                category = "Panel"
            elif any(word in chunk_text.lower() for word in ['module', 'card']):
                category = "Module"
            
            # Extract compatibility lists (simple pattern matching)
            compat_list = []
            compat_patterns = [
                r'compatible\s+with[:\s]+([0-9\-\s,]+)',
                r'works\s+with[:\s]+([0-9\-\s,]+)',
                r'use\s+with[:\s]+([0-9\-\s,]+)'
            ]
            
            for pattern in compat_patterns:
                matches = re.findall(pattern, chunk_text, re.IGNORECASE)
                for match in matches:
                    potential_skus = self.extract_sku_numbers(match)
                    compat_list.extend(potential_skus)
            
            # Create chunk
            chunk = DocumentChunk(
                chunk_id=f"{doc_id}_{chunk_id}",
                content=chunk_text,
                doc_id=doc_id,
                title=title,
                page_from=page,
                page_to=page,
                product_models=product_models,
                product_category=category,
                compat_list=list(set(compat_list)),
                function_tags=function_tags,
                protocols=["IDNet"] if "idnet" in chunk_text.lower() else [],
                is_table=is_table,
                is_bullet='•' in chunk_text or '-' in chunk_text[:50],
                confidence_source=table_confidence if is_table else 0.5
            )
            
            chunks.append(chunk)
            chunk_id += 1
            
            # Move start position with overlap
            start = end - self.overlap
            if start >= len(content):
                break
        
        return chunks
    
    def add_document(self, content: str, doc_id: str, title: str, page: int = 1):
        """Add document with enhanced chunking"""
        new_chunks = self.chunk_document(content, doc_id, title, page)
        self.chunks.extend(new_chunks)
        print(f"✅ Added {len(new_chunks)} chunks from {doc_id}")
    
    def build_vectors(self):
        """Build TF-IDF vectors with fire alarm optimizations"""
        if not self.chunks:
            print("⚠️ No chunks to vectorize")
            return
        
        # Prepare texts with enhanced features
        texts = []
        for chunk in self.chunks:
            # Enhance text with metadata for better matching
            enhanced_text = chunk.content
            
            # Boost important terms
            if chunk.product_models:
                enhanced_text += " " + " ".join(chunk.product_models) * 2  # Boost SKUs
            
            if chunk.function_tags:
                enhanced_text += " " + " ".join(chunk.function_tags) * 2  # Boost functions
            
            if chunk.compat_list:
                enhanced_text += " compatible " + " ".join(chunk.compat_list)  # Boost compatibility
            
            texts.append(enhanced_text)
        
        # Configure TF-IDF with fire alarm optimizations
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words=list(set(self.custom_stop_words)),  # Custom stop words
            ngram_range=(1, 2),  # Include bigrams for better matching
            min_df=2,  # Must appear in at least 2 documents
            max_df=0.8,  # Ignore terms in >80% of docs
            lowercase=True,
            token_pattern=r'(?u)\b\w+\b|\b\d{4}-\d{4}\b'  # Include SKU patterns
        )
        
        print("🔄 Building TF-IDF vectors...")
        self.vectors = self.vectorizer.fit_transform(texts)
        print(f"✅ Built vectors: {self.vectors.shape}")
    
    def search(self, query: str, top_k: int = 10, category_filter: Optional[str] = None) -> List[Tuple[DocumentChunk, float]]:
        """Enhanced search with fire alarm specific boosting"""
        if not self.vectorizer or self.vectors is None:
            print("⚠️ Vectors not built. Call build_vectors() first.")
            return []
        
        # Extract query features
        query_skus = self.extract_sku_numbers(query)
        query_functions = self.extract_function_tags(query)
        
        # Enhanced query with boosted terms
        enhanced_query = query
        if query_skus:
            enhanced_query += " " + " ".join(query_skus) * 3  # Heavy boost for SKUs
        if query_functions:
            enhanced_query += " " + " ".join(query_functions) * 2  # Boost functions
        
        # Vectorize query
        query_vector = self.vectorizer.transform([enhanced_query])
        
        # Compute similarities
        similarities = cosine_similarity(query_vector, self.vectors).flatten()
        
        # Apply boosting factors
        boosted_scores = []
        for i, (chunk, sim) in enumerate(zip(self.chunks, similarities)):
            score = sim
            
            # Boost exact SKU matches
            if query_skus:
                for sku in query_skus:
                    if sku in chunk.product_models or sku in chunk.compat_list:
                        score *= 10.0  # Heavy boost for exact SKU match
            
            # Boost table content
            if chunk.is_table:
                score *= 2.0
            
            # Boost high confidence sources
            score *= (1.0 + chunk.confidence_source)
            
            # Boost function matches
            if query_functions:
                common_functions = set(query_functions) & set(chunk.function_tags)
                if common_functions:
                    score *= (1.0 + len(common_functions) * 0.5)
            
            # Apply category filter
            if category_filter and chunk.product_category.lower() != category_filter.lower():
                score *= 0.1  # Heavy penalty for wrong category
            
            boosted_scores.append((chunk, score))
        
        # Sort by boosted score
        boosted_scores.sort(key=lambda x: x[1], reverse=True)
        
        return boosted_scores[:top_k]
    
    def save_database(self):
        """Save database to disk"""
        data = {
            'chunks': self.chunks,
            'vectorizer': self.vectorizer,
            'vectors': self.vectors
        }
        
        with open(self.db_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✅ Saved database with {len(self.chunks)} chunks to {self.db_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        if not self.chunks:
            return {"chunks": 0}
        
        categories = {}
        functions = {}
        table_count = 0
        
        for chunk in self.chunks:
            # Count categories
            cat = chunk.product_category
            categories[cat] = categories.get(cat, 0) + 1
            
            # Count functions
            for func in chunk.function_tags:
                functions[func] = functions.get(func, 0) + 1
            
            # Count tables
            if chunk.is_table:
                table_count += 1
        
        return {
            "total_chunks": len(self.chunks),
            "table_chunks": table_count,
            "categories": categories,
            "functions": functions,
            "vector_dimensions": self.vectors.shape[1] if self.vectors is not None else 0
        }