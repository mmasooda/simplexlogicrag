#!/usr/bin/env python3
"""
Fusion Logic and Structured Answer Generation based on ChatGPT-5 recommendations
Combines multiple retrieval results with confidence scoring and generates structured answers
"""

import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np

from hybrid_retrieval import RetrievalResult, EvidenceBundle

@dataclass
class AnswerSection:
    """Structured answer section"""
    title: str
    content: List[str]
    citations: List[str]
    confidence: float

@dataclass
class StructuredAnswer:
    """Complete structured answer following ChatGPT-5 template"""
    query: str
    primary_sku: str
    answer_sections: List[AnswerSection]
    evidence_bundle: Dict[str, Any]
    overall_confidence: float
    reasoning_chain: List[str]
    panel_caveats: List[str]
    citations: List[str]

class FusionEngine:
    """Fusion engine for combining retrieval results with guardrails"""
    
    def __init__(self, min_corroborating_sources: int = 2):
        self.min_corroborating_sources = min_corroborating_sources
        
        # Confidence thresholds
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.5, 
            'low': 0.2
        }
        
        # Source reliability weights
        self.source_weights = {
            'graph': 1.0,           # Highest reliability
            'table_content': 0.9,   # High reliability for table data
            'vector': 0.7,          # Medium reliability
            'keyword': 0.6,         # Lower reliability
            'inference': 0.4        # Lowest reliability
        }
        
        # Function type display names
        self.function_display_names = {
            'standard': 'Standard Base',
            'relay': 'Relay Base', 
            'relay_supervised': 'Supervised Relay Base',
            'relay_supervised_4wire': '4-Wire Supervised Relay Base',
            'relay_supervised_2wire': '2-Wire Supervised Relay Base',
            'sounder': 'Sounder Base',
            'isolator': 'Isolator Base',
            'co_base': 'CO Sensor Base',
            'co_sounder': 'CO Sounder Base',
            'multi_sensor': 'Multi-Sensor Base',
            'multi_sensor_sounder': 'Multi-Sensor Sounder Base'
        }
    
    def validate_result_consistency(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Validate consistency across multiple retrieval results with guardrails"""
        validated_results = []
        
        # Group results by SKU for consistency checking
        sku_groups = defaultdict(list)
        for result in results:
            sku_groups[result.sku_model].append(result)
        
        for sku, sku_results in sku_groups.items():
            if sku == 'unknown':
                continue
                
            # Check for consistency in function type
            function_types = [r.function_type for r in sku_results if r.function_type != 'unknown']
            
            if len(set(function_types)) == 1:
                # Consistent function type - boost confidence
                for result in sku_results:
                    result.confidence *= 1.2
            elif len(function_types) > 1:
                # Inconsistent function types - check source reliability
                weighted_functions = defaultdict(float)
                for result in sku_results:
                    if result.function_type != 'unknown':
                        weight = self.source_weights.get(result.source_type.split('_')[0], 0.5)
                        weighted_functions[result.function_type] += weight * result.confidence
                
                # Use most reliable function type
                best_function = max(weighted_functions.items(), key=lambda x: x[1])[0]
                for result in sku_results:
                    if result.function_type != best_function:
                        result.confidence *= 0.8  # Reduce confidence for inconsistent results
            
            # Require minimum corroborating sources for high confidence claims
            if len(sku_results) >= self.min_corroborating_sources:
                for result in sku_results:
                    result.confidence *= 1.3  # Boost for multiple sources
            else:
                # Single source - require high base confidence
                for result in sku_results:
                    if result.confidence < 0.7:
                        result.confidence *= 0.7  # Reduce confidence for single low-confidence source
            
            validated_results.extend(sku_results)
        
        return validated_results
    
    def resolve_conflicts(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Resolve conflicts between different sources using ChatGPT-5 guidelines"""
        resolved_results = []
        
        # Group by SKU to identify conflicts
        sku_groups = defaultdict(list)
        for result in results:
            sku_groups[result.sku_model].append(result)
        
        for sku, sku_results in sku_groups.items():
            if len(sku_results) == 1:
                resolved_results.extend(sku_results)
                continue
            
            # Check for conflicts in key attributes
            source_types = set(r.source_type for r in sku_results)
            
            if 'graph' in source_types:
                # Prefer graph data as most reliable
                graph_results = [r for r in sku_results if r.source_type == 'graph']
                resolved_results.extend(graph_results)
                
                # Keep non-graph results if they have table content
                other_results = [r for r in sku_results if r.source_type != 'graph']
                table_results = [r for r in other_results if r.metadata.get('is_table', False)]
                
                for table_result in table_results:
                    table_result.confidence *= 0.8  # Reduce confidence but keep
                    resolved_results.append(table_result)
            
            else:
                # No graph data - prefer table content over general content
                table_results = [r for r in sku_results if r.metadata.get('is_table', False)]
                if table_results:
                    resolved_results.extend(table_results)
                else:
                    # Keep highest confidence result
                    best_result = max(sku_results, key=lambda r: r.confidence)
                    resolved_results.append(best_result)
        
        return resolved_results
    
    def apply_panel_caveats(self, results: List[RetrievalResult]) -> Tuple[List[RetrievalResult], List[str]]:
        """Extract and apply panel compatibility caveats"""
        caveats = []
        
        # Extract caveats from result metadata
        for result in results:
            result_caveats = result.metadata.get('panel_caveats', [])
            if isinstance(result_caveats, list):
                caveats.extend(result_caveats)
            
            # Check content for caveat patterns
            content = result.content.lower()
            if 'not compatible' in content or 'except' in content:
                if '2120 cdt' in content:
                    caveats.append("Not compatible with 2120 CDT panels")
                if '4007' in content:
                    caveats.append("May require special configuration with 4007 series panels")
        
        # Remove duplicates
        unique_caveats = list(set(caveats))
        
        # Apply caveats as confidence adjustments
        adjusted_results = []
        for result in results:
            # Check if this result is affected by any caveats
            affected_by_caveat = False
            for caveat in unique_caveats:
                if 'relay' in result.function_type and 'not compatible' in caveat.lower():
                    affected_by_caveat = True
                    break
            
            if affected_by_caveat:
                result.confidence *= 0.9  # Slight reduction for caveat-affected results
            
            adjusted_results.append(result)
        
        return adjusted_results, unique_caveats
    
    def fuse_results(self, results: List[RetrievalResult]) -> Tuple[List[RetrievalResult], float]:
        """Complete fusion pipeline following ChatGPT-5 guidelines"""
        print(f"🔄 Fusing {len(results)} retrieval results")
        
        # Step 1: Validate consistency
        validated_results = self.validate_result_consistency(results)
        print(f"   After validation: {len(validated_results)} results")
        
        # Step 2: Resolve conflicts
        resolved_results = self.resolve_conflicts(validated_results)
        print(f"   After conflict resolution: {len(resolved_results)} results")
        
        # Step 3: Apply panel caveats
        final_results, panel_caveats = self.apply_panel_caveats(resolved_results)
        print(f"   After caveat application: {len(final_results)} results, {len(panel_caveats)} caveats")
        
        # Step 4: Calculate overall confidence
        if final_results:
            overall_confidence = np.mean([r.confidence for r in final_results])
        else:
            overall_confidence = 0.0
        
        # Step 5: Filter by minimum confidence
        high_confidence_results = [r for r in final_results if r.confidence >= self.confidence_thresholds['low']]
        
        print(f"   Final high-confidence results: {len(high_confidence_results)}")
        
        return high_confidence_results, overall_confidence

class AnswerGenerator:
    """Generates structured answers following ChatGPT-5 template"""
    
    def __init__(self, fusion_engine: FusionEngine):
        self.fusion_engine = fusion_engine
        
    def group_by_function(self, results: List[RetrievalResult]) -> Dict[str, List[str]]:
        """Group SKUs by function type for organized presentation"""
        function_groups = defaultdict(list)
        
        for result in results:
            function = result.function_type
            if result.sku_model not in function_groups[function]:
                function_groups[function].append(result.sku_model)
        
        # Sort each group for consistent output
        for function in function_groups:
            function_groups[function].sort()
        
        return dict(function_groups)
    
    def create_compatibility_sections(self, function_groups: Dict[str, List[str]], 
                                    results: List[RetrievalResult]) -> List[AnswerSection]:
        """Create organized sections for compatibility answer"""
        sections = []
        
        # Create result lookup for descriptions
        result_lookup = {r.sku_model: r for r in results}
        
        for function_type, skus in function_groups.items():
            if not skus:
                continue
            
            # Get display name for function
            display_name = self.fusion_engine.function_display_names.get(function_type, function_type.title())
            
            # Create content for this function group
            content = []
            citations = []
            confidences = []
            
            for sku in skus:
                if sku in result_lookup:
                    result = result_lookup[sku]
                    # Extract short description
                    desc = result.content
                    if len(desc) > 100:
                        desc = desc[:100] + "..."
                    
                    content.append(f"{sku} — {desc}")
                    citations.append(result.source_doc)
                    confidences.append(result.confidence)
                else:
                    content.append(f"{sku}")
            
            # Calculate section confidence
            section_confidence = np.mean(confidences) if confidences else 0.5
            
            section = AnswerSection(
                title=f"{display_name}:",
                content=content,
                citations=list(set(citations)),
                confidence=section_confidence
            )
            sections.append(section)
        
        # Sort sections by confidence (highest first)
        sections.sort(key=lambda s: s.confidence, reverse=True)
        
        return sections
    
    def generate_structured_answer(self, evidence_bundle: EvidenceBundle,
                                 fused_results: List[RetrievalResult],
                                 overall_confidence: float) -> StructuredAnswer:
        """Generate complete structured answer following ChatGPT-5 format"""
        
        # Group results by function for organized presentation
        function_groups = self.group_by_function(fused_results)
        
        # Create answer sections
        answer_sections = self.create_compatibility_sections(function_groups, fused_results)
        
        # Extract panel caveats
        panel_caveats = []
        for result in fused_results:
            result_caveats = result.metadata.get('panel_caveats', [])
            if isinstance(result_caveats, list):
                panel_caveats.extend(result_caveats)
        panel_caveats = list(set(panel_caveats))  # Remove duplicates
        
        # Create reasoning chain
        reasoning_chain = [
            f"Analyzed query for {evidence_bundle.primary_sku}",
            f"Found {len(fused_results)} compatible bases across {len(function_groups)} function categories",
            f"Evidence sourced from {len(set(r.source_doc for r in fused_results))} documents",
            f"Overall confidence: {overall_confidence:.1%}"
        ]
        
        # Create citations
        citations = list(set(r.source_doc for r in fused_results))
        
        # Convert evidence bundle to dictionary
        evidence_dict = {
            'primary_sku': evidence_bundle.primary_sku,
            'related_skus': evidence_bundle.related_skus,
            'function_groups': evidence_bundle.function_groups,
            'compatibility_records': evidence_bundle.compatibility_records,
            'confidence': evidence_bundle.confidence
        }
        
        return StructuredAnswer(
            query=evidence_bundle.query,
            primary_sku=evidence_bundle.primary_sku,
            answer_sections=answer_sections,
            evidence_bundle=evidence_dict,
            overall_confidence=overall_confidence,
            reasoning_chain=reasoning_chain,
            panel_caveats=panel_caveats,
            citations=citations
        )
    
    def format_human_readable(self, structured_answer: StructuredAnswer) -> str:
        """Format structured answer as human-readable text"""
        lines = []
        
        # Header
        primary_sku = structured_answer.primary_sku
        lines.append(f"Compatible bases for Simplex {primary_sku}:")
        lines.append("")
        
        # Function sections
        for section in structured_answer.answer_sections:
            lines.append(f"**{section.title}**")
            for item in section.content:
                lines.append(f"- {item}")
            lines.append("")
        
        # Panel caveats if any
        if structured_answer.panel_caveats:
            lines.append("**Important Notes:**")
            for caveat in structured_answer.panel_caveats:
                lines.append(f"⚠️ {caveat}")
            lines.append("")
        
        # Confidence indicator
        confidence_pct = structured_answer.overall_confidence * 100
        confidence_level = "High" if confidence_pct >= 80 else "Medium" if confidence_pct >= 50 else "Low"
        lines.append(f"*Answer confidence: {confidence_level} ({confidence_pct:.0f}%)*")
        
        return "\n".join(lines)
    
    def format_json_evidence(self, structured_answer: StructuredAnswer) -> str:
        """Format structured answer as JSON evidence bundle"""
        return json.dumps(asdict(structured_answer), indent=2, ensure_ascii=False)

class IntegratedSystem:
    """Complete integrated system combining fusion and answer generation"""
    
    def __init__(self, graph_db, vector_db, sku_normalizer, hybrid_retrieval):
        self.graph_db = graph_db
        self.vector_db = vector_db
        self.sku_normalizer = sku_normalizer
        self.hybrid_retrieval = hybrid_retrieval
        
        self.fusion_engine = FusionEngine()
        self.answer_generator = AnswerGenerator(self.fusion_engine)
    
    def process_query(self, query: str, return_evidence: bool = False) -> Tuple[str, Optional[StructuredAnswer]]:
        """Complete query processing pipeline"""
        print(f"🚀 Processing query: '{query}'")
        
        # Step 1: Hybrid retrieval
        retrieval_results = self.hybrid_retrieval.hybrid_search(query)
        
        # Step 2: Create evidence bundle
        evidence_bundle = self.hybrid_retrieval.create_evidence_bundle(query, retrieval_results)
        
        # Step 3: Fusion with guardrails
        fused_results, overall_confidence = self.fusion_engine.fuse_results(retrieval_results)
        
        # Step 4: Generate structured answer
        structured_answer = self.answer_generator.generate_structured_answer(
            evidence_bundle, fused_results, overall_confidence
        )
        
        # Step 5: Format human-readable answer
        human_answer = self.answer_generator.format_human_readable(structured_answer)
        
        print(f"✅ Generated answer with {overall_confidence:.1%} confidence")
        
        if return_evidence:
            return human_answer, structured_answer
        else:
            return human_answer, None