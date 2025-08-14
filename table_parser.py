#!/usr/bin/env python3
"""
Table Parser for SimplexRAG based on ChatGPT-5 recommendations
Extracts compatibility tables from datasheets with structured output
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import csv
import json

@dataclass
class CompatibilityRecord:
    """Structured compatibility record from table parsing"""
    head_model: str
    base_model: str
    function: str           # standard, relay, sounder, isolator, co_base
    description: str
    panel_caveats: List[str] # ["not 2120 CDT"]
    source_doc: str
    source_page: int
    confidence: float       # 0.0-1.0

@dataclass 
class TableStructure:
    """Detected table structure"""
    headers: List[str]
    rows: List[List[str]]
    table_type: str         # compatibility, specification, feature
    confidence: float

class TableParser:
    """Parse compatibility tables from fire alarm datasheets"""
    
    def __init__(self):
        # Table detection patterns
        self.table_patterns = {
            'pipe_separated': r'\|([^|\n]+)\|',
            'tab_separated': r'([^\t\n]+)\t([^\t\n]+)',
            'space_aligned': r'^(\S+(?:\s+\S+)*?)\s{3,}(\S+(?:\s+\S+)*)$',
        }
        
        # Header patterns for compatibility tables
        self.compatibility_headers = [
            r'head|detector|sensor',
            r'base|mount|mounting',
            r'function|type|feature', 
            r'compatible|compatibility',
            r'model|part\s*number|p/n',
            r'description',
            r'notes|caveats|restrictions'
        ]
        
        # Function categorization patterns
        self.function_patterns = {
            'standard': r'\b(standard|basic|conventional|white)\b',
            'relay': r'\b(relay|remote\s*led|unsupervised)\b',
            'relay_supervised': r'\b(supervised\s*relay|4[\-\s]?wire|2[\-\s]?wire)\b',
            'sounder': r'\b(sounder|audible|piezo|alarm)\b',
            'isolator': r'\b(isolat\w*|isolation|circuit\s*protection)\b',
            'co_base': r'\b(carbon\s*monoxide|co\s*sensor|co\s*base)\b',
            'multi_sensor': r'\b(multi[\-\s]?sensor|combination)\b'
        }
        
        # Panel caveat patterns
        self.caveat_patterns = [
            r'not\s+compatible\s+with\s+([0-9A-Z\-\s]+)',
            r'except\s+([0-9A-Z\-\s]+)',
            r'excluding\s+([0-9A-Z\-\s]+)',
            r'not\s+supported\s+by\s+([0-9A-Z\-\s]+)'
        ]
        
        # SKU extraction patterns
        self.sku_pattern = r'\b\d{4}[-\s]?\d{4}\b'
        
    def detect_table_structure(self, text: str) -> Optional[TableStructure]:
        """Detect and parse table structure from text"""
        lines = text.strip().split('\n')
        
        # Try different table formats
        for format_name, pattern in self.table_patterns.items():
            table_data = self._parse_table_format(lines, pattern, format_name)
            if table_data:
                return table_data
        
        return None
    
    def _parse_table_format(self, lines: List[str], pattern: str, format_type: str) -> Optional[TableStructure]:
        """Parse table using specific format pattern"""
        parsed_rows = []
        potential_headers = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if format_type == 'pipe_separated':
                # Extract cells between pipes
                matches = re.findall(pattern, line)
                if matches and len(matches) >= 2:
                    parsed_rows.append([cell.strip() for cell in matches])
            
            elif format_type == 'tab_separated':
                # Split by tabs
                if '\t' in line:
                    cells = [cell.strip() for cell in line.split('\t') if cell.strip()]
                    if len(cells) >= 2:
                        parsed_rows.append(cells)
            
            elif format_type == 'space_aligned':
                # Detect space-aligned columns
                if re.match(pattern, line, re.MULTILINE):
                    # Split by multiple spaces (3 or more)
                    cells = re.split(r'\s{3,}', line)
                    if len(cells) >= 2:
                        parsed_rows.append([cell.strip() for cell in cells])
        
        if len(parsed_rows) < 2:  # Need at least header + 1 data row
            return None
        
        # Assume first row is headers
        headers = parsed_rows[0]
        rows = parsed_rows[1:]
        
        # Determine table type based on headers
        table_type = self._classify_table_type(headers)
        
        # Calculate confidence based on structure quality
        confidence = self._calculate_structure_confidence(headers, rows, format_type)
        
        return TableStructure(
            headers=headers,
            rows=rows,
            table_type=table_type,
            confidence=confidence
        )
    
    def _classify_table_type(self, headers: List[str]) -> str:
        """Classify table type based on headers"""
        header_text = ' '.join(headers).lower()
        
        # Check for compatibility table indicators
        compat_indicators = ['compatible', 'base', 'head', 'detector', 'function']
        compat_score = sum(1 for indicator in compat_indicators if indicator in header_text)
        
        if compat_score >= 2:
            return 'compatibility'
        elif 'specification' in header_text or 'spec' in header_text:
            return 'specification'
        elif 'feature' in header_text or 'function' in header_text:
            return 'feature'
        else:
            return 'unknown'
    
    def _calculate_structure_confidence(self, headers: List[str], rows: List[List[str]], format_type: str) -> float:
        """Calculate confidence in table structure detection"""
        confidence = 0.0
        
        # Base confidence by format type
        format_confidence = {
            'pipe_separated': 0.9,
            'tab_separated': 0.8, 
            'space_aligned': 0.6
        }
        confidence += format_confidence.get(format_type, 0.5)
        
        # Bonus for consistent column count
        if rows:
            col_counts = [len(row) for row in rows]
            if len(set(col_counts)) == 1:  # All rows have same column count
                confidence += 0.2
        
        # Bonus for compatibility table headers
        header_text = ' '.join(headers).lower()
        for pattern in self.compatibility_headers:
            if re.search(pattern, header_text):
                confidence += 0.1
        
        return min(confidence, 1.0)
    
    def extract_compatibility_records(self, table: TableStructure, doc_id: str, page: int) -> List[CompatibilityRecord]:
        """Extract structured compatibility records from parsed table"""
        if table.table_type != 'compatibility':
            return []
        
        records = []
        
        # Map headers to semantic meaning
        header_map = self._map_headers_to_semantics(table.headers)
        
        for row in table.rows:
            if len(row) < 2:  # Need at least 2 columns
                continue
            
            # Extract components based on header mapping
            head_model = self._extract_from_row(row, header_map, 'head')
            base_model = self._extract_from_row(row, header_map, 'base')
            function = self._extract_from_row(row, header_map, 'function')
            description = self._extract_from_row(row, header_map, 'description')
            notes = self._extract_from_row(row, header_map, 'notes')
            
            # Normalize SKUs
            head_skus = re.findall(self.sku_pattern, head_model or '')
            base_skus = re.findall(self.sku_pattern, base_model or '')
            
            if not head_skus and not base_skus:
                continue  # No valid SKUs found
            
            # Classify function if not explicitly provided
            if not function and description:
                function = self._classify_function(description)
            
            # Extract panel caveats from notes
            panel_caveats = self._extract_panel_caveats(notes or '')
            
            # Create records for each head-base combination
            for head in (head_skus or ['Unknown']):
                for base in (base_skus or ['Unknown']):
                    record = CompatibilityRecord(
                        head_model=self._normalize_sku(head),
                        base_model=self._normalize_sku(base),
                        function=function or 'unknown',
                        description=description or '',
                        panel_caveats=panel_caveats,
                        source_doc=doc_id,
                        source_page=page,
                        confidence=table.confidence
                    )
                    records.append(record)
        
        return records
    
    def _map_headers_to_semantics(self, headers: List[str]) -> Dict[str, int]:
        """Map table headers to semantic categories"""
        header_map = {}
        
        for i, header in enumerate(headers):
            header_lower = header.lower()
            
            # Map to categories
            if re.search(r'head|detector|sensor', header_lower):
                header_map['head'] = i
            elif re.search(r'base|mount', header_lower):
                header_map['base'] = i
            elif re.search(r'function|type|feature', header_lower):
                header_map['function'] = i
            elif re.search(r'model|part.*number|p/n', header_lower):
                # Could be either head or base - use context
                if 'head' not in header_map:
                    header_map['head'] = i
                elif 'base' not in header_map:
                    header_map['base'] = i
            elif re.search(r'description|desc', header_lower):
                header_map['description'] = i
            elif re.search(r'notes|caveats|restrictions', header_lower):
                header_map['notes'] = i
        
        return header_map
    
    def _extract_from_row(self, row: List[str], header_map: Dict[str, int], semantic: str) -> Optional[str]:
        """Extract value from row based on semantic header mapping"""
        if semantic in header_map:
            col_index = header_map[semantic]
            if col_index < len(row):
                return row[col_index].strip()
        return None
    
    def _classify_function(self, text: str) -> str:
        """Classify base function from description text"""
        text_lower = text.lower()
        
        for function, pattern in self.function_patterns.items():
            if re.search(pattern, text_lower):
                return function
        
        return 'standard'  # Default to standard if no pattern matches
    
    def _extract_panel_caveats(self, notes: str) -> List[str]:
        """Extract panel compatibility caveats from notes"""
        caveats = []
        
        if not notes:
            return caveats
        
        for pattern in self.caveat_patterns:
            matches = re.findall(pattern, notes, re.IGNORECASE)
            for match in matches:
                caveat = match.strip()
                if caveat:
                    caveats.append(caveat)
        
        return caveats
    
    def _normalize_sku(self, sku: str) -> str:
        """Normalize SKU to standard format"""
        if not sku or sku == 'Unknown':
            return sku
        
        # Extract numbers and format as XXXX-XXXX
        match = re.search(r'(\d{4})[-\s]?(\d{4})', sku)
        if match:
            return f"{match.group(1)}-{match.group(2)}"
        
        return sku
    
    def parse_document_tables(self, content: str, doc_id: str) -> List[CompatibilityRecord]:
        """Parse all tables in a document and extract compatibility records"""
        records = []
        
        # Split document into potential table sections
        # Look for table indicators and parse each section
        sections = re.split(r'\n\s*\n', content)  # Split by double newlines
        
        page = 1  # Simple page tracking
        for i, section in enumerate(sections):
            # Skip very short sections
            if len(section.strip()) < 100:
                continue
            
            # Check if section contains table-like structure
            table_indicators = ['|', '\t', 'Compatible', 'Base Type', 'Function']
            has_table_indicator = any(indicator in section for indicator in table_indicators)
            
            if not has_table_indicator:
                continue
            
            # Try to parse as table
            table_structure = self.detect_table_structure(section)
            if table_structure and table_structure.confidence > 0.5:
                section_records = self.extract_compatibility_records(
                    table_structure, doc_id, page + i // 10  # Rough page estimation
                )
                records.extend(section_records)
        
        return records
    
    def save_records_to_csv(self, records: List[CompatibilityRecord], output_path: str):
        """Save compatibility records to CSV file"""
        if not records:
            print("No records to save")
            return
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'Head Model', 'Base Model', 'Function', 'Description',
                'Panel Caveats', 'Source Doc', 'Source Page', 'Confidence'
            ])
            
            # Write records
            for record in records:
                writer.writerow([
                    record.head_model,
                    record.base_model, 
                    record.function,
                    record.description,
                    ', '.join(record.panel_caveats),
                    record.source_doc,
                    record.source_page,
                    record.confidence
                ])
        
        print(f"✅ Saved {len(records)} compatibility records to {output_path}")
    
    def save_records_to_json(self, records: List[CompatibilityRecord], output_path: str):
        """Save compatibility records to JSON file"""
        records_dict = []
        for record in records:
            records_dict.append({
                'head_model': record.head_model,
                'base_model': record.base_model,
                'function': record.function,
                'description': record.description,
                'panel_caveats': record.panel_caveats,
                'source_doc': record.source_doc,
                'source_page': record.source_page,
                'confidence': record.confidence
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(records_dict, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved {len(records)} compatibility records to {output_path}")
    
    def get_stats(self, records: List[CompatibilityRecord]) -> Dict[str, Any]:
        """Get statistics about parsed compatibility records"""
        if not records:
            return {"total_records": 0}
        
        functions = {}
        sources = {}
        head_models = set()
        base_models = set()
        
        for record in records:
            # Count functions
            func = record.function
            functions[func] = functions.get(func, 0) + 1
            
            # Count sources
            source = record.source_doc
            sources[source] = sources.get(source, 0) + 1
            
            # Track unique models
            head_models.add(record.head_model)
            base_models.add(record.base_model)
        
        avg_confidence = sum(r.confidence for r in records) / len(records)
        
        return {
            "total_records": len(records),
            "unique_heads": len(head_models),
            "unique_bases": len(base_models), 
            "functions": functions,
            "sources": sources,
            "avg_confidence": round(avg_confidence, 3)
        }