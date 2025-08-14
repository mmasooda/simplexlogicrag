#!/usr/bin/env python3
"""
SKU Normalizer for SimplexRAG based on ChatGPT-5 recommendations
Handles model number variants and synonyms for fire alarm components
"""

import re
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass

@dataclass
class SKUVariant:
    """Represents different ways a SKU can be written"""
    canonical: str      # Standard format: "4098-9714"
    variants: List[str] # All possible variants
    synonyms: List[str] # Descriptive synonyms
    category: str       # Component category

class SKUNormalizer:
    """Normalizes SKU variants and synonyms following ChatGPT-5 recommendations"""
    
    def __init__(self):
        # Core normalization patterns
        self.sku_patterns = [
            r'\b(\d{4})[-\s/](\d{4})\b',        # 4098-9714, 4098 9714, 4098/9714
            r'\b(\d{4})[\s]*(\d{4})\b',         # 40989714, 4098 9714
            r'\b(\d{1,4})[-\s](\d{3,4})\b',     # Short forms
        ]
        
        # Fire alarm specific synonyms based on ChatGPT-5 analysis
        self.synonym_map = {
            # Base synonyms
            "standard base": ["4098-9792"],
            "white base": ["4098-9792"], 
            "relay base": ["4098-9789"],
            "remote led base": ["4098-9789"],
            "sounder base": ["4098-9794"],
            "piezo base": ["4098-9794"],
            "isolator base": ["4098-9793"],
            "isolator2 base": ["4098-9766", "4098-9777"],
            "co base": ["4098-9770"],
            "co sounder base": ["4098-9771"],
            "multi sensor base": ["4098-9796"],
            "multi sensor sounder": ["4098-9795"],
            "4-wire relay base": ["4098-9791"],
            "2-wire relay base": ["4098-9780"],
            
            # Head synonyms  
            "photoelectric head": ["4098-9714"],
            "smoke head": ["4098-9714"],
            "photo head": ["4098-9714"],
            "multi sensor head": ["4098-9754"],
            "photo heat head": ["4098-9754"],
            "combination head": ["4098-9754"],
            "heat head": ["4098-9733"],
            "thermal head": ["4098-9733"],
            
            # Function-based synonyms
            "truealarm photoelectric": ["4098-9714"],
            "truealarm photo/heat": ["4098-9754"], 
            "truealarm heat": ["4098-9733"],
        }
        
        # Reverse synonym lookup
        self.reverse_synonyms = {}
        for desc, skus in self.synonym_map.items():
            for sku in skus:
                if sku not in self.reverse_synonyms:
                    self.reverse_synonyms[sku] = []
                self.reverse_synonyms[sku].append(desc)
        
        # Common abbreviations
        self.abbreviations = {
            "det": "detector",
            "hd": "head", 
            "bs": "base",
            "stnd": "standard",
            "std": "standard",
            "rel": "relay",
            "snd": "sounder",
            "iso": "isolator",
            "co": "carbon monoxide",
            "ms": "multi sensor",
            "ph": "photo heat",
            "led": "light emitting diode"
        }
        
        # Model family patterns
        self.family_patterns = {
            r"409[0-9]": "TrueAlarm",
            r"410[0-9]": "4100ES Series", 
            r"240[0-9]": "4002 Series",
            r"209[0-9]": "TrueAlarm Accessories"
        }
    
    def normalize_sku(self, text: str) -> str:
        """Convert various SKU formats to canonical XXXX-XXXX format"""
        if not text:
            return text
        
        # Try each pattern
        for pattern in self.sku_patterns:
            match = re.search(pattern, text)
            if match:
                part1, part2 = match.groups()
                # Ensure 4-digit format
                part1 = part1.zfill(4)
                part2 = part2.zfill(4)
                return f"{part1}-{part2}"
        
        # If no pattern matches, return original
        return text.strip()
    
    def extract_all_skus(self, text: str) -> List[str]:
        """Extract and normalize all SKUs from text"""
        skus = set()
        
        # Extract using patterns
        for pattern in self.sku_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                normalized = self.normalize_sku(match.group())
                if self.is_valid_sku(normalized):
                    skus.add(normalized)
        
        return list(skus)
    
    def is_valid_sku(self, sku: str) -> bool:
        """Validate if string is a valid Simplex SKU"""
        # Must be XXXX-XXXX format
        if not re.match(r'^\d{4}-\d{4}$', sku):
            return False
        
        # Check for known Simplex patterns
        first_part = sku[:4]
        valid_prefixes = ['2081', '2098', '2099', '4081', '4090', '4098', '4099', 
                         '4100', '4190', '4902', '4903', '4904', '4905', '4906', '4907']
        
        return first_part in valid_prefixes
    
    def resolve_synonym(self, text: str) -> List[str]:
        """Resolve descriptive text to possible SKUs"""
        text_lower = text.lower().strip()
        
        # Direct lookup
        if text_lower in self.synonym_map:
            return self.synonym_map[text_lower]
        
        # Partial matching for compound descriptions
        possible_skus = set()
        
        # Expand abbreviations first
        expanded_text = text_lower
        for abbr, full in self.abbreviations.items():
            expanded_text = re.sub(r'\b' + abbr + r'\b', full, expanded_text)
        
        # Check each synonym for partial matches
        for desc, skus in self.synonym_map.items():
            # Check if description words are in query
            desc_words = set(desc.lower().split())
            query_words = set(expanded_text.split())
            
            # If most description words match, consider it a hit
            if len(desc_words & query_words) >= len(desc_words) * 0.6:  # 60% word overlap
                possible_skus.update(skus)
        
        return list(possible_skus)
    
    def find_sku_variants(self, sku: str) -> List[str]:
        """Find all possible ways to write a given SKU"""
        if not self.is_valid_sku(sku):
            return [sku]
        
        part1, part2 = sku.split('-')
        
        variants = [
            sku,                    # 4098-9714 (canonical)
            f"{part1} {part2}",    # 4098 9714 (space)
            f"{part1}/{part2}",    # 4098/9714 (slash) 
            f"{part1}{part2}",     # 40989714 (no separator)
            f"{part1}.{part2}",    # 4098.9714 (period)
        ]
        
        # Add short forms (remove leading zeros)
        short_part1 = part1.lstrip('0') or '0'
        short_part2 = part2.lstrip('0') or '0'
        
        if short_part1 != part1 or short_part2 != part2:
            variants.extend([
                f"{short_part1}-{short_part2}",
                f"{short_part1} {short_part2}",
                f"{short_part1}/{short_part2}"
            ])
        
        return list(set(variants))  # Remove duplicates
    
    def get_sku_context(self, sku: str) -> Dict[str, str]:
        """Get contextual information about an SKU"""
        if not self.is_valid_sku(sku):
            return {"sku": sku, "valid": False}
        
        # Determine family
        family = "Unknown"
        for pattern, fam in self.family_patterns.items():
            if re.match(pattern, sku):
                family = fam
                break
        
        # Get synonyms
        synonyms = self.reverse_synonyms.get(sku, [])
        
        # Determine category based on known patterns
        category = "Unknown"
        second_part = int(sku.split('-')[1])
        
        if 9100 <= second_part <= 9199:
            category = "Module"
        elif 9200 <= second_part <= 9299:
            category = "Notification"
        elif 9700 <= second_part <= 9899:
            category = "Base"
        elif 9600 <= second_part <= 9699:
            category = "Detector/Head"
        elif 9400 <= second_part <= 9499:
            category = "Detector/Head"
        
        return {
            "sku": sku,
            "valid": True,
            "family": family,
            "category": category,
            "synonyms": synonyms,
            "variants": self.find_sku_variants(sku)
        }
    
    def normalize_query(self, query: str) -> Tuple[str, List[str], List[str]]:
        """Comprehensive query normalization"""
        # Extract explicit SKUs
        explicit_skus = self.extract_all_skus(query)
        
        # Resolve synonym-based SKUs
        synonym_skus = self.resolve_synonym(query)
        
        # Clean up the query by removing found SKUs and normalizing
        cleaned_query = query
        for sku in explicit_skus:
            # Remove various representations of the found SKU
            variants = self.find_sku_variants(sku)
            for variant in variants:
                cleaned_query = re.sub(re.escape(variant), '', cleaned_query, flags=re.IGNORECASE)
        
        # Normalize whitespace
        cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()
        
        return cleaned_query, explicit_skus, synonym_skus
    
    def expand_sku_search(self, sku: str) -> List[str]:
        """Expand a single SKU into all searchable forms"""
        variants = self.find_sku_variants(sku)
        synonyms = self.reverse_synonyms.get(sku, [])
        
        # Add context terms for better matching
        context = self.get_sku_context(sku)
        search_terms = variants + synonyms
        
        if context["category"] != "Unknown":
            search_terms.append(context["category"])
        
        if context["family"] != "Unknown":
            search_terms.append(context["family"])
        
        return list(set(search_terms))

    def get_stats(self) -> Dict[str, int]:
        """Get normalizer statistics"""
        return {
            "synonyms_defined": len(self.synonym_map),
            "reverse_mappings": len(self.reverse_synonyms),
            "abbreviations": len(self.abbreviations),
            "family_patterns": len(self.family_patterns),
            "sku_patterns": len(self.sku_patterns)
        }