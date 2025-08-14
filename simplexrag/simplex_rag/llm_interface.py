"""
LLM integration for the Simplex RAG engine.

This module abstracts interactions with large language models.  It
supports both standard chat completions for on‑the‑fly parsing and
report generation, and the OpenAI batch API for efficient bulk
extraction of structured data from datasheets.  When API keys are
unavailable, the interface falls back to regex‑based heuristics.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

try:
    import google.generativeai as genai  # type: ignore
except ImportError:
    genai = None  # type: ignore

import openai  # type: ignore
import re

from .config import settings

logger = logging.getLogger(__name__)


class LLMInterface:
    """Interface for interacting with LLMs for parsing and report generation."""

    def __init__(self) -> None:
        self.model_name = settings.llm_model or "gpt-5-mini"  # Default to GPT-5-mini
        self.use_batch_api = settings.use_batch_api
        # Determine provider
        self._openai_available = bool(os.environ.get("OPENAI_API_KEY"))
        self._gemini_available = bool(os.environ.get("GEMINI_API_KEY")) and genai is not None
        
        # Initialize OpenAI client (v1+ API)
        if self._openai_available:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        
        # Initialize Gemini
        if self._gemini_available:
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            # Use latest Gemini model
            gemini_model_name = "gemini-2.5-flash" if "gpt" in self.model_name else self.model_name
            self.gemini_model = genai.GenerativeModel(gemini_model_name)

    def is_available(self) -> bool:
        """Return True if any LLM provider is available."""
        return self._openai_available or self._gemini_available

    # ------------------------------------------------------------------
    # Batch extraction via OpenAI
    # ------------------------------------------------------------------
    def batch_extract(self, tasks: List[Dict]) -> Dict[str, str]:
        """Submit a batch of extraction tasks and return a dict of id->response.

        Each task must be a dict with ``id`` and ``prompt`` keys.  When
        the batch API is unavailable or keys are missing, this method
        returns empty strings for each id.
        """
        if not self.use_batch_api or not self._openai_available:
            return {task["id"]: "" for task in tasks}
        # Prepare JSONL file of tasks
        tasks_file = settings.batch_tasks_path
        with open(tasks_file, "w") as f:
            for task in tasks:
                payload = {
                    "custom_id": task["id"],
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model_name,  # Now defaults to gpt-5-mini
                        "temperature": 0,
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant that extracts structured data from text."},
                            {"role": "user", "content": task["prompt"]},
                        ],
                    },
                }
                f.write(json.dumps(payload) + "\n")
        logger.info(f"Submitted {len(tasks)} tasks to batch API; stored in {tasks_file}")
        
        # Calculate dynamic timeout based on batch size and number of tasks
        import os
        batch_file_size_kb = os.path.getsize(tasks_file) / 1024  # Convert to KB
        
        # Dynamic timeout calculation:
        # - Base: 15 minutes (to handle OpenAI's processing overhead)
        # - Add 30 seconds per task (since each task is an LLM call)
        # - Add extra time for larger batches
        base_timeout = 900  # 15 minutes base
        timeout_per_task = 30  # 30 seconds per task
        
        # Calculate timeout
        max_wait_time = base_timeout + (len(tasks) * timeout_per_task)
        
        # For very small batches, ensure at least 20 minutes
        max_wait_time = max(max_wait_time, 1200)  # At least 20 minutes
        
        # Cap at 60 minutes maximum
        max_wait_time = min(max_wait_time, 3600)
        
        logger.info(f"Batch: {len(tasks)} tasks, {batch_file_size_kb:.1f} KB, timeout: {max_wait_time/60:.1f} minutes")
        
        # Upload file and start batch job using modern OpenAI client
        try:
            with open(tasks_file, "rb") as f:
                upload = self.openai_client.files.create(file=f, purpose="batch")
            batch = self.openai_client.batches.create(
                input_file_id=upload.id, 
                endpoint="/v1/chat/completions", 
                completion_window="24h"
            )
            # Poll until completion with timeout
            import time
            start_time = time.time()
            
            while batch.status not in {"completed", "succeeded", "failed"}:
                if time.time() - start_time > max_wait_time:
                    logger.warning(f"Batch job {batch.id} taking too long, continuing without LLM enhancement")
                    return {task["id"]: "" for task in tasks}
                    
                logger.info(f"Batch job {batch.id} status: {batch.status}")
                time.sleep(5)  # Wait 5 seconds between polls
                batch = self.openai_client.batches.retrieve(batch.id)
            if batch.status == "failed":
                logger.error(f"Batch job failed: {batch.error}")
                return {task["id"]: "" for task in tasks}
            # Download output file using modern client
            result_file = self.openai_client.files.content(batch.output_file_id)
            responses = {}
            # Check if result_file needs decoding (bytes) or is already a string
            if isinstance(result_file, bytes):
                result_content = result_file.decode()
            else:
                result_content = result_file
            for line in result_content.splitlines():
                obj = json.loads(line)
                custom_id = obj.get("custom_id")
                # Navigate the correct structure: response.body.choices[0].message.content
                response = obj.get("response", {})
                body = response.get("body", {})
                choices = body.get("choices", [])
                content = ""
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                responses[custom_id] = content
            return responses
        except Exception as e:
            logger.error(f"Batch extraction failed: {e}")
            return {task["id"]: "" for task in tasks}

    # ------------------------------------------------------------------
    # BOQ parsing and report generation
    # ------------------------------------------------------------------
    def parse_boq(self, text: str) -> Dict:
        """Parse a BOQ text into structured requirements using the best available method."""
        if self._gemini_available:
            # Use Google Gemini
            prompt = (
                "Extract fire alarm system requirements from the following BOQ text.\n"
                "Return a JSON object with keys: smoke_detectors, heat_detectors, manual_stations, horn_strobes, speakers, voice_evac, region, protocol, certifications."
                "\nBOQ:\n" + text
            )
            try:
                response = self.gemini_model.generate_content(prompt)
                content = response.text.strip()
                # Remove markdown code fences
                content = content.replace("```json", "").replace("```", "")
                return json.loads(content)
            except Exception as e:
                logger.warning(f"Gemini parse failed: {e}")
        # Fallback: simple regex
        return self._regex_parse_boq(text)

    def _regex_parse_boq(self, text: str) -> Dict:
        """Simple regex fallback for BOQ parsing."""
        requirements = {
            "smoke_detectors": 0,
            "heat_detectors": 0,
            "manual_stations": 0,
            "horn_strobes": 0,
            "speakers": 0,
            "voice_evac": False,
            "region": "US",
            "protocol": "IDNET",
            "certifications": ["UL"],
        }
        patterns = {
            "smoke_detectors": r"smoke\s+detector[s]?\s*[:=]?[\s]*(\d+)",
            "heat_detectors": r"heat\s+detector[s]?\s*[:=]?[\s]*(\d+)",
            "manual_stations": r"(?:manual|pull)\s+station[s]?\s*[:=]?[\s]*(\d+)",
            "horn_strobes": r"horn[/\s]*strobe[s]?\s*[:=]?[\s]*(\d+)",
            "speakers": r"speaker[s]?\s*[:=]?[\s]*(\d+)",
        }
        lower = text.lower()
        for field, pattern in patterns.items():
            match = re.search(pattern, lower, re.IGNORECASE)
            if match:
                requirements[field] = int(match.group(1))
        if "voice" in lower or requirements["speakers"] > 0:
            requirements["voice_evac"] = True
        if any(term in lower for term in ["middle east", "asia"]):
            requirements["region"] = "ME"
            requirements["protocol"] = "MX"
        elif "canada" in lower:
            requirements["region"] = "CANADA"
            requirements["certifications"] = ["ULC"]
        return requirements

    def generate_report(self, configuration: Dict, validation: Dict) -> str:
        """Generate a human‑readable report for the configuration."""
        if self._gemini_available:
            prompt = (
                "You are a fire alarm system engineer. Generate a professional report summarizing the following configuration and validation results.\n"
                f"Configuration JSON:\n{json.dumps(configuration, indent=2)}\n"
                f"Validation JSON:\n{json.dumps(validation, indent=2)}\n"
                "Include an executive summary, a breakdown by component categories, compliance notes, power/battery calculations and installation hints."
            )
            try:
                response = self.gemini_model.generate_content(prompt)
                return response.text
            except Exception as e:
                logger.warning(f"Gemini report generation failed: {e}")
        # Fallback: simple template
        return self._template_generate_report(configuration, validation)

    def _template_generate_report(self, configuration: Dict, validation: Dict) -> str:
        lines: List[str] = []
        lines.append("=" * 80)
        lines.append("SIMPLEX FIRE ALARM SYSTEM - QUOTATION REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {configuration.get('metadata', {}).get('generated_at', '')}")
        lines.append(f"System Version: {configuration.get('metadata', {}).get('version', '')}")
        lines.append("\nEXECUTIVE SUMMARY\n" + "=" * 40)
        total_devices = sum(
            comp.get("quantity", 1)
            for comp in configuration.get("components", [])
            if any(cat in comp.get("description", "").lower() for cat in ["detector", "station", "horn", "speaker"])
        )
        lines.append(f"Total Addressable Points: {total_devices}")
        lines.append(f"Configuration Status: {'VALID' if validation.get('valid') else 'INVALID'}")
        # Category grouping
        from collections import defaultdict
        categories = defaultdict(list)
        for comp in configuration.get("components", []):
            desc = comp.get("description", "").lower()
            if "controller" in desc:
                cat = "Control Panel"
            elif any(k in desc for k in ["power", "battery"]):
                cat = "Power System"
            elif any(k in desc for k in ["module", "card"]):
                cat = "Modules"
            elif any(k in desc for k in ["detector", "sensor"]):
                cat = "Detection Devices"
            elif "base" in desc:
                cat = "Bases"
            elif any(k in desc for k in ["horn", "strobe", "speaker"]):
                cat = "Notification"
            elif any(k in desc for k in ["audio", "amplifier"]):
                cat = "Audio System"
            else:
                cat = "Accessories"
            categories[cat].append(comp)
        lines.append("\nSYSTEM COMPONENTS\n" + "=" * 40)
        for cat, comps in categories.items():
            lines.append(f"\n{cat}:")
            for comp in comps:
                lines.append(f"  - {comp.get('description', '')} (Part #: {comp['part_number']}, Qty: {comp.get('quantity', 1)})")
        lines.append("\nVALIDATION RESULTS\n" + "=" * 40)
        for name, res in validation.get("checks", {}).items():
            status = "PASS" if res.get("valid") else "FAIL"
            lines.append(f"- {name.replace('_', ' ').title()}: {status} - {res.get('message')}")
        lines.append("\nCOMPLIANCE\n" + "=" * 40)
        lines.append(f"Required Certifications: {', '.join(configuration.get('requirements', {}).get('certifications', ['UL']))}")
        lines.append("All selected components meet or exceed the specified certification requirements.")
        lines.append("\nINSTALLATION NOTES\n" + "=" * 40)
        lines.append("• Follow NFPA 72 guidelines for device spacing and wiring.")
        lines.append("• Use appropriate wire gauge for all circuits.")
        lines.append("• Ensure proper grounding and shielding.")
        lines.append("• Program devices sequentially for optimal performance.")
        lines.append("\n" + "=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)
        return "\n".join(lines)