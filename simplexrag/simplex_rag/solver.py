"""
Constraint‑based system sizing for Simplex RAG
================================================

This module encapsulates deterministic formulas for selecting the
appropriate panels, modules, and accessories needed to build a
compliant fire alarm system given a set of project requirements.  The
algorithms here are transparent and reproducible: given a BOQ
specification (number of detectors, horn/strobes, speakers, etc.)
they compute the minimum set of components required to satisfy
capacity, power and physical slot constraints.

The design targets Simplex hardware but the abstractions are generic
enough to accommodate other manufacturers by adjusting capacity
constants.  The formulas draw on typical capacities found in
Simplex data sheets: IDNet SLC loops support 250 addressable points,
conventional NAC modules provide four circuits each, audio
amplifiers deliver 100 W at 25 or 70 V and handle eight speaker
circuits, and cabinet bays provide eight 2‑inch slots.  These values
can be tuned via the constants at the top of the module.

Usage:

    from simplex_rag.solver import ConstraintEngine
    engine = ConstraintEngine(db)
    configuration, validation = engine.size_system(requirements)

The ``requirements`` dict is produced by ``LLMInterface.parse_boq`` and
must include keys such as ``smoke_detectors``, ``heat_detectors``,
``horn_strobes`` and ``speakers``.  Optional keys include
``voice_evac``, ``protocol`` and ``certifications``.

"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict
from typing import Dict, List, Tuple

from .data_models import Component, ComponentType
from .database import SimplexDatabase

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Capacity constants (tunable)
# ----------------------------------------------------------------------

SLC_POINTS_PER_LOOP = 250
"""Number of addressable points per IDNet loop."""

NAC_CIRCUITS_PER_MODULE = 4
"""Conventional NAC module provides four circuits."""

HORN_PER_CIRCUIT = 20
"""Assumed number of horn/strobes allowed per NAC circuit (approx.)."""

SPEAKERS_PER_CIRCUIT = 25
"""Approximate number of 1 W speakers per speaker circuit."""

AMPLIFIER_POWER_W = 100.0
"""Power output of a single audio amplifier module (watts)."""

AMPLIFIER_CIRCUITS = 8
"""Number of speaker circuits per amplifier."""

CABINET_SLOTS_PER_BAY = 8
"""Number of 2‑inch module slots per cabinet bay."""


class ConstraintEngine:
    """Compute a panel configuration from project requirements.

    This engine uses simple algebraic formulas to derive the number of
    SLC loops, NAC circuits, amplifiers, power supplies and cabinets
    required by the system.  It then chooses suitable part numbers
    from the database based on component categories and heuristic
    matching.  If the database lacks the necessary components, the
    engine falls back to generic placeholders.
    """

    def __init__(self, db: SimplexDatabase) -> None:
        self.db = db

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def size_system(self, requirements: Dict) -> Tuple[Dict, Dict]:
        """Calculate a bill of materials and validation summary.

        :param requirements: dictionary of project requirements as
            produced by ``LLMInterface.parse_boq``.  Expected keys
            include ``smoke_detectors``, ``heat_detectors``,
            ``manual_stations``, ``horn_strobes``, ``speakers``, and
            ``voice_evac``.
        :return: a tuple ``(configuration, validation)`` where
            ``configuration`` is a dict with component list and
            ``validation`` contains pass/fail checks and reasoning.
        """
        # Aggregate addressable devices
        total_detectors = requirements.get("smoke_detectors", 0) + requirements.get("heat_detectors", 0)
        total_manuals = requirements.get("manual_stations", 0)
        total_addressable = total_detectors + total_manuals
        logger.debug(f"Total addressable devices: {total_addressable}")

        # Calculate loops needed (two loops included in base panel)
        loops_needed = math.ceil(total_addressable / SLC_POINTS_PER_LOOP)
        logger.debug(f"Loops needed (raw): {loops_needed}")

        # Determine NAC circuits required for horns/strobes
        horns = requirements.get("horn_strobes", 0)
        circuits_needed = math.ceil(horns / HORN_PER_CIRCUIT) if horns > 0 else 0
        nac_modules = math.ceil(circuits_needed / NAC_CIRCUITS_PER_MODULE) if circuits_needed > 0 else 0
        logger.debug(f"Horns: {horns}, Circuits: {circuits_needed}, NAC modules: {nac_modules}")

        # Determine speaker circuits and amplifiers if voice evacuation
        speakers = requirements.get("speakers", 0)
        if requirements.get("voice_evac", False) and speakers > 0:
            speaker_circuits = math.ceil(speakers / SPEAKERS_PER_CIRCUIT)
            amp_by_power = math.ceil((speakers * 1.0) / AMPLIFIER_POWER_W)
            amps = max(math.ceil(speaker_circuits / AMPLIFIER_CIRCUITS), amp_by_power)
            audio_modules = 1  # digital audio controller
        else:
            speaker_circuits = 0
            amps = 0
            audio_modules = 0
        logger.debug(
            f"Speakers: {speakers}, speaker circuits: {speaker_circuits}, amps: {amps}, audio controllers: {audio_modules}"
        )

        # Calculate panel type based on loops and voice
        panel_type = "4100ES" if loops_needed > 2 or requirements.get("voice_evac", False) else "4007ES"
        logger.debug(f"Selected panel type: {panel_type}")

        # Determine additional loop cards needed beyond base panel
        base_loops = 2 if panel_type == "4100ES" else 1
        loop_modules = max(0, loops_needed - base_loops)
        logger.debug(f"Loop modules needed: {loop_modules}")

        # Compile initial component list
        components: List[Dict] = []
        # Panel controller
        components.append(self._select_component(panel_type, 1))
        # Loop cards
        if loop_modules > 0:
            components.append(self._select_component("IDNET_LOOP_CARD", loop_modules))
        # NAC modules
        if nac_modules > 0:
            components.append(self._select_component("NAC_MODULE", nac_modules))
        # Audio controller
        if audio_modules > 0:
            components.append(self._select_component("AUDIO_CONTROLLER", audio_modules))
        # Amplifiers
        if amps > 0:
            components.append(self._select_component("AMPLIFIER", amps))
        # Batteries and power supplies (simplified sizing)
        power_supply_count = 1
        components.append(self._select_component("POWER_SUPPLY", power_supply_count))
        battery_ah = self._estimate_battery_ah(total_addressable, horns, speakers, loops_needed)
        battery_quantity = math.ceil(battery_ah / 7.0)  # assume 7 Ah battery units
        if battery_quantity > 0:
            components.append(self._select_component("BATTERY", battery_quantity))

        # Add detectors, bases and manual stations
        components.append(self._select_component("DETECTOR", total_detectors))
        components.append(self._select_component("BASE", total_detectors))
        if total_manuals > 0:
            components.append(self._select_component("MANUAL_STATION", total_manuals))
        if horns > 0:
            components.append(self._select_component("HORN_STROBE", horns))
        if speakers > 0:
            components.append(self._select_component("SPEAKER", speakers))

        # Validate the physical slot usage
        total_slots = self._calculate_total_slots(components)
        bays_needed = math.ceil(total_slots / CABINET_SLOTS_PER_BAY)
        components.append(self._select_component("CABINET", bays_needed))
        logger.debug(f"Total slot units: {total_slots}, bays needed: {bays_needed}")

        configuration = {
            "components": components,
            "requirements": requirements,
            "metadata": {"generated_at": "", "version": "1.0"},
        }

        # Generate validation results
        validation = self._validate(configuration)
        return configuration, validation

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _select_component(self, role: str, quantity: int) -> Dict:
        """Select a component from the database matching the role and return a dict.

        The role is a high‑level identifier (e.g. ``"AMPLIFIER"``, ``"IDNET_LOOP_CARD"``).
        This function searches the database for a component whose description or
        category matches the role.  If none is found, it returns a generic
        placeholder with the role as the description.
        """
        # Map role to category or name pattern
        role_to_category = {
            "AMPLIFIER": ComponentType.AMPLIFIER,
            "AUDIO_CONTROLLER": ComponentType.AUDIO_MODULE,
            "IDNET_LOOP_CARD": ComponentType.SLC_MODULE,
            "NAC_MODULE": ComponentType.NAC_MODULE,
            "POWER_SUPPLY": ComponentType.POWER_SUPPLY,
            "BATTERY": ComponentType.BATTERY,
            "DETECTOR": ComponentType.SENSOR,
            "BASE": ComponentType.BASE,
            "MANUAL_STATION": ComponentType.MANUAL_STATION,
            "HORN_STROBE": ComponentType.NOTIFICATION,
            "SPEAKER": ComponentType.NOTIFICATION,
            "CABINET": ComponentType.ACCESSORY,
            # Additional internal modules
            "RELAY_CARD": ComponentType.RELAY,
            "CLASS_A_ADAPTOR": ComponentType.ACCESSORY,
            "FLEX_50_AMPLIFIER": ComponentType.AMPLIFIER,
            "FLEX_35_AMPLIFIER": ComponentType.AMPLIFIER,
            "SPEAKER_EXPANSION_CARD": ComponentType.AUDIO_MODULE,
            "SUPERVISION_ADAPTOR": ComponentType.ACCESSORY,
            "RUI_CARD": ComponentType.INTERFACE,
            "RS232_CARD": ComponentType.INTERFACE,
            "EXPANSION_PSU": ComponentType.POWER_SUPPLY,
            "RPS": ComponentType.POWER_SUPPLY,
            "SPS": ComponentType.POWER_SUPPLY,
        }
        category = role_to_category.get(role)
        comp: Component | None = None
        # Search for first component matching category
        for pn, data in self.db.graph.nodes(data=True):
            if category and data.get("category") == category.value:
                comp = Component(**{**data})  # type: ignore
                break
        # If none found, create placeholder
        if comp is None:
            description = role.replace("_", " ").title()
            comp = Component(
                part_number=f"{role}_GENERIC",
                sku_type="product",
                category=category or ComponentType.ACCESSORY,
                description=description,
            )
        return {
            "part_number": comp.part_number,
            "description": comp.description,
            "quantity": quantity,
        }

    def _estimate_battery_ah(self, total_addr: int, horns: int, speakers: int, loops: int) -> float:
        """Estimate required battery amp‑hour capacity.

        This simplified calculation assumes current draw per device and adds a
        margin.  Standby current (24 h) and alarm current (15 min) are
        combined to yield a total amp‑hour requirement.
        """
        standby_mA = total_addr * 0.5 + horns * 2.0 + speakers * 1.5 + loops * 10.0
        alarm_mA = total_addr * 2.0 + horns * 8.0 + speakers * 5.0 + loops * 20.0
        ah = ((standby_mA * 24.0) + (alarm_mA * 0.25)) / 1000.0
        logger.debug(
            f"Battery sizing: standby {standby_mA} mA, alarm {alarm_mA} mA, total AH {ah:.2f}"
        )
        return ah

    def _calculate_total_slots(self, components: List[Dict]) -> int:
        """Sum the slot units of selected components based on their categories."""
        # Rough slot sizes based on part types
        slot_sizes = {
            ComponentType.AMPLIFIER.value: 2,
            ComponentType.AUDIO_MODULE.value: 2,
            ComponentType.SLC_MODULE.value: 1,
            ComponentType.NAC_MODULE.value: 1,
            ComponentType.POWER_SUPPLY.value: 1,
            ComponentType.BATTERY.value: 0,
            ComponentType.SENSOR.value: 0,
            ComponentType.BASE.value: 0,
            ComponentType.MANUAL_STATION.value: 0,
            ComponentType.NOTIFICATION.value: 0,
            ComponentType.ACCESSORY.value: 2,  # Cabinet
        }
        total = 0
        for comp in components:
            # Lookup actual slot size if we have the component in graph
            data = self.db.graph.nodes.get(comp["part_number"], {})
            slot = data.get("slot_size")
            if slot is not None:
                total += slot * comp.get("quantity", 1)
            else:
                # Fallback to mapping
                cat = data.get("category") or comp.get("description", "").upper()
                # Use first word as key if description provided
                if cat in slot_sizes:
                    total += slot_sizes.get(cat, 1) * comp.get("quantity", 1)
                else:
                    # Default to 1 slot unit
                    total += 1 * comp.get("quantity", 1)
        return total

    def _validate(self, config: Dict) -> Dict:
        """Validate the final configuration and return pass/fail checks."""
        checks = {}
        valid = True
        # Check loops vs modules
        loops_in_config = sum(
            comp["quantity"] * (SLC_POINTS_PER_LOOP if "IDNET_LOOP_CARD" in comp["part_number"] else 0)
            for comp in config["components"]
        )
        addr_req = config["requirements"].get("smoke_detectors", 0) + config["requirements"].get("heat_detectors", 0) + config["requirements"].get("manual_stations", 0)
        loops_needed = math.ceil(addr_req / SLC_POINTS_PER_LOOP)
        checks["loop_capacity"] = {
            "valid": loops_in_config >= loops_needed,
            "message": f"System requires {loops_needed} loops; configuration provides {loops_in_config/SLC_POINTS_PER_LOOP:.1f}",
        }
        if not checks["loop_capacity"]["valid"]:
            valid = False
        # Additional checks can be added here (power budget, speaker circuits, etc.)
        return {"valid": valid, "checks": checks}
