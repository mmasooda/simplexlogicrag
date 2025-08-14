"""
Data models for the Simplex RAG engine.

This module defines the structured types used throughout the system,
including component categories, protocols and component definitions.
These models are intentionally simple; more complex validation is
performed at runtime when populating the database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple


class Protocol(Enum):
    """Fire alarm communication protocols with key specifications."""

    IDNET = ("IDNet", 250, 10000, "ft", 3333)
    IDNET_PLUS = ("IDNet+", 250, 10000, "ft", 3333)
    MX = ("MX", 250, 2000, "m", 9600)
    MAPNET_II = ("MAPNET II", 127, 10000, "ft", 2400)
    MAPNET_III = ("MAPNET III", 127, 10000, "ft", 2400)
    IDNAC = ("IDNAC", 127, 8000, "ft", 9600)

    def __init__(self, display_name: str, max_devices: int, max_distance: int, distance_unit: str, baud_rate: int) -> None:
        self.display_name = display_name
        self.max_devices = max_devices
        self.max_distance = max_distance
        self.distance_unit = distance_unit
        self.baud_rate = baud_rate


class ComponentType(Enum):
    """Component categories."""

    PANEL = "panel"
    POWER_SUPPLY = "power_supply"
    CPU_MODULE = "cpu_module"
    SLC_MODULE = "slc_module"
    NAC_MODULE = "nac_module"
    AUDIO_MODULE = "audio_module"
    AMPLIFIER = "amplifier"
    SENSOR = "sensor"
    BASE = "base"
    MANUAL_STATION = "manual_station"
    NOTIFICATION = "notification"
    ISOLATOR = "isolator"
    RELAY = "relay"
    INTERFACE = "interface"
    NETWORK_CARD = "network_card"
    BATTERY = "battery"
    ACCESSORY = "accessory"


class CertificationType(Enum):
    """Certification standards."""

    UL = "UL 864"
    ULC = "ULC-S527"
    FM = "FM 3010"
    CSFM = "California State Fire Marshal"
    NYC_MEA = "NYC MEA"
    EN54 = "EN 54"


@dataclass
class Component:
    """A component definition extracted from a datasheet."""

    part_number: str
    sku_type: str
    category: ComponentType
    description: str
    manufacturer: str = "Simplex"
    document_reference: str = ""
    revision: str = ""
    protocols: List[Protocol] = field(default_factory=list)
    voltage_nominal: float = 24.0
    voltage_min: float = 19.5
    voltage_max: float = 31.5
    current_supervisory_mA: float = 0.0
    current_alarm_mA: float = 0.0
    power_output_W: Optional[float] = None
    capacity_points: Optional[int] = None
    capacity_devices: Optional[int] = None
    capacity_circuits: Optional[int] = None
    max_wire_distance_ft: Optional[int] = None
    max_wire_resistance_ohms: Optional[float] = None
    max_wire_capacitance_uF: Optional[float] = None
    slot_type: str = "block"
    slot_size: int = 1
    weight_kg: Optional[float] = None
    dimensions_mm: Optional[Dict[str, float]] = None
    mounting_location: List[str] = field(default_factory=list)
    compatible_panels: List[str] = field(default_factory=list)
    compatible_with: List[str] = field(default_factory=list)
    requires: List[str] = field(default_factory=list)
    excludes: List[str] = field(default_factory=list)
    max_per_system: Optional[int] = None
    max_per_panel: Optional[int] = None
    max_per_loop: Optional[int] = None
    max_per_bay: Optional[int] = None
    certifications: List[CertificationType] = field(default_factory=list)
    unit_cost: Optional[float] = None
    lead_time_days: Optional[int] = None
    date_created: str = field(default_factory=lambda: datetime.now().isoformat())
    date_modified: str = field(default_factory=lambda: datetime.now().isoformat())

    def validate(self) -> None:
        """Perform sanity checks on the component definition."""
        if self.current_alarm_mA < self.current_supervisory_mA:
            raise ValueError(f"{self.part_number}: alarm current less than supervisory current")
        if self.voltage_min > self.voltage_max:
            raise ValueError(f"{self.part_number}: invalid voltage range")