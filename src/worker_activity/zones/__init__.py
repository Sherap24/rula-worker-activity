"""Restricted-zone event detection from pose body-center (Week 6).

Events are spatial crossings — not activity class labels.
"""

from __future__ import annotations

from worker_activity.zones.events import (
    ZoneEventDetectionResult,
    detect_zone_events,
    load_zones_config,
)

__all__ = [
    "ZoneEventDetectionResult",
    "detect_zone_events",
    "load_zones_config",
]
