"""Polygon geometry helpers for zone-event detection."""

from __future__ import annotations

from typing import Sequence


Point = tuple[float, float]


def point_in_polygon(x: float, y: float, vertices: Sequence[Point]) -> bool:
    """Ray-casting point-in-polygon test (includes boundary approximately)."""
    if len(vertices) < 3:
        return False
    inside = False
    n = len(vertices)
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-15) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def normalize_vertices(raw: Sequence[Sequence[float]]) -> list[Point]:
    points: list[Point] = []
    for item in raw:
        if len(item) < 2:
            continue
        points.append((float(item[0]), float(item[1])))
    return points
