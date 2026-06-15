from __future__ import annotations

from math import hypot
from typing import Sequence, Tuple

BBox = Sequence[float]
Point = Tuple[int, int]


def get_center_bbox(bbox: BBox) -> Point:
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_width_bbox(bbox: BBox) -> float:
    return float(bbox[2] - bbox[0])


def measure_distance(p1: Sequence[float], p2: Sequence[float]) -> float:
    return hypot(float(p1[0]) - float(p2[0]), float(p1[1]) - float(p2[1]))


def measure_xy_distance(p1: Sequence[float], p2: Sequence[float]) -> tuple[float, float]:
    return float(p1[0]) - float(p2[0]), float(p1[1]) - float(p2[1])


def get_foot_position(bbox: BBox) -> Point:
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)
