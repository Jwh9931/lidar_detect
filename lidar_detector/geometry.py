"""Geometry primitives for 3D point cloud detections."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BBox3D:
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    yaw: float

    @property
    def length(self) -> float:
        return self.size[0]

    @property
    def width(self) -> float:
        return self.size[1]

    @property
    def height(self) -> float:
        return self.size[2]

    def to_dict(self) -> dict[str, Any]:
        return {
            "center": [round(v, 4) for v in self.center],
            "size": [round(v, 4) for v in self.size],
            "yaw": round(self.yaw, 4),
        }


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: BBox3D
    num_points: int
    rail_overlap: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "num_points": int(self.num_points),
            "rail_overlap": round(float(self.rail_overlap), 4),
            "bbox": self.bbox.to_dict(),
        }


def oriented_bbox(points_xyz: np.ndarray) -> BBox3D:
    """Estimate a PCA-oriented box in XY and axis-aligned extent in Z."""
    if points_xyz.ndim != 2 or points_xyz.shape[1] < 3:
        raise ValueError("points_xyz must have shape N x 3.")
    if len(points_xyz) == 0:
        raise ValueError("Cannot compute a bounding box for an empty cluster.")

    xyz = np.asarray(points_xyz[:, :3], dtype=float)
    xy = xyz[:, :2]
    mean_xy = xy.mean(axis=0)
    centered = xy - mean_xy

    if len(xy) >= 3 and np.linalg.norm(centered) > 1e-9:
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        axes = eigvecs[:, order]
    else:
        axes = np.eye(2)

    local = centered @ axes
    min_xy = local.min(axis=0)
    max_xy = local.max(axis=0)
    extents = max_xy - min_xy

    axis_index = 0
    if extents[1] > extents[0]:
        axes = axes[:, [1, 0]]
        local = centered @ axes
        min_xy = local.min(axis=0)
        max_xy = local.max(axis=0)
        extents = max_xy - min_xy
        axis_index = 1

    center_local = (min_xy + max_xy) / 2.0
    center_xy = center_local @ axes.T + mean_xy
    z_min = float(xyz[:, 2].min())
    z_max = float(xyz[:, 2].max())
    center_z = (z_min + z_max) / 2.0
    yaw = atan2(float(axes[1, 0]), float(axes[0, 0]))

    length = float(extents[0])
    width = float(extents[1])
    height = float(z_max - z_min)
    if axis_index == 1:
        yaw = atan2(float(axes[1, 0]), float(axes[0, 0]))

    return BBox3D(
        center=(float(center_xy[0]), float(center_xy[1]), center_z),
        size=(length, width, height),
        yaw=float(yaw),
    )


def point_to_segment_distances(points_xy: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom <= 1e-12:
        return np.linalg.norm(points_xy - start, axis=1)

    t = ((points_xy - start) @ segment) / denom
    t = np.clip(t, 0.0, 1.0)
    projection = start + t[:, None] * segment
    return np.linalg.norm(points_xy - projection, axis=1)


def point_to_polyline_distances(points_xy: np.ndarray, polyline: list[list[float]]) -> np.ndarray:
    if len(polyline) < 2:
        return np.full(len(points_xy), np.inf)

    distances = np.full(len(points_xy), np.inf)
    vertices = np.asarray(polyline, dtype=float)
    for i in range(len(vertices) - 1):
        segment_distances = point_to_segment_distances(points_xy, vertices[i], vertices[i + 1])
        distances = np.minimum(distances, segment_distances)
    return distances


def rail_overlap(points_xyz: np.ndarray, centerlines: list[list[list[float]]], corridor_width: float) -> float:
    if not centerlines or len(points_xyz) == 0:
        return 0.0

    points_xy = np.asarray(points_xyz[:, :2], dtype=float)
    threshold = max(float(corridor_width), 0.0) / 2.0
    min_distances = np.full(len(points_xy), np.inf)
    for centerline in centerlines:
        min_distances = np.minimum(min_distances, point_to_polyline_distances(points_xy, centerline))

    return float(np.mean(min_distances <= threshold))
