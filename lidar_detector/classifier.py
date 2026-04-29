"""Rule-based cluster classifier."""

from __future__ import annotations

import numpy as np

from .geometry import Detection, oriented_bbox, rail_overlap


def _range_score(value: float, bounds: list[float], tolerance: float = 0.15) -> float:
    lower, upper = float(bounds[0]), float(bounds[1])
    if lower <= value <= upper:
        return 1.0
    span = max(upper - lower, 1e-6)
    margin = span * tolerance
    if value < lower:
        return max(0.0, 1.0 - (lower - value) / max(margin, 1e-6))
    return max(0.0, 1.0 - (value - upper) / max(margin, 1e-6))


def _min_score(value: float, minimum: float) -> float:
    return min(1.0, float(value) / max(float(minimum), 1e-6))


def _max_score(value: float, maximum: float) -> float:
    return min(1.0, max(float(maximum), 1e-6) / max(float(value), 1e-6))


class RuleBasedClassifier:
    def __init__(self, config: dict):
        self.config = config
        self.class_config = config.get("classification", {})
        self.rail_config = config.get("rail", {})

    def _vehicle_score(self, length: float, width: float, height: float, num_points: int) -> float:
        cfg = self.class_config.get("vehicle", {})
        aspect_ratio = length / max(width, 1e-6)
        footprint_area = length * width
        min_footprint_area = float(cfg.get("min_footprint_area", 1.4))
        if footprint_area < min_footprint_area:
            return 0.0
        dimension_score = min(
            _range_score(length, cfg.get("length", [2.0, 12.0])),
            _range_score(width, cfg.get("width", [1.2, 3.5])),
            _range_score(height, cfg.get("height", [0.9, 4.0])),
            _range_score(aspect_ratio, cfg.get("length_width_ratio", [1.1, 4.2])),
            _max_score(footprint_area, cfg.get("max_footprint_area", 28.0)),
        )
        return float(0.85 * dimension_score + 0.15 * _min_score(num_points, cfg.get("min_points", 25)))

    def _pedestrian_score(self, length: float, width: float, height: float, num_points: int) -> float:
        cfg = self.class_config.get("pedestrian", {})
        footprint = max(length, width)
        narrow_side = min(length, width)
        dimension_score = min(
            _range_score(height, cfg.get("height", [0.8, 2.4])),
            _range_score(footprint, cfg.get("footprint", [0.2, 1.2])),
            _min_score(narrow_side, cfg.get("min_width", 0.12)),
        )
        return float(0.85 * dimension_score + 0.15 * _min_score(num_points, cfg.get("min_points", 8)))

    def _xy_fill_ratio(self, cluster: np.ndarray, bbox_length: float, bbox_width: float, yaw: float) -> float:
        if bbox_length <= 1e-6 or bbox_width <= 1e-6:
            return 0.0
        shape_cfg = self.class_config.get("shape_features", {})
        cell_size = float(shape_cfg.get("cell_size", 0.25))
        if cell_size <= 0 or len(cluster) == 0:
            return 1.0

        xy = cluster[:, :2].astype(float, copy=False)
        center = xy.mean(axis=0)
        c, s = np.cos(-yaw), np.sin(-yaw)
        rot = np.array([[c, -s], [s, c]], dtype=float)
        local = (xy - center) @ rot.T
        cells = np.floor((local - local.min(axis=0)) / cell_size).astype(np.int64)
        occupied = len(np.unique(cells, axis=0))
        total = max(1, int(np.ceil(bbox_length / cell_size)) * int(np.ceil(bbox_width / cell_size)))
        return float(occupied / total)

    def _is_static_structure_like(self, cluster: np.ndarray, length: float, width: float, height: float, yaw: float) -> bool:
        cfg = self.class_config.get("static_rejection", {})
        if not cfg.get("enabled", True):
            return False

        aspect_ratio = length / max(width, 1e-6)
        footprint_area = length * width
        height_width_ratio = height / max(width, 1e-6)
        fill_ratio = self._xy_fill_ratio(cluster, length, width, yaw)
        wall_cfg = cfg.get("wall_like", {})
        building_cfg = cfg.get("building_like", {})
        thin_cfg = cfg.get("thin_vertical_like", {})
        line_cfg = cfg.get("line_like", {})

        wall_like = (
            length >= float(wall_cfg.get("min_length", 5.0))
            and width <= float(wall_cfg.get("max_width", 1.4))
            and height >= float(wall_cfg.get("min_height", 1.6))
            and aspect_ratio >= float(wall_cfg.get("min_aspect_ratio", 4.0))
        )
        building_like = (
            footprint_area >= float(building_cfg.get("max_footprint_area", 26.0))
            and height >= float(building_cfg.get("min_height", 2.2))
        )
        thin_vertical_like = (
            length >= float(thin_cfg.get("min_length", 1.8))
            and width <= float(thin_cfg.get("max_width", 1.0))
            and height >= float(thin_cfg.get("min_height", 0.8))
            and height_width_ratio >= float(thin_cfg.get("min_height_width_ratio", 1.45))
            and aspect_ratio >= float(thin_cfg.get("min_aspect_ratio", 2.3))
        )
        line_like = (
            length >= float(line_cfg.get("min_length", 2.0))
            and aspect_ratio >= float(line_cfg.get("min_aspect_ratio", 3.0))
            and fill_ratio <= float(line_cfg.get("max_fill_ratio", 0.32))
        )
        return wall_like or building_like or thin_vertical_like or line_like

    def _rail_obstacle_label(self, length: float, width: float, height: float) -> str:
        cfg = self.class_config.get("large_rail_obstacle", {})
        if not cfg.get("enabled", True):
            return "unknown_rail_obstacle"
        min_size = cfg.get("min_size", [2.0, 1.0, 1.0])
        sorted_xy = sorted([length, width], reverse=True)
        required_xy = sorted([float(min_size[0]), float(min_size[1])], reverse=True)
        if sorted_xy[0] >= required_xy[0] and sorted_xy[1] >= required_xy[1] and height >= float(min_size[2]):
            return "large_rail_obstacle"
        return "unknown_rail_obstacle"

    def classify(self, cluster: np.ndarray) -> Detection | None:
        min_points = int(self.class_config.get("min_cluster_points", 10))
        if len(cluster) < min_points:
            return None

        bbox = oriented_bbox(cluster[:, :3])
        if bbox.height < float(self.class_config.get("min_height", 0.25)):
            return None

        length, width, height = bbox.size
        overlap = rail_overlap(
            cluster[:, :3],
            self.rail_config.get("centerlines", []),
            float(self.rail_config.get("corridor_width", 0.0)),
        )
        min_overlap = float(self.rail_config.get("min_overlap", 0.2))

        if self._is_static_structure_like(cluster, length, width, height, bbox.yaw):
            if overlap >= min_overlap:
                return Detection(
                    label=self._rail_obstacle_label(length, width, height),
                    confidence=min(0.95, 0.5 + overlap * 0.5),
                    bbox=bbox,
                    num_points=len(cluster),
                    rail_overlap=overlap,
                )
            return None

        scores = {
            "vehicle": self._vehicle_score(length, width, height, len(cluster)),
            "pedestrian": self._pedestrian_score(length, width, height, len(cluster)),
        }
        label, confidence = max(scores.items(), key=lambda item: item[1])
        threshold = float(self.class_config.get("confidence_threshold", 0.45))

        if confidence >= threshold:
            return Detection(
                label=label,
                confidence=confidence,
                bbox=bbox,
                num_points=len(cluster),
                rail_overlap=overlap,
            )

        if overlap >= min_overlap:
            return Detection(
                label=self._rail_obstacle_label(length, width, height),
                confidence=min(0.95, 0.5 + overlap * 0.5),
                bbox=bbox,
                num_points=len(cluster),
                rail_overlap=overlap,
            )

        return None
