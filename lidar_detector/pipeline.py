"""End-to-end LiDAR object detection pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .classifier import RuleBasedClassifier
from .clustering import dbscan, extract_clusters, grid_connected_components
from .config import load_config
from .geometry import Detection
from .io import load_point_cloud
from .postprocess import apply_output_policy
from .preprocess import preprocess_points


class LidarObjectDetector:
    def __init__(self, config: dict | None = None):
        self.config = load_config(override=config)
        self.classifier = RuleBasedClassifier(self.config)

    def detect_array(self, points: np.ndarray) -> list[Detection]:
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError("points must have shape N x 3 or N x 4+.")

        prepared = preprocess_points(points, self.config)
        cluster_config = self.config.get("clustering", {})
        if cluster_config.get("method", "grid") == "grid":
            clusters = grid_connected_components(
                prepared,
                cell_size=float(cluster_config.get("grid_cell_size", cluster_config.get("eps", 0.75))),
                min_points=int(cluster_config.get("min_points", 8)),
                max_cluster_points=int(cluster_config.get("max_cluster_points", 30000)),
                connectivity=int(cluster_config.get("connectivity", 8)),
                merge_gap=float(cluster_config.get("merge_gap", 0.0)),
                merge_max_points=int(cluster_config.get("merge_max_points", 12000)),
                merge_vehicle_size=tuple(cluster_config.get("merge_vehicle_size", [12.0, 3.8, 4.0])),
            )
        else:
            labels = dbscan(
                prepared[:, :3],
                eps=float(cluster_config.get("eps", 0.75)),
                min_points=int(cluster_config.get("min_points", 8)),
            )
            clusters = extract_clusters(
                prepared,
                labels,
                max_cluster_points=int(cluster_config.get("max_cluster_points", 30000)),
            )

        detections = []
        for cluster in clusters:
            detection = self.classifier.classify(cluster)
            if detection is not None:
                detections.append(detection)

        return apply_output_policy(detections, self.config)

    def detect_file(self, path: str | Path, bin_dim: int | None = None) -> list[Detection]:
        points = load_point_cloud(path, bin_dim=bin_dim)
        return self.detect_array(points)
