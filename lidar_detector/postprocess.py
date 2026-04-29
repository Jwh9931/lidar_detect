"""Detection output filtering policies."""

from __future__ import annotations

from collections import defaultdict
from math import hypot

from .geometry import Detection


def _detection_range(detection: Detection, metric: str) -> float:
    x, y, _ = detection.bbox.center
    if metric == "euclidean_xy":
        return hypot(x, y)
    if metric == "abs_forward_x":
        return abs(x)
    return x


def _passes_min_size(detection: Detection, min_size: list[float] | None) -> bool:
    if not min_size:
        return True
    if len(min_size) != 3:
        raise ValueError("min_size must contain three values: length, width, height.")
    length, width, height = detection.bbox.size
    sorted_xy = sorted([length, width], reverse=True)
    required_xy = sorted([float(min_size[0]), float(min_size[1])], reverse=True)
    return sorted_xy[0] >= required_xy[0] and sorted_xy[1] >= required_xy[1] and height >= float(min_size[2])


def apply_output_policy(detections: list[Detection], config: dict) -> list[Detection]:
    policy = config.get("output_policy", {})
    per_label = policy.get("per_label", {})
    metric = policy.get("range_metric", "forward_x")
    sort_by = policy.get("sort_by", "confidence")

    grouped: dict[str, list[Detection]] = defaultdict(list)
    for detection in detections:
        label_policy = per_label.get(detection.label, {})
        distance = _detection_range(detection, metric)
        min_range = float(label_policy.get("min_range", float("-inf")))
        max_range = float(label_policy.get("max_range", float("inf")))
        if distance < min_range or distance > max_range:
            continue
        if not _passes_min_size(detection, label_policy.get("min_size")):
            continue
        grouped[detection.label].append(detection)

    output: list[Detection] = []
    for label, items in grouped.items():
        label_policy = per_label.get(label, {})
        if sort_by == "range":
            items.sort(key=lambda item: _detection_range(item, metric))
        else:
            items.sort(key=lambda item: item.confidence, reverse=True)
        max_outputs = int(label_policy.get("max_outputs", len(items)))
        output.extend(items[:max_outputs])

    output.sort(key=lambda item: (item.label, _detection_range(item, metric), -item.confidence))
    return output
