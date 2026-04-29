"""Configuration helpers."""

from __future__ import annotations

from copy import deepcopy
import ast
import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_CONFIG: dict[str, Any] = {
    "roi": {
        "x": [-20.0, 300.0],
        "y": [-30.0, 30.0],
        "z": [-3.0, 6.0],
    },
    "preprocess": {
        "voxel_size": 0.15,
        "voxel_mode": "first",
    },
    "ground": {
        "method": "ransac",
        "distance_threshold": 0.18,
        "iterations": 80,
        "sample_size": 8000,
        "min_ground_normal_z": 0.85,
        "fallback_percentile": 5.0,
        "fallback_height": 0.25,
    },
    "clustering": {
        "method": "grid",
        "grid_cell_size": 0.45,
        "connectivity": 4,
        "merge_gap": 0.35,
        "merge_max_points": 12000,
        "merge_vehicle_size": [12.0, 3.8, 4.0],
        "eps": 0.75,
        "min_points": 8,
        "max_cluster_points": 30000,
    },
    "rail": {
        "centerlines": [[[0.0, 0.0], [300.0, 0.0]]],
        "corridor_width": 6.0,
        "min_overlap": 0.2,
    },
    "output_policy": {
        "range_metric": "forward_x",
        "sort_by": "confidence",
        "per_label": {
            "vehicle": {
                "min_range": 0.0,
                "max_range": 120.0,
                "max_outputs": 30,
            },
            "pedestrian": {
                "min_range": 0.0,
                "max_range": 120.0,
                "max_outputs": 15,
            },
            "unknown_rail_obstacle": {
                "min_range": 5.0,
                "max_range": 60.0,
                "max_outputs": 20,
                "min_size": [0.3, 0.3, 0.3],
            },
            "large_rail_obstacle": {
                "min_range": 0.0,
                "max_range": 300.0,
                "max_outputs": 15,
            },
        },
    },
    "classification": {
        "min_cluster_points": 10,
        "min_height": 0.15,
        "confidence_threshold": 0.42,
        "static_rejection": {
            "enabled": True,
            "wall_like": {
                "min_length": 5.0,
                "max_width": 1.4,
                "min_height": 1.6,
                "min_aspect_ratio": 4.0,
            },
            "building_like": {
                "max_footprint_area": 26.0,
                "min_height": 2.2,
            },
            "thin_vertical_like": {
                "min_length": 1.8,
                "max_width": 1.0,
                "min_height": 0.8,
                "min_height_width_ratio": 1.45,
                "min_aspect_ratio": 2.3,
            },
            "line_like": {
                "min_length": 2.0,
                "min_aspect_ratio": 3.0,
                "max_fill_ratio": 0.32,
            },
        },
        "shape_features": {
            "cell_size": 0.25,
        },
        "vehicle": {
            "length": [1.2, 12.0],
            "width": [0.6, 3.5],
            "height": [0.35, 4.0],
            "length_width_ratio": [0.9, 5.5],
            "min_footprint_area": 1.4,
            "max_footprint_area": 28.0,
            "min_points": 12,
        },
        "pedestrian": {
            "height": [0.8, 2.4],
            "footprint": [0.2, 1.2],
            "min_width": 0.12,
            "min_points": 8,
        },
        "large_rail_obstacle": {
            "enabled": True,
            "min_size": [2.0, 1.0, 1.0],
        },
    },
}


def deep_update(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge override values into base."""
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return {}
    if value.startswith("[") or value.startswith("{"):
        return ast.literal_eval(value)
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        if any(char in value for char in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def _simple_yaml_load(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by the default config.

    This fallback keeps the CLI usable when PyYAML is not installed. It supports
    nested mappings, list items, comments, and inline Python/YAML-style lists.
    """
    lines: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        lines.append((len(line) - len(line.lstrip(" ")), line.strip()))

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        if index >= len(lines):
            return {}, index

        if lines[index][1].startswith("- "):
            items = []
            while index < len(lines) and lines[index][0] == indent and lines[index][1].startswith("- "):
                item = lines[index][1][2:].strip()
                if item:
                    items.append(_parse_scalar(item))
                    index += 1
                else:
                    child, index = parse_block(index + 1, lines[index + 1][0])
                    items.append(child)
            return items, index

        mapping: dict[str, Any] = {}
        while index < len(lines) and lines[index][0] == indent and not lines[index][1].startswith("- "):
            text_line = lines[index][1]
            if ":" not in text_line:
                raise ValueError(f"Invalid config line: {text_line}")
            key, raw_value = text_line.split(":", 1)
            key = key.strip()
            raw_value = raw_value.strip()
            index += 1
            if raw_value:
                mapping[key] = _parse_scalar(raw_value)
            elif index < len(lines) and lines[index][0] > indent:
                child, index = parse_block(index, lines[index][0])
                mapping[key] = child
            else:
                mapping[key] = {}
        return mapping, index

    parsed, index = parse_block(0, lines[0][0] if lines else 0)
    if index != len(lines) or not isinstance(parsed, dict):
        raise ValueError("Config file must contain a YAML mapping.")
    return parsed


def load_config(path: str | Path | None = None, override: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if path:
        config_path = Path(path)
        text = config_path.read_text(encoding="utf-8")
        if config_path.suffix.lower() == ".json":
            loaded = json.loads(text)
        else:
            try:
                import yaml

                loaded = yaml.safe_load(text) or {}
            except ImportError:
                loaded = _simple_yaml_load(text)
        if not isinstance(loaded, Mapping):
            raise ValueError(f"Config file must contain a mapping: {path}")
        deep_update(config, loaded)

    if override:
        deep_update(config, override)
    return config
