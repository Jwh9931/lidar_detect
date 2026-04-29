"""Point cloud preprocessing."""

from __future__ import annotations

import numpy as np


def filter_finite(points: np.ndarray) -> np.ndarray:
    mask = np.isfinite(points[:, :3]).all(axis=1)
    return points[mask]


def filter_roi(points: np.ndarray, roi: dict) -> np.ndarray:
    x_min, x_max = roi.get("x", [-np.inf, np.inf])
    y_min, y_max = roi.get("y", [-np.inf, np.inf])
    z_min, z_max = roi.get("z", [-np.inf, np.inf])
    xyz = points[:, :3]
    mask = (
        (xyz[:, 0] >= x_min)
        & (xyz[:, 0] <= x_max)
        & (xyz[:, 1] >= y_min)
        & (xyz[:, 1] <= y_max)
        & (xyz[:, 2] >= z_min)
        & (xyz[:, 2] <= z_max)
    )
    return points[mask]


def voxel_downsample(points: np.ndarray, voxel_size: float, mode: str = "first") -> np.ndarray:
    if voxel_size <= 0 or len(points) == 0:
        return points

    coords = np.floor(points[:, :3] / float(voxel_size)).astype(np.int64)
    if mode == "first":
        normalized = coords - coords.min(axis=0)
        dims = normalized.max(axis=0) + 1
        if np.prod(dims.astype(np.float64)) < np.iinfo(np.int64).max:
            keys = normalized[:, 0] * dims[1] * dims[2] + normalized[:, 1] * dims[2] + normalized[:, 2]
            _, unique_indices = np.unique(keys, return_index=True)
            return points[np.sort(unique_indices)]

    _, inverse = np.unique(coords, axis=0, return_inverse=True)
    sums = np.zeros((inverse.max() + 1, points.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, points)
    counts = np.bincount(inverse).astype(np.float64)
    return (sums / counts[:, None]).astype(points.dtype, copy=False)


def _fit_plane_ransac(
    points_xyz: np.ndarray,
    iterations: int,
    distance_threshold: float,
    sample_size: int,
    seed: int = 7,
) -> tuple[np.ndarray, float, np.ndarray] | None:
    if len(points_xyz) < 3:
        return None

    rng = np.random.default_rng(seed)
    if len(points_xyz) > sample_size:
        sample_indices = rng.choice(len(points_xyz), size=sample_size, replace=False)
        sample = points_xyz[sample_indices]
    else:
        sample = points_xyz

    best_plane = None
    best_inliers = None
    best_count = 0
    for _ in range(max(1, int(iterations))):
        ids = rng.choice(len(sample), size=3, replace=False)
        p1, p2, p3 = sample[ids]
        normal = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(normal)
        if norm <= 1e-9:
            continue
        normal = normal / norm
        if normal[2] < 0:
            normal = -normal
        d = -float(np.dot(normal, p1))
        distances = np.abs(sample @ normal + d)
        inliers = distances <= distance_threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_plane = normal
            best_inliers = inliers
            best_d = d

    if best_plane is None or best_inliers is None:
        return None
    return best_plane, float(best_d), best_inliers


def remove_ground(points: np.ndarray, ground_config: dict) -> np.ndarray:
    if len(points) == 0:
        return points

    method = ground_config.get("method", "ransac")
    if method in {"none", "skip", None}:
        return points

    xyz = points[:, :3]
    distance_threshold = float(ground_config.get("distance_threshold", 0.18))

    if method == "ransac" and len(points) >= 20:
        result = _fit_plane_ransac(
            xyz,
            iterations=int(ground_config.get("iterations", 80)),
            distance_threshold=distance_threshold,
            sample_size=int(ground_config.get("sample_size", 8000)),
        )
        if result is not None:
            normal, d, _ = result
            min_normal_z = float(ground_config.get("min_ground_normal_z", 0.85))
            if abs(float(normal[2])) >= min_normal_z:
                distances = np.abs(xyz @ normal + d)
                return points[distances > distance_threshold]

    percentile = float(ground_config.get("fallback_percentile", 5.0))
    fallback_height = float(ground_config.get("fallback_height", 0.25))
    z_ref = float(np.percentile(xyz[:, 2], percentile))
    return points[xyz[:, 2] > z_ref + fallback_height]


def preprocess_points(points: np.ndarray, config: dict) -> np.ndarray:
    filtered = filter_finite(points)
    filtered = filter_roi(filtered, config.get("roi", {}))
    preprocess_config = config.get("preprocess", {})
    filtered = voxel_downsample(
        filtered,
        float(preprocess_config.get("voxel_size", 0.0)),
        mode=str(preprocess_config.get("voxel_mode", "first")),
    )
    return remove_ground(filtered, config.get("ground", {}))
