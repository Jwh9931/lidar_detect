"""Density-based clustering for object candidates."""

from __future__ import annotations

from collections import defaultdict, deque
from itertools import product

import numpy as np


def _build_cells(points_xyz: np.ndarray, cell_size: float) -> tuple[np.ndarray, dict[tuple[int, int, int], list[int]]]:
    cells = np.floor(points_xyz / cell_size).astype(np.int64)
    mapping: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, cell in enumerate(cells):
        mapping[tuple(int(v) for v in cell)].append(index)
    return cells, mapping


def dbscan(points_xyz: np.ndarray, eps: float = 0.75, min_points: int = 8) -> np.ndarray:
    """Small dependency-free DBSCAN implementation using a spatial hash grid."""
    points_xyz = np.asarray(points_xyz[:, :3], dtype=float)
    n_points = len(points_xyz)
    labels = np.full(n_points, -1, dtype=int)
    if n_points == 0:
        return labels

    eps = float(eps)
    eps_sq = eps * eps
    cells, mapping = _build_cells(points_xyz, eps)
    visited = np.zeros(n_points, dtype=bool)
    offsets = list(product((-1, 0, 1), repeat=3))

    def neighbors(index: int) -> list[int]:
        cell = cells[index]
        candidates: list[int] = []
        for offset in offsets:
            key = (int(cell[0] + offset[0]), int(cell[1] + offset[1]), int(cell[2] + offset[2]))
            candidates.extend(mapping.get(key, ()))
        if not candidates:
            return []
        candidate_array = np.asarray(candidates, dtype=int)
        deltas = points_xyz[candidate_array] - points_xyz[index]
        mask = np.einsum("ij,ij->i", deltas, deltas) <= eps_sq
        return candidate_array[mask].tolist()

    cluster_id = 0
    for index in range(n_points):
        if visited[index]:
            continue
        visited[index] = True
        index_neighbors = neighbors(index)
        if len(index_neighbors) < min_points:
            continue

        labels[index] = cluster_id
        queue = deque(index_neighbors)
        queued = set(index_neighbors)
        while queue:
            candidate = queue.popleft()
            if not visited[candidate]:
                visited[candidate] = True
                candidate_neighbors = neighbors(candidate)
                if len(candidate_neighbors) >= min_points:
                    for neighbor in candidate_neighbors:
                        if neighbor not in queued:
                            queue.append(neighbor)
                            queued.add(neighbor)
            if labels[candidate] == -1:
                labels[candidate] = cluster_id
        cluster_id += 1

    return labels


def extract_clusters(points: np.ndarray, labels: np.ndarray, max_cluster_points: int = 30000) -> list[np.ndarray]:
    clusters: list[np.ndarray] = []
    for label in sorted(int(v) for v in np.unique(labels) if v >= 0):
        cluster = points[labels == label]
        if 0 < len(cluster) <= max_cluster_points:
            clusters.append(cluster)
    return clusters


def grid_connected_components(
    points: np.ndarray,
    cell_size: float = 0.6,
    min_points: int = 8,
    max_cluster_points: int = 30000,
    connectivity: int = 8,
    merge_gap: float = 0.0,
    merge_max_points: int = 12000,
    merge_vehicle_size: tuple[float, float, float] = (12.0, 3.8, 4.0),
) -> list[np.ndarray]:
    """Fast BEV grid connected components clustering.

    This is less exact than DBSCAN, but much faster for realtime ROS use because
    it traverses occupied XY cells instead of every point neighborhood.
    """
    if len(points) == 0:
        return []

    xy_cells = np.floor(points[:, :2] / float(cell_size)).astype(np.int64)
    cell_to_indices: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, cell in enumerate(xy_cells):
        cell_to_indices[(int(cell[0]), int(cell[1]))].append(index)

    if connectivity == 4:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        offsets = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    visited: set[tuple[int, int]] = set()
    clusters: list[np.ndarray] = []
    for start in cell_to_indices:
        if start in visited:
            continue
        queue = deque([start])
        visited.add(start)
        component_indices: list[int] = []
        while queue:
            cell = queue.popleft()
            component_indices.extend(cell_to_indices[cell])
            for dx, dy in offsets:
                neighbor = (cell[0] + dx, cell[1] + dy)
                if neighbor in cell_to_indices and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        if len(component_indices) < min_points or len(component_indices) > max_cluster_points:
            continue
        clusters.append(points[np.asarray(component_indices, dtype=int)])

    if merge_gap > 0:
        clusters = merge_nearby_vehicle_parts(
            clusters,
            max_gap=merge_gap,
            max_cluster_points=max_cluster_points,
            max_merge_points=merge_max_points,
            max_size=merge_vehicle_size,
        )
    return clusters


def _aabb(cluster: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz = cluster[:, :3]
    return xyz.min(axis=0), xyz.max(axis=0)


def _aabb_gap(a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray) -> float:
    gap_x = max(0.0, float(max(a_min[0], b_min[0]) - min(a_max[0], b_max[0])))
    gap_y = max(0.0, float(max(a_min[1], b_min[1]) - min(a_max[1], b_max[1])))
    return float(np.hypot(gap_x, gap_y))


def _merge_size_ok(min_xyz: np.ndarray, max_xyz: np.ndarray, max_size: tuple[float, float, float]) -> bool:
    extents = max_xyz - min_xyz
    xy = sorted([float(extents[0]), float(extents[1])], reverse=True)
    return xy[0] <= max_size[0] and xy[1] <= max_size[1] and float(extents[2]) <= max_size[2]


def merge_nearby_vehicle_parts(
    clusters: list[np.ndarray],
    max_gap: float,
    max_cluster_points: int,
    max_merge_points: int,
    max_size: tuple[float, float, float],
) -> list[np.ndarray]:
    """Merge nearby small components that can form one vehicle-sized object."""
    if len(clusters) <= 1:
        return clusters

    boxes = [_aabb(cluster) for cluster in clusters]
    parent = list(range(len(clusters)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for i in range(len(clusters)):
        if len(clusters[i]) > max_merge_points:
            continue
        i_min, i_max = boxes[i]
        for j in range(i + 1, len(clusters)):
            if len(clusters[j]) > max_merge_points:
                continue
            j_min, j_max = boxes[j]
            if _aabb_gap(i_min, i_max, j_min, j_max) > max_gap:
                continue
            merged_min = np.minimum(i_min, j_min)
            merged_max = np.maximum(i_max, j_max)
            if _merge_size_ok(merged_min, merged_max, max_size):
                union(i, j)

    groups: dict[int, list[np.ndarray]] = defaultdict(list)
    for index, cluster in enumerate(clusters):
        groups[find(index)].append(cluster)

    merged: list[np.ndarray] = []
    for items in groups.values():
        if len(items) == 1:
            merged.append(items[0])
            continue
        cluster = np.vstack(items)
        if len(cluster) <= max_cluster_points:
            merged.append(cluster)
        else:
            merged.extend(items)
    return merged
