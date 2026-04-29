"""Point cloud IO utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .geometry import Detection


def _ensure_points(array: np.ndarray, source: str) -> np.ndarray:
    points = np.asarray(array, dtype=np.float32)
    if points.ndim == 1:
        if points.size % 4 == 0:
            points = points.reshape(-1, 4)
        elif points.size % 3 == 0:
            points = points.reshape(-1, 3)
        else:
            raise ValueError(f"Cannot infer point dimension for {source}.")
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Point cloud must have shape N x 3 or N x 4+: {source}")
    return points


def _load_bin(path: Path, dim: int | None = None) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if dim:
        if raw.size % dim != 0:
            raise ValueError(f"{path} does not contain a whole number of {dim}D points.")
        return raw.reshape(-1, dim)
    if raw.size % 4 == 0:
        return raw.reshape(-1, 4)
    if raw.size % 3 == 0:
        return raw.reshape(-1, 3)
    raise ValueError(f"Cannot infer .bin point dimension for {path}; pass --bin-dim.")


def _load_ascii_pcd(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        lines = handle.readlines()

    fields: list[str] = []
    data_index = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("FIELDS"):
            fields = stripped.split()[1:]
        if upper.startswith("DATA"):
            if "ascii" not in stripped.lower():
                raise RuntimeError("Binary PCD requires open3d; ASCII fallback cannot read it.")
            data_index = idx + 1
            break

    if data_index is None:
        raise ValueError(f"Invalid PCD file: {path}")

    data = np.loadtxt(lines[data_index:], dtype=np.float32)
    data = _ensure_points(data, str(path))
    if fields:
        indices = [fields.index(name) for name in ("x", "y", "z") if name in fields]
        if len(indices) == 3:
            extra = [i for i in range(data.shape[1]) if i not in indices]
            ordered = indices + extra
            data = data[:, ordered]
    return data


def _load_with_open3d(path: Path) -> np.ndarray:
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError(f"Reading {path.suffix} files requires open3d.") from exc

    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=np.float32)
    return _ensure_points(points, str(path))


_POINT_FIELD_DTYPES = {
    1: np.int8,
    2: np.uint8,
    3: np.int16,
    4: np.uint16,
    5: np.int32,
    6: np.uint32,
    7: np.float32,
    8: np.float64,
}


def pointcloud2_to_array(msg) -> np.ndarray:
    """Convert a ROS sensor_msgs/PointCloud2 message to N x 3/4 float32."""
    n_points = int(msg.width) * int(msg.height)
    field_map = {field.name: field for field in msg.fields}
    if not msg.is_bigendian and all(name in field_map for name in ("x", "y", "z")):
        x_field = field_map["x"]
        y_field = field_map["y"]
        z_field = field_map["z"]
        intensity_field = field_map.get("intensity")
        xyz_is_float32 = (
            int(x_field.datatype) == 7
            and int(y_field.datatype) == 7
            and int(z_field.datatype) == 7
            and int(getattr(x_field, "count", 1) or 1) == 1
            and int(getattr(y_field, "count", 1) or 1) == 1
            and int(getattr(z_field, "count", 1) or 1) == 1
            and int(x_field.offset) == 0
            and int(y_field.offset) == 4
            and int(z_field.offset) == 8
        )
        xyzi_is_contiguous = (
            intensity_field is not None
            and int(intensity_field.datatype) == 7
            and int(getattr(intensity_field, "count", 1) or 1) == 1
            and int(intensity_field.offset) == 12
            and int(msg.point_step) == 16
        )
        xyz_is_contiguous = int(msg.point_step) == 12
        if xyz_is_float32 and (xyzi_is_contiguous or xyz_is_contiguous):
            dim = 4 if xyzi_is_contiguous else 3
            return _ensure_points(np.frombuffer(msg.data, dtype=np.float32, count=n_points * dim).reshape(n_points, dim), "PointCloud2")

    dtype_fields = []
    for field in msg.fields:
        dtype = _POINT_FIELD_DTYPES.get(int(field.datatype))
        if dtype is None:
            continue
        endian_dtype = np.dtype(dtype).newbyteorder(">" if msg.is_bigendian else "<")
        count = int(getattr(field, "count", 1) or 1)
        dtype_fields.append((field.name, endian_dtype, (count,), int(field.offset)))

    if not dtype_fields:
        raise ValueError("PointCloud2 message has no supported fields.")

    dtype = np.dtype(
        {
            "names": [item[0] for item in dtype_fields],
            "formats": [item[1] if item[2] == (1,) else (item[1], item[2]) for item in dtype_fields],
            "offsets": [item[3] for item in dtype_fields],
            "itemsize": int(msg.point_step),
        }
    )
    cloud = np.frombuffer(msg.data, dtype=dtype, count=n_points)
    field_names = set(cloud.dtype.names or [])
    for required in ("x", "y", "z"):
        if required not in field_names:
            raise ValueError(f"PointCloud2 message is missing '{required}' field.")

    columns = [np.asarray(cloud[name], dtype=np.float32).reshape(-1) for name in ("x", "y", "z")]
    if "intensity" in field_names:
        columns.append(np.asarray(cloud["intensity"], dtype=np.float32).reshape(-1))
    return _ensure_points(np.column_stack(columns), "PointCloud2")


def load_bag_pointclouds(
    path: str | Path,
    topic: str,
    max_frames: int | None = None,
    frame_index: int | None = None,
) -> Iterator[tuple[dict, np.ndarray]]:
    """Yield point clouds from a ROS1 bag PointCloud2 topic."""
    try:
        import rosbag
    except ImportError as exc:  # pragma: no cover - requires ROS runtime
        raise RuntimeError(
            "Reading .bag files requires ROS1 Python packages. On Jetson, run "
            "'source /opt/ros/noetic/setup.bash' before this command."
        ) from exc

    emitted = 0
    with rosbag.Bag(str(path), "r") as bag:
        for index, (_, msg, stamp) in enumerate(bag.read_messages(topics=[topic])):
            if frame_index is not None and index != frame_index:
                continue
            header = getattr(msg, "header", None)
            metadata = {
                "topic": topic,
                "frame_index": index,
                "stamp": float(stamp.to_sec()) if hasattr(stamp, "to_sec") else None,
                "frame_id": getattr(header, "frame_id", "") if header is not None else "",
                "seq": int(getattr(header, "seq", index)) if header is not None else index,
            }
            yield metadata, pointcloud2_to_array(msg)
            emitted += 1
            if frame_index is not None or (max_frames is not None and emitted >= max_frames):
                break


def load_point_cloud(path: str | Path, bin_dim: int | None = None) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".bin":
        points = _load_bin(path, bin_dim)
    elif suffix == ".npy":
        points = np.load(path)
    elif suffix == ".csv":
        points = np.loadtxt(path, delimiter=",", dtype=np.float32)
    elif suffix == ".txt":
        points = np.loadtxt(path, dtype=np.float32)
    elif suffix == ".pcd":
        try:
            points = _load_ascii_pcd(path)
        except Exception:
            points = _load_with_open3d(path)
    elif suffix == ".ply":
        points = _load_with_open3d(path)
    else:
        raise ValueError(f"Unsupported point cloud format: {path.suffix}")

    return _ensure_points(points, str(path))


def detections_payload(source: str, detections: Iterable[Detection]) -> dict:
    items = [detection.to_dict() for detection in detections]
    return {
        "source": source,
        "num_detections": len(items),
        "detections": items,
    }


def bag_detections_payload(source: str, frames: Iterable[dict]) -> dict:
    items = list(frames)
    return {
        "source": source,
        "num_frames": len(items),
        "frames": items,
    }


def save_detections_json(path: str | Path, payload: dict) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
