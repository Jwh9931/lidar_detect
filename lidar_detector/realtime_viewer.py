"""Realtime ROS PointCloud2 detection viewer."""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass

import numpy as np

from .config import load_config
from .geometry import Detection
from .io import pointcloud2_to_array
from .pipeline import LidarObjectDetector


MODE_BOTH = "pointcloud+detections"
MODE_POINTS = "pointcloud only"
MODE_DETECTIONS = "detections only"


@dataclass
class FrameState:
    points: np.ndarray | None = None
    detections: list[Detection] | None = None
    stamp: float | None = None
    frame_id: str = ""
    seq: int = 0
    error: str = ""
    updated_at: float = 0.0


class LatestFrameBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._points: np.ndarray | None = None
        self._stamp: float | None = None
        self._frame_id = ""
        self._seq = 0
        self._error = ""
        self._version = 0

    def update_msg(self, msg) -> None:
        try:
            points = pointcloud2_to_array(msg)
            header = getattr(msg, "header", None)
            stamp = getattr(header, "stamp", None) if header is not None else None
            stamp_sec = float(stamp.to_sec()) if hasattr(stamp, "to_sec") else None
            frame_id = getattr(header, "frame_id", "") if header is not None else ""
            seq = int(getattr(header, "seq", 0)) if header is not None else 0
            error = ""
        except Exception as exc:  # pragma: no cover - ROS runtime path
            points = None
            stamp_sec = None
            frame_id = ""
            seq = 0
            error = str(exc)

        with self._lock:
            self._points = points
            self._stamp = stamp_sec
            self._frame_id = frame_id
            self._seq = seq
            self._error = error
            self._version += 1

    def snapshot(self) -> tuple[int, FrameState]:
        with self._lock:
            return self._version, FrameState(
                points=self._points,
                stamp=self._stamp,
                frame_id=self._frame_id,
                seq=self._seq,
                error=self._error,
                updated_at=time.time(),
            )


def _parse_centerline(value: str) -> list[list[float]]:
    points = []
    for item in value.split(";"):
        coords = [float(v.strip()) for v in item.split(",")]
        if len(coords) != 2:
            raise argparse.ArgumentTypeError("Centerline points must be formatted as x,y;x,y;...")
        points.append(coords)
    if len(points) < 2:
        raise argparse.ArgumentTypeError("A rail centerline needs at least two points.")
    return points


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Realtime BEV viewer for ROS PointCloud2 detections.")
    parser.add_argument("--topic", default="/ground_segmentation/nonground", help="ROS PointCloud2 topic")
    parser.add_argument("--config", default="config/default.yaml", help="YAML config path")
    parser.add_argument("--skip-ground-removal", action="store_true", help="Use this when input is already non-ground point cloud")
    parser.add_argument("--rail-centerline", type=_parse_centerline, default=None, help="Override rail centerline, e.g. '-50,0;50,0'")
    parser.add_argument("--rail-width", type=float, default=None, help="Override rail corridor width in meters")
    parser.add_argument("--max-display-points", type=int, default=20000, help="Maximum points drawn per frame")
    parser.add_argument("--interval-ms", type=int, default=120, help="Viewer refresh interval in milliseconds")
    parser.add_argument("--initial-mode", choices=(MODE_BOTH, MODE_POINTS, MODE_DETECTIONS), default=MODE_BOTH)
    return parser


def _make_override(args: argparse.Namespace) -> dict:
    override: dict = {}
    if args.rail_centerline is not None:
        override.setdefault("rail", {})["centerlines"] = [args.rail_centerline]
    if args.rail_width is not None:
        override.setdefault("rail", {})["corridor_width"] = args.rail_width
    if args.skip_ground_removal:
        override.setdefault("ground", {})["method"] = "none"
    return override


def _draw_rail(ax, config: dict) -> None:
    rail = config.get("rail", {})
    for centerline in rail.get("centerlines", []):
        line = np.asarray(centerline, dtype=float)
        if len(line) >= 2:
            ax.plot(line[:, 0], line[:, 1], color="#d4a72c", linewidth=1.6, label="rail")


def _draw_points(ax, points: np.ndarray, max_display_points: int) -> None:
    if len(points) == 0:
        return
    xyz = points[:, :3]
    if len(xyz) > max_display_points:
        step = max(1, len(xyz) // max_display_points)
        xyz = xyz[::step]
    color = xyz[:, 2]
    ax.scatter(xyz[:, 0], xyz[:, 1], c=color, s=1.0, cmap="viridis", alpha=0.8, linewidths=0)


def _draw_detection(ax, detection: Detection) -> None:
    color_map = {
        "vehicle": "#28a745",
        "pedestrian": "#1f77b4",
        "unknown_rail_obstacle": "#d62728",
    }
    color = color_map.get(detection.label, "#eeeeee")
    center = np.asarray(detection.bbox.center[:2], dtype=float)
    length, width, _ = detection.bbox.size
    yaw = detection.bbox.yaw
    corners = np.array(
        [
            [length / 2.0, width / 2.0],
            [length / 2.0, -width / 2.0],
            [-length / 2.0, -width / 2.0],
            [-length / 2.0, width / 2.0],
            [length / 2.0, width / 2.0],
        ],
        dtype=float,
    )
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]], dtype=float)
    rotated = corners @ rot.T + center
    ax.plot(rotated[:, 0], rotated[:, 1], color=color, linewidth=2.0)
    heading = np.array([[0.0, 0.0], [length / 2.0, 0.0]]) @ rot.T + center
    ax.plot(heading[:, 0], heading[:, 1], color=color, linewidth=2.0)
    ax.text(center[0], center[1], f"{detection.label} {detection.confidence:.2f}", color=color, fontsize=8)


def run_viewer(args: argparse.Namespace) -> int:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.widgets import RadioButtons
    except ImportError as exc:
        raise RuntimeError("Realtime viewer requires matplotlib. Install it with: sudo apt install python3-matplotlib") from exc

    try:
        import rospy
        from sensor_msgs.msg import PointCloud2
    except ImportError as exc:
        raise RuntimeError("Realtime viewer requires ROS1. Run: source /opt/ros/noetic/setup.bash") from exc

    config = load_config(args.config, override=_make_override(args))
    detector = LidarObjectDetector(config)
    buffer = LatestFrameBuffer()
    state = FrameState()
    processed_version = -1
    mode = {"value": args.initial_mode}

    rospy.init_node("lidar_detector_realtime_viewer", anonymous=True, disable_signals=True)
    rospy.Subscriber(args.topic, PointCloud2, buffer.update_msg, queue_size=1, buff_size=2**24)

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.canvas.manager.set_window_title("LiDAR Realtime Detection Viewer")
    plt.subplots_adjust(left=0.06, right=0.78, bottom=0.08, top=0.92)
    radio_ax = plt.axes([0.8, 0.55, 0.18, 0.22])
    radio = RadioButtons(radio_ax, (MODE_BOTH, MODE_POINTS, MODE_DETECTIONS), active=(MODE_BOTH, MODE_POINTS, MODE_DETECTIONS).index(args.initial_mode))

    def on_mode(label: str) -> None:
        mode["value"] = label

    radio.on_clicked(on_mode)

    roi = config.get("roi", {})
    x_limits = roi.get("x", [-80.0, 80.0])
    y_limits = roi.get("y", [-40.0, 40.0])

    def update(_frame_index: int):
        nonlocal state, processed_version
        version, incoming = buffer.snapshot()
        if version != processed_version:
            processed_version = version
            state = incoming
            if state.points is not None and not state.error:
                state.detections = detector.detect_array(state.points)

        ax.clear()
        ax.set_facecolor("#101820")
        fig.patch.set_facecolor("#101820")
        ax.set_xlim(float(x_limits[0]), float(x_limits[1]))
        ax.set_ylim(float(y_limits[0]), float(y_limits[1]))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="#2c3a47", linewidth=0.6)
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
        ax.tick_params(colors="#dce3ea")
        ax.xaxis.label.set_color("#dce3ea")
        ax.yaxis.label.set_color("#dce3ea")
        for spine in ax.spines.values():
            spine.set_color("#566573")

        _draw_rail(ax, config)
        if state.error:
            title = f"{args.topic} | error: {state.error}"
        elif state.points is None:
            title = f"{args.topic} | waiting for PointCloud2..."
        else:
            detections = state.detections or []
            if mode["value"] in {MODE_BOTH, MODE_POINTS}:
                _draw_points(ax, state.points, args.max_display_points)
            if mode["value"] in {MODE_BOTH, MODE_DETECTIONS}:
                for detection in detections:
                    _draw_detection(ax, detection)
            stamp = "" if state.stamp is None else f" stamp={state.stamp:.3f}"
            title = (
                f"{args.topic} | mode={mode['value']} | points={len(state.points)} "
                f"| detections={len(detections)} | frame={state.frame_id}{stamp}"
            )
        ax.set_title(title, color="#f2f5f7")
        return []

    FuncAnimation(fig, update, interval=max(30, int(args.interval_ms)), cache_frame_data=False)
    print(f"Subscribed to {args.topic}. Close the viewer window to exit.")
    plt.show()
    rospy.signal_shutdown("viewer closed")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_viewer(build_parser().parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
