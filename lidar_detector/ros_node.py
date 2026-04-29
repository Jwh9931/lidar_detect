"""ROS1 realtime detection node.

Subscribes to a PointCloud2 topic and publishes:
- a passthrough PointCloud2 topic for visualization
- a MarkerArray topic with 3D boxes and labels
- a JSON String topic for downstream consumers
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time

from .config import load_config
from .io import pointcloud2_to_array
from .pipeline import LidarObjectDetector


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
    parser = argparse.ArgumentParser(description="ROS1 node that publishes LiDAR detection results as topics.")
    parser.add_argument("--input-topic", default="/ground_segmentation/nonground", help="Input PointCloud2 topic")
    parser.add_argument("--points-topic", default="/lidar_detector/points", help="Published passthrough PointCloud2 topic")
    parser.add_argument("--markers-topic", default="/lidar_detector/detection_markers", help="Published MarkerArray topic")
    parser.add_argument("--json-topic", default="/lidar_detector/detections_json", help="Published std_msgs/String JSON topic")
    parser.add_argument("--config", default="config/default.yaml", help="YAML config path")
    parser.add_argument("--skip-ground-removal", action="store_true", help="Use this when input is already non-ground point cloud")
    parser.add_argument("--rail-centerline", type=_parse_centerline, default=None, help="Override rail centerline, e.g. '-50,0;50,0'")
    parser.add_argument("--rail-width", type=float, default=None, help="Override rail corridor width in meters")
    parser.add_argument("--queue-size", type=int, default=1, help="ROS subscriber queue size")
    parser.add_argument("--max-processing-points", type=int, default=60000, help="Stride-sample input above this point count before detection; use 0 to disable")
    parser.add_argument("--process-every-n", type=int, default=1, help="Only run detection for every Nth incoming frame")
    parser.add_argument("--no-points-topic", action="store_true", help="Do not republish passthrough PointCloud2")
    parser.add_argument("--no-label-markers", action="store_true", help="Publish only detection boxes, without text labels")
    parser.add_argument("--compact-json", action="store_true", help="Publish summary JSON without full detection list")
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


def _color_for_label(label: str) -> tuple[float, float, float, float]:
    if label == "vehicle":
        return 0.1, 0.85, 0.25, 0.55
    if label == "pedestrian":
        return 0.1, 0.45, 1.0, 0.65
    if label == "unknown_rail_obstacle":
        return 1.0, 0.08, 0.08, 0.75
    return 0.9, 0.9, 0.9, 0.55


def _set_marker_color(marker, color: tuple[float, float, float, float]) -> None:
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]


def _yaw_to_quaternion(marker, yaw: float) -> None:
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = math.sin(yaw / 2.0)
    marker.pose.orientation.w = math.cos(yaw / 2.0)


def detections_to_marker_array(detections, header, include_labels: bool = True):
    from visualization_msgs.msg import Marker, MarkerArray

    marker_array = MarkerArray()

    clear = Marker()
    clear.header = header
    clear.action = Marker.DELETEALL
    marker_array.markers.append(clear)

    marker_id = 1
    for detection in detections:
        bbox = detection.bbox
        color = _color_for_label(detection.label)

        box = Marker()
        box.header = header
        box.ns = "lidar_detector_boxes"
        box.id = marker_id
        marker_id += 1
        box.type = Marker.CUBE
        box.action = Marker.ADD
        box.pose.position.x = bbox.center[0]
        box.pose.position.y = bbox.center[1]
        box.pose.position.z = bbox.center[2]
        _yaw_to_quaternion(box, bbox.yaw)
        box.scale.x = max(bbox.length, 0.05)
        box.scale.y = max(bbox.width, 0.05)
        box.scale.z = max(bbox.height, 0.05)
        _set_marker_color(box, color)
        box.lifetime.from_sec(0.25)
        marker_array.markers.append(box)

        if not include_labels:
            continue

        label = Marker()
        label.header = header
        label.ns = "lidar_detector_labels"
        label.id = marker_id
        marker_id += 1
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x = bbox.center[0]
        label.pose.position.y = bbox.center[1]
        label.pose.position.z = bbox.center[2] + max(bbox.height / 2.0 + 0.4, 0.6)
        label.pose.orientation.w = 1.0
        label.scale.z = 0.45
        label.text = f"{detection.label} {detection.confidence:.2f}"
        _set_marker_color(label, (color[0], color[1], color[2], 1.0))
        label.lifetime.from_sec(0.25)
        marker_array.markers.append(label)

    return marker_array


def _limit_points(points, max_points: int):
    if max_points <= 0 or len(points) <= max_points:
        return points
    step = max(1, int(math.ceil(len(points) / max_points)))
    return points[::step]


class LidarDetectionNode:
    def __init__(self, args: argparse.Namespace) -> None:
        import rospy
        from sensor_msgs.msg import PointCloud2
        from std_msgs.msg import String
        from visualization_msgs.msg import MarkerArray

        config = load_config(args.config, override=_make_override(args))
        self.detector = LidarObjectDetector(config)
        self.args = args
        self.points_pub = rospy.Publisher(args.points_topic, PointCloud2, queue_size=1)
        self.markers_pub = rospy.Publisher(args.markers_topic, MarkerArray, queue_size=1)
        self.json_pub = rospy.Publisher(args.json_topic, String, queue_size=1)
        self.last_log_time = 0.0
        self.frame_count = 0
        self.processed_count = 0
        self.lock = threading.Lock()
        self.latest_msg = None
        self.latest_seq = 0
        self.worker = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker.start()
        rospy.Subscriber(args.input_topic, PointCloud2, self.callback, queue_size=args.queue_size, buff_size=2**24)

    def callback(self, msg) -> None:
        self.frame_count += 1
        if not self.args.no_points_topic:
            self.points_pub.publish(msg)
        if self.args.process_every_n > 1 and self.frame_count % self.args.process_every_n != 0:
            return

        with self.lock:
            self.latest_msg = msg
            self.latest_seq += 1

    def _take_latest_msg(self):
        with self.lock:
            return self.latest_seq, self.latest_msg

    def worker_loop(self) -> None:
        import rospy
        from std_msgs.msg import String

        last_seq = -1
        rate = rospy.Rate(200)
        while not rospy.is_shutdown():
            seq, msg = self._take_latest_msg()
            if msg is None or seq == last_seq:
                rate.sleep()
                continue
            last_seq = seq
            start = time.time()

            try:
                points = pointcloud2_to_array(msg)
                processing_points = _limit_points(points, self.args.max_processing_points)
                detections = self.detector.detect_array(processing_points)
            except Exception as exc:
                rospy.logerr_throttle(1.0, "lidar_detector failed: %s", exc)
                continue

            processing_ms = (time.time() - start) * 1000.0
            self.processed_count += 1
            self.markers_pub.publish(detections_to_marker_array(detections, msg.header, include_labels=not self.args.no_label_markers))
            payload = {
                "stamp": msg.header.stamp.to_sec() if hasattr(msg.header.stamp, "to_sec") else None,
                "frame_id": msg.header.frame_id,
                "num_points": int(len(points)),
                "num_processing_points": int(len(processing_points)),
                "num_detections": len(detections),
                "processing_ms": round(processing_ms, 3),
            }
            if not self.args.compact_json:
                payload["detections"] = [detection.to_dict() for detection in detections]
            self.json_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

            now = time.time()
            if now - self.last_log_time > 1.0:
                rospy.loginfo(
                    "lidar_detector: input=%d processed=%d points=%d used=%d detections=%d %.1fms",
                    self.frame_count,
                    self.processed_count,
                    len(points),
                    len(processing_points),
                    len(detections),
                    processing_ms,
                )
                self.last_log_time = now


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import rospy
    except ImportError as exc:
        raise RuntimeError("ROS1 is required. Run: source /opt/ros/noetic/setup.bash") from exc

    rospy.init_node("lidar_detector_node", anonymous=False)
    LidarDetectionNode(args)
    rospy.loginfo("lidar_detector subscribed to %s", args.input_topic)
    rospy.loginfo("publishing points: %s", args.points_topic)
    rospy.loginfo("publishing markers: %s", args.markers_topic)
    rospy.loginfo("publishing json: %s", args.json_topic)
    rospy.spin()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
