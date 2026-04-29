import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lidar_detector.config import load_config
from lidar_detector.io import pointcloud2_to_array
from lidar_detector.pipeline import LidarObjectDetector
from lidar_detector.realtime_viewer import MODE_BOTH, build_parser as build_realtime_parser
from lidar_detector.ros_node import build_parser as build_ros_node_parser
from lidar_detector.visualize import main as visualize_main


def make_box(rng, center, size, count):
    center = np.asarray(center, dtype=np.float32)
    size = np.asarray(size, dtype=np.float32)
    return center + (rng.random((count, 3), dtype=np.float32) - 0.5) * size


class PipelineTest(unittest.TestCase):
    def test_loads_default_yaml_without_external_yaml_dependency(self):
        config = load_config("config/default.yaml")

        self.assertIn("roi", config)
        self.assertEqual(config["rail"]["centerlines"][0][0], [0.0, 0.0])
        self.assertEqual(config["output_policy"]["per_label"]["vehicle"]["max_range"], 120.0)

    def test_converts_pointcloud2_message(self):
        points = np.array([(1.0, 2.0, 3.0, 0.5), (4.0, 5.0, 6.0, 0.8)], dtype=np.float32)
        fields = [
            SimpleNamespace(name="x", datatype=7, count=1, offset=0),
            SimpleNamespace(name="y", datatype=7, count=1, offset=4),
            SimpleNamespace(name="z", datatype=7, count=1, offset=8),
            SimpleNamespace(name="intensity", datatype=7, count=1, offset=12),
        ]
        msg = SimpleNamespace(
            fields=fields,
            is_bigendian=False,
            point_step=16,
            width=2,
            height=1,
            data=points.tobytes(),
        )

        converted = pointcloud2_to_array(msg)

        np.testing.assert_allclose(converted, points)

    def test_detects_vehicle_pedestrian_and_unknown_rail_obstacle(self):
        rng = np.random.default_rng(42)
        ground = np.column_stack(
            [
                rng.uniform(-30, 30, 1200),
                rng.uniform(-10, 10, 1200),
                rng.normal(0.0, 0.01, 1200),
            ]
        ).astype(np.float32)
        vehicle = make_box(rng, center=[20.0, 5.0, 0.9], size=[4.6, 2.0, 1.8], count=450)
        pedestrian = make_box(rng, center=[10.0, -4.0, 0.9], size=[0.6, 0.5, 1.8], count=180)
        rail_obstacle = make_box(rng, center=[15.0, 0.25, 0.25], size=[0.8, 0.8, 0.5], count=160)
        points = np.vstack([ground, vehicle, pedestrian, rail_obstacle]).astype(np.float32)

        detector = LidarObjectDetector(
            {
                "preprocess": {"voxel_size": 0.05},
                "ground": {"distance_threshold": 0.08, "iterations": 100},
                "clustering": {"method": "dbscan", "eps": 0.9, "min_points": 10},
                "rail": {
                    "centerlines": [[[-40.0, 0.0], [40.0, 0.0]]],
                    "corridor_width": 3.0,
                    "min_overlap": 0.2,
                },
                "output_policy": {
                    "per_label": {
                        "vehicle": {"min_range": 0.0, "max_range": 80.0, "max_outputs": 15},
                        "pedestrian": {"min_range": 0.0, "max_range": 120.0, "max_outputs": 15},
                        "unknown_rail_obstacle": {
                            "min_range": 0.0,
                            "max_range": 60.0,
                            "max_outputs": 20,
                            "min_size": [0.3, 0.3, 0.3],
                        },
                        "large_rail_obstacle": {"min_range": 0.0, "max_range": 300.0, "max_outputs": 15},
                    }
                },
            }
        )
        detections = detector.detect_array(points)
        labels = {detection.label for detection in detections}

        self.assertIn("vehicle", labels)
        self.assertIn("pedestrian", labels)
        self.assertIn("unknown_rail_obstacle", labels)

    def test_output_policy_limits_vehicle_range_and_count(self):
        rng = np.random.default_rng(8)
        boxes = []
        for index in range(18):
            boxes.append(make_box(rng, center=[20.0 + index * 2.0, 4.0, 0.9], size=[3.0, 1.6, 1.5], count=80))
        too_near = make_box(rng, center=[10.0, 4.0, 0.9], size=[3.0, 1.6, 1.5], count=80)
        points = np.vstack([*boxes, too_near]).astype(np.float32)
        detector = LidarObjectDetector(
            {
                "ground": {"method": "none"},
                "preprocess": {"voxel_size": 0.02},
                "clustering": {"method": "dbscan", "eps": 0.45, "min_points": 8},
                "output_policy": {
                    "per_label": {
                        "vehicle": {"min_range": 15.0, "max_range": 80.0, "max_outputs": 15},
                    }
                },
            }
        )

        detections = [d for d in detector.detect_array(points) if d.label == "vehicle"]

        self.assertLessEqual(len(detections), 15)
        self.assertTrue(all(d.bbox.center[0] >= 15.0 for d in detections))

    def test_rejects_wall_like_cluster_as_vehicle(self):
        rng = np.random.default_rng(7)
        wall = make_box(rng, center=[12.0, 8.0, 1.2], size=[7.0, 0.6, 2.4], count=500)
        detector = LidarObjectDetector(
            {
                "ground": {"method": "none"},
                "preprocess": {"voxel_size": 0.05},
                "clustering": {"method": "dbscan", "eps": 0.8, "min_points": 10},
            }
        )

        detections = detector.detect_array(wall.astype(np.float32))

        self.assertEqual([], detections)

    def test_rejects_thin_line_like_structure_as_vehicle(self):
        rng = np.random.default_rng(13)
        line_points = make_box(rng, center=[30.0, 4.0, 0.8], size=[3.2, 0.45, 1.4], count=180)
        detector = LidarObjectDetector(
            {
                "ground": {"method": "none"},
                "preprocess": {"voxel_size": 0.05},
                "clustering": {"method": "grid", "grid_cell_size": 0.35, "connectivity": 4, "min_points": 6},
            }
        )

        detections = detector.detect_array(line_points.astype(np.float32))

        self.assertEqual([], detections)

    def test_detects_partial_vehicle_cluster(self):
        rng = np.random.default_rng(11)
        partial_vehicle = make_box(rng, center=[8.0, 3.0, 0.6], size=[2.1, 1.0, 0.7], count=80)
        detector = LidarObjectDetector(
            {
                "ground": {"method": "none"},
                "preprocess": {"voxel_size": 0.05},
                "clustering": {"method": "grid", "grid_cell_size": 0.35, "connectivity": 4, "min_points": 6},
                "output_policy": {"per_label": {"vehicle": {"min_range": 0.0, "max_range": 120.0}}},
            }
        )

        detections = detector.detect_array(partial_vehicle.astype(np.float32))

        self.assertTrue(any(detection.label == "vehicle" for detection in detections))

    def test_merges_split_vehicle_parts(self):
        rng = np.random.default_rng(17)
        front = make_box(rng, center=[25.0, 2.0, 0.7], size=[1.4, 1.5, 0.9], count=80)
        rear = make_box(rng, center=[26.65, 2.0, 0.7], size=[1.4, 1.5, 0.9], count=80)
        points = np.vstack([front, rear]).astype(np.float32)
        detector = LidarObjectDetector(
            {
                "ground": {"method": "none"},
                "preprocess": {"voxel_size": 0.05},
                "clustering": {
                    "method": "grid",
                    "grid_cell_size": 0.35,
                    "connectivity": 4,
                    "merge_gap": 0.45,
                    "min_points": 6,
                },
            }
        )

        detections = detector.detect_array(points)

        self.assertTrue(any(detection.label == "vehicle" for detection in detections))

    def test_visualize_writes_html(self):
        payload = {
            "source": "demo",
            "num_frames": 1,
            "frames": [
                {
                    "frame_index": 0,
                    "num_detections": 1,
                    "detections": [
                        {
                            "label": "vehicle",
                            "confidence": 0.9,
                            "num_points": 100,
                            "rail_overlap": 0.0,
                            "bbox": {"center": [1, 2, 0], "size": [4, 2, 1], "yaw": 0.1},
                        }
                    ],
                }
            ],
        }
        tmp = Path(".codex_tmp")
        tmp.mkdir(exist_ok=True)
        json_path = tmp / "visual_test_detections.json"
        html_path = tmp / "visual_test_detections.html"
        with json_path.open("w", encoding="utf-8") as handle:
            import json

            json.dump(payload, handle)

        visualize_main([str(json_path), "--config", "config/default.yaml", "--output", str(html_path)])

        html = html_path.read_text(encoding="utf-8")
        self.assertIn("vehicle", html)

    def test_realtime_viewer_parser_defaults_to_nonground_topic(self):
        args = build_realtime_parser().parse_args([])

        self.assertEqual(args.topic, "/ground_segmentation/nonground")
        self.assertEqual(args.initial_mode, MODE_BOTH)

    def test_ros_node_parser_defaults_to_topic_publishers(self):
        args = build_ros_node_parser().parse_args([])

        self.assertEqual(args.input_topic, "/ground_segmentation/nonground")
        self.assertEqual(args.points_topic, "/lidar_detector/points")
        self.assertEqual(args.markers_topic, "/lidar_detector/detection_markers")
        self.assertEqual(args.max_processing_points, 60000)

    def test_grid_clustering_detects_vehicle(self):
        rng = np.random.default_rng(9)
        vehicle = make_box(rng, center=[25.0, 2.0, 0.9], size=[4.2, 1.8, 1.6], count=300)
        detector = LidarObjectDetector(
            {
                "ground": {"method": "none"},
                "preprocess": {"voxel_size": 0.05},
                "clustering": {"method": "grid", "grid_cell_size": 0.6, "min_points": 8},
            }
        )

        detections = detector.detect_array(vehicle.astype(np.float32))

        self.assertIn("vehicle", {d.label for d in detections})


if __name__ == "__main__":
    unittest.main()
