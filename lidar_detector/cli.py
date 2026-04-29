"""Command line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .io import bag_detections_payload, detections_payload, load_bag_pointclouds, load_point_cloud, save_detections_json
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
    parser = argparse.ArgumentParser(description="Detect vehicles, pedestrians, and unknown rail obstacles in LiDAR point clouds.")
    parser.add_argument("input", help="Point cloud path: .bag, .bin, .npy, .txt, .csv, .pcd, or .ply")
    parser.add_argument("--config", help="YAML config path", default=None)
    parser.add_argument("--output", help="Write detections JSON to this path", default=None)
    parser.add_argument("--bin-dim", type=int, choices=(3, 4, 5), default=None, help="Point dimension for .bin files")
    parser.add_argument("--topic", default="/ground_segmentation/nonground", help="PointCloud2 topic for .bag input")
    parser.add_argument("--frame-index", type=int, default=None, help="Only process one zero-based bag frame index")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after this many bag frames")
    parser.add_argument("--skip-ground-removal", action="store_true", help="Use this when input is already non-ground point cloud")
    parser.add_argument("--rail-centerline", type=_parse_centerline, default=None, help="Override rail centerline, e.g. '-50,0;50,0'")
    parser.add_argument("--rail-width", type=float, default=None, help="Override rail corridor width in meters")
    parser.add_argument("--print-json", action="store_true", help="Print full JSON payload")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    override = {}
    if args.rail_centerline is not None:
        override.setdefault("rail", {})["centerlines"] = [args.rail_centerline]
    if args.rail_width is not None:
        override.setdefault("rail", {})["corridor_width"] = args.rail_width
    if args.skip_ground_removal:
        override.setdefault("ground", {})["method"] = "none"

    config = load_config(args.config, override=override)
    detector = LidarObjectDetector(config)
    input_path = Path(args.input)
    if input_path.suffix.lower() == ".bag":
        frames = []
        for metadata, points in load_bag_pointclouds(
            input_path,
            topic=args.topic,
            max_frames=args.max_frames,
            frame_index=args.frame_index,
        ):
            detections = detector.detect_array(points)
            frame_payload = {
                **metadata,
                "num_points": int(len(points)),
                "num_detections": len(detections),
                "detections": [detection.to_dict() for detection in detections],
            }
            frames.append(frame_payload)
        payload = bag_detections_payload(str(input_path), frames)
    else:
        points = load_point_cloud(input_path, bin_dim=args.bin_dim)
        detections = detector.detect_array(points)
        payload = detections_payload(str(input_path), detections)

    if args.output:
        save_detections_json(args.output, payload)
        if "frames" in payload:
            count = sum(frame["num_detections"] for frame in payload["frames"])
            print(f"Wrote {count} detections from {payload['num_frames']} frames to {args.output}")
        else:
            print(f"Wrote {payload['num_detections']} detections to {args.output}")

    if args.print_json or not args.output:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
