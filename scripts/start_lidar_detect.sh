#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "${SCRIPT_DIR}/lidar_detector" ]]; then
  PROJECT_DIR="${SCRIPT_DIR}"
else
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

BAG_PATH="${1:-nogroud.bag}"
INPUT_TOPIC="${INPUT_TOPIC:-/ground_segmentation/nonground}"
CONFIG_PATH="${CONFIG_PATH:-config/default.yaml}"
MAX_PROCESSING_POINTS="${MAX_PROCESSING_POINTS:-30000}"
BAG_RATE="${BAG_RATE:-2}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/noetic/setup.bash}"
VENV_DIR="${VENV_DIR:-.venv}"
SKIP_GROUND_REMOVAL="${SKIP_GROUND_REMOVAL:-1}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"

ROSCORE_PID=""
DETECTOR_PID=""
BAG_PID=""

cleanup() {
  set +e
  if [[ -n "${BAG_PID}" ]]; then kill "${BAG_PID}" 2>/dev/null; fi
  if [[ -n "${DETECTOR_PID}" ]]; then kill "${DETECTOR_PID}" 2>/dev/null; fi
  if [[ -n "${ROSCORE_PID}" ]]; then kill "${ROSCORE_PID}" 2>/dev/null; fi
  wait 2>/dev/null
}
trap cleanup EXIT INT TERM

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

if [[ ! -f "${ROS_SETUP}" ]]; then
  echo "ROS setup not found: ${ROS_SETUP}" >&2
  exit 1
fi

if [[ ! -f "${BAG_PATH}" ]]; then
  echo "Bag file not found: ${BAG_PATH}" >&2
  echo "Usage: $0 [bag_path]" >&2
  exit 1
fi

source "${ROS_SETUP}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating virtual environment: ${VENV_DIR}"
  python3 -m venv --system-site-packages "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

if [[ "${SKIP_INSTALL}" != "1" ]]; then
  python - <<'PY' >/dev/null 2>&1 || python -m pip install -r requirements.txt
import numpy
PY
fi

python - <<'PY'
import lidar_detector.ros_node
print("lidar_detector import OK")
PY

if ! rostopic list >/dev/null 2>&1; then
  echo "Starting roscore..."
  roscore >/tmp/lidar_detect_roscore.log 2>&1 &
  ROSCORE_PID="$!"
  for _ in $(seq 1 30); do
    if rostopic list >/dev/null 2>&1; then
      break
    fi
    sleep 0.5
  done
fi

DETECTOR_ARGS=(
  -m lidar_detector.ros_node
  --input-topic "${INPUT_TOPIC}"
  --config "${CONFIG_PATH}"
  --max-processing-points "${MAX_PROCESSING_POINTS}"
)

if [[ "${SKIP_GROUND_REMOVAL}" == "1" ]]; then
  DETECTOR_ARGS+=(--skip-ground-removal)
fi

echo "Starting detector..."
python "${DETECTOR_ARGS[@]}" &
DETECTOR_PID="$!"

sleep 1

echo "Playing bag: ${BAG_PATH}"
echo "Input topic: ${INPUT_TOPIC}"
echo "Output topics:"
echo "  /lidar_detector/points"
echo "  /lidar_detector/detection_markers"
echo "  /lidar_detector/detections_json"
echo "Press Ctrl+C to stop."

rosbag play "${BAG_PATH}" -l -r "${BAG_RATE}" &
BAG_PID="$!"
wait "${BAG_PID}"
