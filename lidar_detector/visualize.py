"""Create a lightweight browser-based BEV visualization for detections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config


HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LiDAR Detections BEV</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      background: #11151b;
      color: #e9eef5;
    }}
    header {{
      display: flex;
      align-items: center;
      gap: 16px;
      padding: 12px 16px;
      border-bottom: 1px solid #2a3441;
      background: #171d25;
    }}
    canvas {{
      display: block;
      width: 100vw;
      height: calc(100vh - 58px);
      background: #0c1015;
    }}
    input[type="range"] {{
      width: min(560px, 42vw);
    }}
    .legend {{
      display: flex;
      gap: 12px;
      margin-left: auto;
      font-size: 13px;
    }}
    .swatch {{
      display: inline-block;
      width: 10px;
      height: 10px;
      margin-right: 5px;
      border-radius: 2px;
    }}
    button {{
      background: #243143;
      color: #e9eef5;
      border: 1px solid #3a4a5f;
      border-radius: 4px;
      padding: 6px 10px;
      cursor: pointer;
    }}
  </style>
</head>
<body>
  <header>
    <button id="prev">上一帧</button>
    <input id="frame" type="range" min="0" max="0" value="0">
    <button id="next">下一帧</button>
    <span id="status"></span>
    <div class="legend">
      <span><span class="swatch" style="background:#35c46a"></span>vehicle</span>
      <span><span class="swatch" style="background:#4aa3ff"></span>pedestrian</span>
      <span><span class="swatch" style="background:#ff5a5f"></span>unknown rail obstacle</span>
      <span><span class="swatch" style="background:#f5c84b"></span>rail</span>
    </div>
  </header>
  <canvas id="bev"></canvas>
  <script>
    const payload = {payload_json};
    const config = {config_json};
    const frames = payload.frames || [payload];
    const roi = config.roi || {{x: [-80, 80], y: [-40, 40]}};
    const rail = config.rail || {{}};
    const colors = {{
      vehicle: "#35c46a",
      pedestrian: "#4aa3ff",
      unknown_rail_obstacle: "#ff5a5f"
    }};
    const canvas = document.getElementById("bev");
    const ctx = canvas.getContext("2d");
    const slider = document.getElementById("frame");
    const status = document.getElementById("status");
    slider.max = Math.max(0, frames.length - 1);

    function resize() {{
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(canvas.clientWidth * dpr);
      canvas.height = Math.floor(canvas.clientHeight * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw(Number(slider.value));
    }}

    function toScreen(x, y) {{
      const pad = 36;
      const xMin = roi.x[0], xMax = roi.x[1];
      const yMin = roi.y[0], yMax = roi.y[1];
      const w = canvas.clientWidth - pad * 2;
      const h = canvas.clientHeight - pad * 2;
      const scale = Math.min(w / (xMax - xMin), h / (yMax - yMin));
      const ox = (canvas.clientWidth - (xMax - xMin) * scale) / 2;
      const oy = (canvas.clientHeight - (yMax - yMin) * scale) / 2;
      return [ox + (x - xMin) * scale, oy + (yMax - y) * scale, scale];
    }}

    function drawGrid() {{
      ctx.strokeStyle = "#202a35";
      ctx.lineWidth = 1;
      for (let x = Math.ceil(roi.x[0] / 10) * 10; x <= roi.x[1]; x += 10) {{
        const a = toScreen(x, roi.y[0]), b = toScreen(x, roi.y[1]);
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
      }}
      for (let y = Math.ceil(roi.y[0] / 10) * 10; y <= roi.y[1]; y += 10) {{
        const a = toScreen(roi.x[0], y), b = toScreen(roi.x[1], y);
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
      }}
    }}

    function drawRail() {{
      ctx.strokeStyle = "#f5c84b";
      ctx.lineWidth = 2;
      for (const line of rail.centerlines || []) {{
        ctx.beginPath();
        for (let i = 0; i < line.length; i++) {{
          const p = toScreen(line[i][0], line[i][1]);
          if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
        }}
        ctx.stroke();
      }}
    }}

    function drawBox(det) {{
      const box = det.bbox;
      const c = toScreen(box.center[0], box.center[1]);
      const scale = c[2];
      const length = box.size[0] * scale;
      const width = box.size[1] * scale;
      ctx.save();
      ctx.translate(c[0], c[1]);
      ctx.rotate(-box.yaw);
      ctx.strokeStyle = colors[det.label] || "#ddd";
      ctx.fillStyle = (colors[det.label] || "#ddd") + "22";
      ctx.lineWidth = Math.max(2, Math.min(5, scale * 0.04));
      ctx.beginPath();
      ctx.rect(-length / 2, -width / 2, length, width);
      ctx.fill();
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(length / 2, 0);
      ctx.lineTo(length / 2 - Math.min(12, length * 0.25), -Math.min(8, width * 0.35));
      ctx.lineTo(length / 2 - Math.min(12, length * 0.25), Math.min(8, width * 0.35));
      ctx.closePath();
      ctx.fillStyle = colors[det.label] || "#ddd";
      ctx.fill();
      ctx.restore();

      ctx.fillStyle = colors[det.label] || "#ddd";
      ctx.font = "12px Arial";
      ctx.fillText(`${{det.label}} ${{det.confidence.toFixed(2)}}`, c[0] + 5, c[1] - 5);
    }}

    function draw(index) {{
      ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
      drawGrid();
      drawRail();
      const frame = frames[index] || {{}};
      for (const det of frame.detections || []) drawBox(det);
      status.textContent = `帧 ${{index + 1}} / ${{frames.length}}，检测 ${{(frame.detections || []).length}} 个`;
    }}

    slider.addEventListener("input", () => draw(Number(slider.value)));
    document.getElementById("prev").addEventListener("click", () => {{
      slider.value = Math.max(0, Number(slider.value) - 1);
      draw(Number(slider.value));
    }});
    document.getElementById("next").addEventListener("click", () => {{
      slider.value = Math.min(frames.length - 1, Number(slider.value) + 1);
      draw(Number(slider.value));
    }});
    window.addEventListener("resize", resize);
    resize();
  </script>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a browser BEV visualization from detections JSON.")
    parser.add_argument("input", help="detections.json produced by lidar_detector.cli")
    parser.add_argument("--config", default="config/default.yaml", help="Config used for ROI and rail centerlines")
    parser.add_argument("--output", default="detections.html", help="Output HTML path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    config = load_config(args.config)
    html = HTML_TEMPLATE.format(
        payload_json=json.dumps(payload, ensure_ascii=False),
        config_json=json.dumps(config, ensure_ascii=False),
    )
    Path(args.output).write_text(html, encoding="utf-8")
    print(f"Wrote visualization to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
