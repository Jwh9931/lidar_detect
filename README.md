# LiDAR Rail Object Detection Baseline

这是一个面向激光雷达点云的目标识别基线工程，支持识别：

- `vehicle`: 车辆
- `pedestrian`: 行人
- `unknown_rail_obstacle`: 铁轨走廊内的未知障碍物

当前实现是无需训练数据的几何规则基线：先做 ROI 过滤、体素降采样、地面分割、DBSCAN 聚类，再用 3D 包围盒尺寸规则识别车辆和行人；未能识别为车辆/行人的聚类，如果落在配置的铁轨走廊内，就输出为铁轨未知障碍物。

## 快速开始

```bash
python -m pip install -r requirements.txt
python -m lidar_detector.cli path/to/pointcloud.bin --config config/default.yaml --output detections.json
```

支持输入格式：

- `.bin`: KITTI 风格 `float32` 点云，默认自动判断 `x y z intensity` 或 `x y z`
- `.npy`: NumPy 数组，形状为 `N x 3` 或 `N x 4+`
- `.txt` / `.csv`: 至少包含 `x y z` 三列
- `.pcd` / `.ply`: 需要安装可选依赖 `open3d`
- `.bag`: ROS1 bag 中的 `sensor_msgs/PointCloud2` 话题，需要先 source ROS 环境

ROS bag 示例：

```bash
source /opt/ros/noetic/setup.bash
python -m lidar_detector.cli /data/datalog/PointPillars/nogroud.bag \
  --topic /ground_segmentation/nonground \
  --skip-ground-removal \
  --config config/default.yaml \
  --output detections.json
```

只跑第 1 帧做调参：

```bash
python -m lidar_detector.cli /data/datalog/PointPillars/nogroud.bag \
  --topic /ground_segmentation/nonground \
  --frame-index 0 \
  --skip-ground-removal \
  --print-json
```

输出示例：

```json
{
  "source": "frame.bin",
  "num_detections": 3,
  "detections": [
    {
      "label": "vehicle",
      "confidence": 0.91,
      "num_points": 420,
      "rail_overlap": 0.0,
      "bbox": {
        "center": [21.1, 4.8, 0.9],
        "size": [4.5, 2.0, 1.8],
        "yaw": 0.03
      }
    }
  ]
}
```

## 配置铁轨区域

在 `config/default.yaml` 中配置铁轨中心线和走廊宽度：

```yaml
rail:
  centerlines:
    - [[-100.0, 0.0], [100.0, 0.0]]
  corridor_width: 3.0
  min_overlap: 0.2
```

`centerlines` 是一条或多条二维折线，每个点为 `[x, y]`。`corridor_width` 表示铁轨附近需要监测的总宽度，聚类点中至少 `min_overlap` 的比例落入该走廊时，才会被认为在铁轨区域内。

## 技术指标过滤

默认配置按雷达常见坐标系处理：`x` 为前方距离，`y` 为横向距离，`z` 为高度。若你的车体坐标系不同，请先修改 `output_policy.range_metric` 或 ROI。

当前默认策略：

- ROI：只保留前向 `0-300m`、横向 `-20m 到 20m`、高度 `-3m 到 6m` 的点云。
- 小车辆：输出距离 `15-80m`，最大输出 `15` 个。
- 行人：输出距离 `0-120m`，最大输出 `15` 个。
- 异常类：输出距离 `5-60m`，最大输出 `20` 个，检测框尺寸不小于 `0.3m x 0.3m x 0.3m`。
- 大型轨道障碍物：输出距离 `0-300m`，用于直道 300m 内列车同等反射面大小障碍物的几何候选。

对应配置在 `config/default.yaml`：

```yaml
output_policy:
  range_metric: forward_x
  per_label:
    vehicle:
      min_range: 15.0
      max_range: 80.0
      max_outputs: 15
    pedestrian:
      max_range: 120.0
      max_outputs: 15
    unknown_rail_obstacle:
      min_range: 5.0
      max_range: 60.0
      max_outputs: 20
      min_size: [0.3, 0.3, 0.3]
```

注意：召回率和准确率需要基于现场标注数据统计。当前版本通过几何规则实现距离、ROI、尺寸和输出数量约束；若要稳定达到 `召回率 >85%`、`准确率 >95%`，建议后续接入经过现场数据训练/验证的 3D 检测模型。

## 自检

```powershell
python -m unittest discover tests
```

## 可视化

生成浏览器可打开的 BEV 俯视图：

```bash
python -m lidar_detector.visualize detections.json \
  --config config/default.yaml \
  --output detections.html
```

然后在 Jetson 桌面环境中打开：

```bash
xdg-open detections.html
```

如果 Jetson 没接显示器，可以把 `detections.html` 拷到自己的电脑上用浏览器打开。

## 实时显示

实时订阅 ROS1 `PointCloud2` 话题并显示 BEV 点云和检测框：

```bash
source /opt/ros/noetic/setup.bash
source .venv/bin/activate

python -m lidar_detector.realtime_viewer \
  --topic /ground_segmentation/nonground \
  --skip-ground-removal \
  --config config/default.yaml
```

窗口右侧有三种切换按钮：

- `pointcloud+detections`: 同时显示点云和检测结果
- `pointcloud only`: 只显示点云
- `detections only`: 只显示检测结果

如果提示缺少 `matplotlib`，在 Jetson 上安装：

```bash
sudo apt install -y python3-matplotlib
```

虚拟环境建议使用 `--system-site-packages` 创建，这样能访问 ROS 和 apt 安装的 Python 包：

```bash
python3 -m venv --system-site-packages .venv
```

## ROS Topic 输出 + RViz 显示

如果设备没有图形窗口，或实时窗口无法显示，推荐发布为 ROS topic 后用 RViz 查看：

```bash
source /opt/ros/noetic/setup.bash
source .venv/bin/activate

python -m lidar_detector.ros_node \
  --input-topic /ground_segmentation/nonground \
  --skip-ground-removal \
  --config config/default.yaml
```

默认输出 topic：

- `/lidar_detector/points`: 点云，类型 `sensor_msgs/PointCloud2`
- `/lidar_detector/detection_markers`: 检测框和类别文字，类型 `visualization_msgs/MarkerArray`
- `/lidar_detector/detections_json`: 检测结果 JSON，类型 `std_msgs/String`

RViz 中添加：

- `PointCloud2`，Topic 选择 `/lidar_detector/points`
- `MarkerArray`，Topic 选择 `/lidar_detector/detection_markers`

三种显示方式可以在 RViz 左侧 Displays 里切换：

- 点云 + 检测结果：同时勾选 `PointCloud2` 和 `MarkerArray`
- 只显示点云：只勾选 `PointCloud2`
- 只显示检测结果：只勾选 `MarkerArray`

### 提高 topic 输出频率

实时节点默认采用“只处理最新帧”的异步模式，避免检测慢时堆积延迟。Jetson 上如果检测结果 topic 频率仍低，可以先用下面的轻量配置：

```bash
python -m lidar_detector.ros_node \
  --input-topic /ground_segmentation/nonground \
  --skip-ground-removal \
  --config config/default.yaml \
  --max-processing-points 30000 \
  --no-label-markers \
  --compact-json
```

如果还不够快，可以每 2 帧处理 1 帧：

```bash
python -m lidar_detector.ros_node \
  --input-topic /ground_segmentation/nonground \
  --skip-ground-removal \
  --config config/default.yaml \
  --max-processing-points 30000 \
  --no-label-markers \
  --compact-json \
  --process-every-n 2
```

如果目标是压到 100 ms 以下，建议先使用：

```yaml
preprocess:
  voxel_size: 0.5
  voxel_mode: first

clustering:
  method: grid
  grid_cell_size: 0.8
```

`voxel_mode: first` 比 `mean` 快，因为它只保留每个体素的首个点，不计算体素均值。还可以继续收小 `roi` 范围。这样会牺牲一部分小目标细节，但能明显提高聚类速度。

## 一键启动

Jetson 上可以用脚本启动完整流程：

```bash
cd /data/datalog/lidar_detect
chmod +x scripts/start_lidar_detect.sh
./scripts/start_lidar_detect.sh nogroud.bag
```

脚本会自动执行：

- `source /opt/ros/noetic/setup.bash`
- 创建并激活 `.venv`
- 启动 `roscore`
- 启动检测节点
- 执行 `rosbag play nogroud.bag -l -r 2`

常用参数通过环境变量调整：

```bash
BAG_RATE=1 MAX_PROCESSING_POINTS=20000 ./scripts/start_lidar_detect.sh nogroud.bag
```

如果 roscore 已经在运行，脚本会复用现有 ROS master。按 `Ctrl+C` 会停止本脚本启动的检测节点、bag 播放和 roscore。

默认聚类方式是适合 Jetson 实时运行的 BEV 网格连通域：

```yaml
clustering:
  method: grid
  grid_cell_size: 0.6
```

如果需要更精细但更慢的聚类，可以改回：

```yaml
clustering:
  method: dbscan
  eps: 0.75
```

提高频率的推荐顺序：

1. `--max-processing-points 30000`
2. `preprocess.voxel_size: 0.20` 或 `0.25`
3. 收窄 `roi.y`，例如只看轨道周边 `[-10, 10]`
4. 增大 `clustering.grid_cell_size` 到 `0.8`

## 生产化建议

这个版本适合快速接入真实点云、验证坐标系和轨道区域逻辑。如果你有标注数据，建议后续把 `classifier.py` 中的规则分类器替换或叠加为 PointPillars、CenterPoint、PV-RCNN 等 3D 检测模型，并保留当前的铁轨走廊后处理来识别模型类别之外的未知障碍物。
