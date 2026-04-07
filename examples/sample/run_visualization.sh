#!/bin/bash
# 轨迹可视化脚本 - 快速参考

cd /home/zsl/RL_AUV_tracking

echo "========================================="
echo "可视化 LQR+SAC 策略的采样轨迹"
echo "========================================="
echo ""

# 设置数据路径
DATA_PATH="log/sample/v1_episodes_lqr_sac/auv_data_final.zarr"
OUTPUT_DIR="log/sample/visualizations"
PYTHON_PATH="/home/zsl/miniconda3/envs/auv_env/bin/python"

echo "✓ 数据源: $DATA_PATH"
echo "✓ 输出目录: $OUTPUT_DIR"
echo ""

echo "생성 2D 俯视图轨迹 (trajectory_2d.png)..."
$PYTHON_PATH examples/sample/visualize_trajectories.py \
  --data "$DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --type 2d \
  --num-episodes 10

echo ""
echo "生成 3D 立体轨迹 (trajectories_3d.png)..."
$PYTHON_PATH examples/sample/visualize_trajectories.py \
  --data "$DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --type 3d \
  --num-episodes 10

echo ""
echo "生成统计分析 (trajectory_statistics.png)..."
$PYTHON_PATH examples/sample/visualize_trajectories.py \
  --data "$DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --type stats

echo ""
echo "生成单个 episode 详细图 (episode_0_detailed.png)..."
$PYTHON_PATH examples/sample/visualize_trajectories.py \
  --data "$DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --type single \
  --single-episode 0

echo ""
echo "========================================="
echo "✓ 所有可视化已完成！"
echo "📁 输出文件位置: $OUTPUT_DIR"
echo "========================================="
echo ""
echo "生成的文件："
ls -lh "$OUTPUT_DIR"/*.png
