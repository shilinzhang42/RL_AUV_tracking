# 轨迹数据可视化 - 使用指南

## 📋 简介

这个工具用于可视化 AUV 采集的轨迹数据（存储在 zarr 格式中）。支持多种可视化方式：

- **2D 轨迹图**：俯视图展示所有 episodes 的轨迹
- **3D 轨迹图**：三维视角展示轨迹
- **统计分析**：episode 长度、轨迹距离等统计信息
- **单 Episode 详细图**：单个 episode 的详细轨迹，带时间步着色

## 🚀 快速开始

### 基本使用

最简单的使用方式：

```bash
cd /home/zsl/RL_AUV_tracking
python examples/sample/visualize_trajectories.py --data log/sample/v1_episodes_lqr_sac/auv_data_final.zarr
```

### 指定可视化类型

生成 2D 轨迹图：
```bash
python examples/sample/visualize_trajectories.py \
  --data log/sample/v1_episodes_lqr_sac/auv_data_final.zarr \
  --type 2d
```

生成 3D 轨迹图：
```bash
python examples/sample/visualize_trajectories.py \
  --data log/sample/v1_episodes_lqr_sac/auv_data_final.zarr \
  --type 3d
```

生成统计图表：
```bash
python examples/sample/visualize_trajectories.py \
  --data log/sample/v1_episodes_lqr_sac/auv_data_final.zarr \
  --type stats
```

生成所有类型（默认）：
```bash
python examples/sample/visualize_trajectories.py \
  --data log/sample/v1_episodes_lqr_sac/auv_data_final.zarr \
  --type all
```

生成单个 episode 的详细图：
```bash
python examples/sample/visualize_trajectories.py \
  --data log/sample/v1_episodes_lqr_sac/auv_data_final.zarr \
  --type single \
  --single-episode 5
```

## 📊 输出文件说明

脚本生成以下图像文件，默认保存在 `log/sample/visualizations/` 目录：

### 1. `trajectories_2d.png` - 2D 轨迹图
- **描述**：俯视图（X-Y 平面），展示多个 episodes
- **用途**：快速查看轨迹的平面分布
- **特点**：
  - 不同颜色区分不同 episodes
  - 圆点标记起点，方形标记终点
  - 便于分析 AUV 在水平面上的运动

### 2. `trajectories_3d.png` - 3D 轨迹图
- **描述**：三维透视图，展示所有轴向的运动
- **用途**：全面了解 AUV 的 3D 运动轨迹
- **特点**：
  - 展示 X、Y、Z 三个坐标轴
  - 不同颜色代表不同 episodes
  - 包含起点和终点标记

### 3. `trajectory_statistics.png` - 统计分析图
包含四个子图：
1. **Episode 长度分布**：显示各 episode 的步数分布
2. **2D 轨迹距离分布**：AUV 在 X-Y 平面上的总行进距离
3. **3D 轨迹距离分布**：考虑深度的总行进距离
4. **位置范围统计**：X、Y、Z 坐标的最大值分布

### 4. `episode_{N}_detailed.png` - 单 Episode 详细图
- **描述**：指定 episode 的详细轨迹
- **用途**：深度分析单个 episode 的轨迹
- **特点**：
  - 时间步用颜色表示（colorbar）
  - 绿色圆点表示起点
  - 红色方块表示终点
  - 便于追踪 AUV 在该 episode 中的运动过程

## 🎯 命令行选项详解

```
--data              数据集路径 (.zarr 文件)
                   默认：log/sample/v1_episodes_lqr_sac/auv_data_final.zarr

--output-dir        输出图像目录
                   默认：log/sample/visualizations

--type             可视化类型
                   选项：2d, 3d, stats, all, single
                   默认：all

--episodes         指定要绘制的 episode 索引（空格分隔）
                   示例：--episodes 0 1 2 3

--num-episodes     默认绘制的 episodes 数量
                   默认：10

--single-episode   单独绘制的 episode 索引（供 --type single 使用）
                   默认：0
```

## 💡 使用示例

### 示例 1：可视化特定的 episodes

```bash
python examples/sample/visualize_trajectories.py \
  --data log/sample/v1_episodes_lqr_sac/auv_data_final.zarr \
  --type 2d \
  --episodes 0 5 10 15 20
```

### 示例 2：自定义输出目录

```bash
python examples/sample/visualize_trajectories.py \
  --data log/sample/v1_episodes_lqr_sac/auv_data_final.zarr \
  --output-dir my_visualizations
```

### 示例 3：生成特定 episode 的详细分析

```bash
python examples/sample/visualize_trajectories.py \
  --data log/sample/v1_episodes_lqr_sac/auv_data_final.zarr \
  --type single \
  --single-episode 42
```

### 示例 4：批量生成多个 episodes 的详细图

```bash
for i in {0..9}; do
  python examples/sample/visualize_trajectories.py \
    --data log/sample/v1_episodes_lqr_sac/auv_data_final.zarr \
    --type single \
    --single-episode $i \
    --output-dir log/sample/visualizations/episodes
done
```

## 📈 数据格式说明

### 状态数据 (state)
- **形状**：`(总步数, 状态维度)`
- **状态维度**：6（对于当前数据）
- **内容**：`[x, y, z, vx, vy, vz]`
- **说明**：前三维是位置坐标，后三维是速度分量

### 轨迹提取
- 脚本自动从状态数据中提取位置信息（前 3 列）
- 支持自动识别 2D 和 3D 数据

## 🔍 分析能力

### 轨迹质量指标
1. **Episode 长度**：每个 episode 包含多少个时间步
2. **轨迹距离**：AUV 实际行进的距离
3. **位置范围**：AUV 活动范围的大小

### 性能观察
- 不同 episodes 的长度是否一致（影响策略稳定性）
- 轨迹距离的变化（反映策略效果的一致性）
- 位置范围是否合理（检查是否超出预期范围）

## ⚙️ 技术细节

### 使用的库
- **matplotlib**：用于绘图
- **numpy**：用于数据处理
- **zarr**：用于读取数据文件
- **diffusion_policy.common.replay_buffer**：用于 zarr 数据加载

### 文件位置
```
/home/zsl/RL_AUV_tracking/
├── examples/sample/
│   ├── visualize_trajectories.py  ← 主脚本
│   └── README_visualization.md    ← 本文件
└── log/sample/
    ├── v1_episodes_lqr_sac/       ← 数据源
    │   └── auv_data_final.zarr
    └── visualizations/            ← 输出目录
        ├── trajectories_2d.png
        ├── trajectories_3d.png
        ├── trajectory_statistics.png
        └── episode_0_detailed.png
```

## 🐛 常见问题

### Q1：提示"找不到数据文件"
**A**：检查数据路径是否正确。可以用以下命令验证：
```bash
ls -lh log/sample/v1_episodes_lqr_sac/auv_data_final.zarr
```

### Q2：脚本运行很慢
**A**：这是正常的，因为需要加载和处理所有 episodes。处理 70 个 episodes 通常需要 30-60 秒。

### Q3：图像未显示
**A**：这是因为脚本使用了非交互后端。图像已自动保存到输出目录。

### Q4：如何处理其他数据集
**A**：只需修改 `--data` 参数指向其他 .zarr 文件即可。

### Q5：如何自定义图像大小或颜色
**A**：编辑脚本中的以下部分：
- 图像大小：`figsize` 参数
- 颜色方案：`colormap` 或 `colors` 变量

## 📝 扩展建议

可根据需要扩展脚本实现以下功能：

1. **动画生成**：为 episodes 生成动画效果
2. **对比分析**：比较不同策略、参数的轨迹差异
3. **热力图**：显示 AUV 访问频繁的区域
4. **性能指标**：计算追踪误差、目标接近度等
5. **视频输出**：生成轨迹的视频演示

## 📞 支持

如有问题，请检查：
1. Python 环境是否正确激活（使用正确的 Python 路径）
2. 依赖库是否已安装
3. 数据文件是否存在且格式正确

---

最后更新：2026年4月7日
