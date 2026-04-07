# 🎯 轨迹数据可视化工具 - 完整指南

## 📌 概述

这是一个专为 AUV 轨迹数据设计的可视化工具，支持从 zarr 格式数据中提取和可视化轨迹信息。

**数据源**：`log/sample/v1_episodes_lqr_sac/auv_data_final.zarr`（70 个 episodes，46,670 个时间步）

---

## 🚀 快速开始

### 一键生成所有可视化

```bash
bash examples/sample/run_visualization.sh
```

### 仅生成 Python 脚本执行

```bash
/home/zsl/miniconda3/envs/auv_env/bin/python examples/sample/visualize_trajectories.py
```

---

## 📊 生成的可视化文件说明

### 1️⃣ **trajectories_2d.png** - 2D俯视轨迹图
- **坐标系**：X-Y 平面（俯视图）
- **内容**：
  - 前 10 个 episodes 的轨迹曲线
  - 不同颜色区分各 episode
  - 绿色圆点 ● = 起点
  - 红色方块 ■ = 终点
- **应用**：快速查看水平运动规律
- **典型特征**：应该看到多条闭合或返回的轨迹

### 2️⃣ **trajectories_3d.png** - 3D立体轨迹图
- **坐标系**：完整的 3D 空间（X, Y, Z）
- **内容**：
  - 展示所有轨迹的三维结构
  - 包含深度信息（Z 坐标）
  - 同样的起点/终点标记
- **应用**：了解 AUV 的垂直运动
- **典型特征**：应该看到 AUV 在某个深度范围内运动

### 3️⃣ **trajectory_statistics.png** - 统计分析图
包含 4 个子图：

#### 子图 1：Episode 长度分布
- **含义**：各 episode 包含多少个时间步
- **解读**：
  - 分布越集中 → 策略越稳定
  - 分布越分散 → 性能波动较大

#### 子图 2：2D 轨迹距离分布
- **含义**：X-Y 平面上的总行进距离
- **单位**：米（m）
- **解读**：反映 AUV 的活跃程度

#### 子图 3：3D 轨迹距离分布
- **含义**：考虑深度的总行进距离
- **单位**：米（m）
- **解读**：更准确的运动距离指标

#### 子图 4：位置范围统计
- **展示**：X, Y, Z 坐标的最大值分布
- **箱线图含义**：
  - 中线 = 中位数
  - 盒子 = 中间50%的数据
  - 须线 = 极值

### 4️⃣ **episode_0_detailed.png** - 单 Episode 详细轨迹
- **特点**：
  - 放大显示第一个 episode
  - 时间步用颜色渐变表示（颜色条）
  - 早期时间 → 蓝色
  - 晚期时间 → 黄色
  - 绿色圆点 = 起点（时间步 0）
  - 红色方块 = 终点（最后一步）
- **应用**：追踪单个 episode 的运动过程

---

## 🔧 常用命令

### 查看帮助
```bash
/home/zsl/miniconda3/envs/auv_env/bin/python examples/sample/visualize_trajectories.py --help
```

### 生成特定类型的图表

**仅 2D：**
```bash
/home/zsl/miniconda3/envs/auv_env/bin/python examples/sample/visualize_trajectories.py \
  --type 2d
```

**仅 3D：**
```bash
/home/zsl/miniconda3/envs/auv_env/bin/python examples/sample/visualize_trajectories.py \
  --type 3d
```

**仅统计图：**
```bash
/home/zsl/miniconda3/envs/auv_env/bin/python examples/sample/visualize_trajectories.py \
  --type stats
```

**单个详细图：**
```bash
/home/zsl/miniconda3/envs/auv_env/bin/python examples/sample/visualize_trajectories.py \
  --type single \
  --single-episode 5
```

### 生成多个 episodes 的详细图

```bash
for i in {0..9}; do
  /home/zsl/miniconda3/envs/auv_env/bin/python examples/sample/visualize_trajectories.py \
    --type single \
    --single-episode $i \
    --output-dir log/sample/episode_details
done
```

### 指定特定的 episodes 可视化

```bash
/home/zsl/miniconda3/envs/auv_env/bin/python examples/sample/visualize_trajectories.py \
  --type 2d \
  --episodes 0 5 10 15 20
```

### 自定义输出目录

```bash
/home/zsl/miniconda3/envs/auv_env/bin/python examples/sample/visualize_trajectories.py \
  --output-dir my_figures
```

---

## 📈 数据统计信息

### 当前数据集特性（v1_episodes_lqr_sac）

```
总 Episodes：70
总时间步：46,670
State 数据维度：6 (x, y, z, vx, vy, vz)

Episode 长度分布：
  最小：423 步
  最大：900 步
  中等：约 666 步（中位数）

数据类型：3D 轨迹（考虑深度）
```

---

## 📂 文件结构

```
examples/sample/
├── visualize_trajectories.py      ✓ 主脚本
├── run_visualization.sh           ✓ 一键运行脚本
├── README_visualization.md        ✓ 详细文档
└── README_v1_sampling.md

log/sample/
├── v1_episodes_lqr_sac/
│   └── auv_data_final.zarr        📦 数据源
└── visualizations/                📊 输出目录
    ├── trajectories_2d.png
    ├── trajectories_3d.png
    ├── trajectory_statistics.png
    └── episode_0_detailed.png
```

---

## 🎓 结果解释指南

### ✅ 好的轨迹应该显示：

1. **2D 图**：
   - 轨迹曲线相对规则
   - 没有突然的大跳跃
   - 起点和终点相对接近（说明任务完成度较好）

2. **3D 图**：
   - Z 坐标（深度）在合理范围内（如 -10m 到 0m）
   - 轨迹在同一深度范围内波动
   - 没有不合理的深度变化

3. **统计图**：
   - Episode 长度分布相对集中
   - 轨迹距离在合理范围内
   - 位置范围的中位数和离群值都合理

### ⚠️ 需要关注的异常：

- **轨迹有大的跳跃**：可能是数据尖峰或策略失控
- **Episode 长度差异特别大**：策略稳定性差
- **深度过深或过浅**：环境或策略问题
- **轨迹聚集在原点**：可能是策略未有效学习

---

## 💡 进阶用法

### 比较多个数据集

```bash
# 数据集 1
/home/zsl/miniconda3/envs/auv_env/bin/python examples/sample/visualize_trajectories.py \
  --data log/sample/v1_episodes_lqr_sac/auv_data_final.zarr \
  --output-dir log/sample/viz_lqr_sac

# 数据集 2
/home/zsl/miniconda3/envs/auv_env/bin/python examples/sample/visualize_trajectories.py \
  --data log/sample/v1_episodes1/auv_data_final.zarr \
  --output-dir log/sample/viz_v1_episodes1
```

然后可视化对比两个目录中的图像。

### 生成报告

创建 Markdown 报告并嵌入生成的图像：

```markdown
# 轨迹分析报告

## 2D 轨迹
![2D 轨迹](log/sample/visualizations/trajectories_2d.png)

## 3D 轨迹  
![3D 轨迹](log/sample/visualizations/trajectories_3d.png)

## 统计分析
![统计](log/sample/visualizations/trajectory_statistics.png)
```

---

## 🔍 技术细节

### 数据处理流程

1. **加载**：使用 `ReplayBuffer` 从 zarr 文件读取数据
2. **提取**：从 state 数据中提取 [x, y, z] 坐标
3. **按 Episode 分组**：根据 episode 边界分割轨迹
4. **可视化**：生成各种图表

### 支持的数据格式

- **State 维度 ≥ 3**：自动识别为 3D（使用前 3 列）
- **State 维度 = 2**：识别为 2D（使用前 2 列）
- **自动检测**：根据数据自动选择 2D 或 3D 可视化

### 后端配置

- **图形后端**：Agg（支持服务器环境）
- **输出格式**：PNG（150 DPI，高质量）
- **内存使用**：流式处理，不一次性加载所有数据

---

## 🐛 故障排查

| 问题 | 解决方案 |
|------|--------|
| `ModuleNotFoundError: numpy` | 激活环境：使用完整 Python 路径 |
| 数据文件不存在 | 检查路径：`ls log/sample/v1_episodes_lqr_sac/auv_data_final.zarr` |
| 脚本运行很慢 | 正常现象，包含 70 个 episodes 需要时间 |
| 图像质量差 | 编辑脚本中的 `dpi=150` 为更高值（如 300） |
| 内存溢出 | 减少 `--num-episodes` 参数值 |

---

## 📚 相关文件

- [README_visualization.md](README_visualization.md) - 详细技术文档
- [visualize_trajectories.py](visualize_trajectories.py) - 脚本源代码
- [run_visualization.sh](run_visualization.sh) - 一键启动脚本

---

## 👤 作者信息

创建日期：2026年4月7日
工作环境：`/home/zsl/RL_AUV_tracking`
Python 环境：`auv_env` (Python 3.11.14)

---

**🎉 现在您已掌握所有可视化技巧！祝您分析愉快！**
