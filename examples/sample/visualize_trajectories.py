"""
可视化轨迹数据脚本
从 zarr 格式的数据中提取轨迹数据并生成 2D/3D 可视化
"""

import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互后端，避免需要图形显示
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from diffusion_policy.common.replay_buffer import ReplayBuffer
from pathlib import Path
import argparse

def extract_trajectory_from_state(state_data):
    """
    从状态数据中提取轨迹
    假设状态格式为: [x, y, z, vx, vy, vz, roll, pitch, yaw, ...]
    或 2D 格式: [x, y, vx, vy, ...]
    
    Args:
        state_data: shape (T, state_dim) 的状态数组
        
    Returns:
        trajectory: shape (T, 3) 或 (T, 2) 的位置数组
    """
    if len(state_data) == 0:
        return None
    
    state_dim = state_data.shape[1]
    
    # 根据状态维度判断是 3D 还是 2D
    if state_dim >= 3:
        # 3D 情况：[x, y, z, ...]
        trajectory = state_data[:, :3]
    elif state_dim >= 2:
        # 2D 情况：[x, y, ...]
        trajectory = state_data[:, :2]
    else:
        return None
    
    return trajectory

def plot_trajectory_2d(trajectories, episode_indices=None, output_path=None):
    """
    绘制 2D 轨迹
    
    Args:
        trajectories: list of trajectories, each of shape (T, 2) or (T, 3)
        episode_indices: 要绘制的 episode 索引列表
        output_path: 保存图像的路径
    """
    if episode_indices is None:
        episode_indices = range(min(len(trajectories), 10))  # 默认绘制前10个
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(episode_indices)))
    
    for idx, ep_idx in enumerate(episode_indices):
        if ep_idx >= len(trajectories):
            print(f"警告: Episode {ep_idx} 超出范围")
            continue
        
        traj = trajectories[ep_idx]
        if traj is None or len(traj) == 0:
            continue
        
        # 绘制轨迹
        if traj.shape[1] >= 2:
            ax.plot(traj[:, 0], traj[:, 1], 
                   color=colors[idx], linewidth=1.5, 
                   alpha=0.7, label=f'Episode {ep_idx}')
            
            # 标记起点和终点
            ax.scatter(traj[0, 0], traj[0, 1], 
                      color=colors[idx], s=100, marker='o', 
                      edgecolors='black', linewidths=1.5, zorder=5)  # 起点
            ax.scatter(traj[-1, 0], traj[-1, 1], 
                      color=colors[idx], s=100, marker='s', 
                      edgecolors='black', linewidths=1.5, zorder=5)  # 终点
    
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title('AUV Trajectories (Top View - 2D)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 2D轨迹图已保存: {output_path}")
    
    return fig, ax

def plot_trajectory_3d(trajectories, episode_indices=None, output_path=None):
    """
    绘制 3D 轨迹
    
    Args:
        trajectories: list of trajectories, each of shape (T, 3)
        episode_indices: 要绘制的 episode 索引列表
        output_path: 保存图像的路径
    """
    if episode_indices is None:
        episode_indices = range(min(len(trajectories), 10))
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(episode_indices)))
    
    for idx, ep_idx in enumerate(episode_indices):
        if ep_idx >= len(trajectories):
            print(f"警告: Episode {ep_idx} 超出范围")
            continue
        
        traj = trajectories[ep_idx]
        if traj is None or len(traj) == 0:
            continue
        
        # 需要 3D 数据
        if traj.shape[1] < 3:
            print(f"跳过 Episode {ep_idx}: 不是 3D 数据 (shape: {traj.shape})")
            continue
        
        # 绘制轨迹
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
               color=colors[idx], linewidth=1.5, 
               alpha=0.7, label=f'Episode {ep_idx}')
        
        # 标记起点和终点
        ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], 
                  color=colors[idx], s=100, marker='o', 
                  edgecolors='black', linewidths=1.5, zorder=5)  # 起点
        ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], 
                  color=colors[idx], s=100, marker='s', 
                  edgecolors='black', linewidths=1.5, zorder=5)  # 终点
    
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_zlabel('Z Position (m)', fontsize=12)
    ax.set_title('AUV Trajectories (3D View)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 3D轨迹图已保存: {output_path}")
    
    return fig, ax

def plot_single_trajectory_3d(trajectory, episode_idx, output_path=None):
    """
    绘制单个 episode 的详细 3D 轨迹
    
    Args:
        trajectory: shape (T, 3) 的轨迹数据
        episode_idx: episode 索引
        output_path: 保存路径
    """
    if trajectory is None or len(trajectory) == 0:
        print(f"轨迹数据为空")
        return
    
    if trajectory.shape[1] < 3:
        print(f"跳过: 不是 3D 数据")
        return
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 使用时间步作为颜色
    times = np.arange(len(trajectory))
    scatter = ax.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                        c=times, cmap='viridis', s=20, alpha=0.6)
    
    # 绘制连接线
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
           'b-', linewidth=1, alpha=0.3)
    
    # 标记起点和终点
    ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2],
              color='green', s=200, marker='o', 
              edgecolors='darkgreen', linewidths=2, label='Start', zorder=10)
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2],
              color='red', s=200, marker='s', 
              edgecolors='darkred', linewidths=2, label='End', zorder=10)
    
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_zlabel('Z Position (m)', fontsize=12)
    ax.set_title(f'Episode {episode_idx} - AUV Trajectory (3D)', 
                fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Time Steps', fontsize=11)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Episode {episode_idx} 详细轨迹图已保存: {output_path}")
    
    return fig, ax

def plot_trajectory_statistics(trajectories, output_path=None):
    """
    绘制轨迹统计信息
    
    Args:
        trajectories: list of trajectories
        output_path: 保存路径
    """
    if len(trajectories) == 0:
        print("没有轨迹数据")
        return
    
    # 计算统计信息
    lengths = [len(traj) if traj is not None else 0 for traj in trajectories]
    distances_2d = []
    distances_3d = []
    max_x = []
    max_y = []
    max_z = []
    
    for traj in trajectories:
        if traj is None or len(traj) == 0:
            continue
        
        # 2D 距离
        if traj.shape[1] >= 2:
            dist = np.sum(np.sqrt(np.sum(np.diff(traj[:, :2], axis=0)**2, axis=1)))
            distances_2d.append(dist)
        
        # 3D 距离
        if traj.shape[1] >= 3:
            dist = np.sum(np.sqrt(np.sum(np.diff(traj[:, :3], axis=0)**2, axis=1)))
            distances_3d.append(dist)
        
        # 位置范围
        max_x.append(np.max(np.abs(traj[:, 0])))
        max_y.append(np.max(np.abs(traj[:, 1])))
        if traj.shape[1] >= 3:
            max_z.append(np.max(np.abs(traj[:, 2])))
    
    # 创建统计图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Episode 长度分布
    axes[0, 0].hist(lengths, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Episode Length (steps)', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Episode Length Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 2D 轨迹距离分布
    if distances_2d:
        axes[0, 1].hist(distances_2d, bins=20, edgecolor='black', alpha=0.7, color='orange')
        axes[0, 1].set_xlabel('2D Travel Distance (m)', fontsize=11)
        axes[0, 1].set_ylabel('Frequency', fontsize=11)
        axes[0, 1].set_title('2D Distance Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 3D 轨迹距离分布
    if distances_3d:
        axes[1, 0].hist(distances_3d, bins=20, edgecolor='black', alpha=0.7, color='green')
        axes[1, 0].set_xlabel('3D Travel Distance (m)', fontsize=11)
        axes[1, 0].set_ylabel('Frequency', fontsize=11)
        axes[1, 0].set_title('3D Distance Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 位置范围
    axes[1, 1].boxplot([max_x, max_y, max_z if max_z else max_x], 
                       tick_labels=['X', 'Y', 'Z'],
                       patch_artist=True)
    axes[1, 1].set_ylabel('Max Absolute Position (m)', fontsize=11)
    axes[1, 1].set_title('Position Range Statistics', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 统计图已保存: {output_path}")
    
    return fig, axes

def main():
    parser = argparse.ArgumentParser(
        description='可视化 AUV 轨迹数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 可视化所有类型的图表
  python examples/sample/visualize_trajectories.py --data log/sample/v1_episodes_lqr_sac/auv_data_final.zarr

  # 指定特定类型的图表
  python examples/sample/visualize_trajectories.py --data log/sample/v1_episodes_lqr_sac/auv_data_final.zarr --type 2d

  # 可视化特定的 episodes
  python examples/sample/visualize_trajectories.py --data log/sample/v1_episodes_lqr_sac/auv_data_final.zarr --episodes 0 1 2 3
        """)
    
    parser.add_argument('--data', type=str, 
                       default='log/sample/v1_episodes_lqr_sac/auv_data_final.zarr',
                       help='数据集路径 (.zarr 格式)')
    parser.add_argument('--output-dir', type=str, 
                       default='log/sample/visualizations',
                       help='输出图像目录')
    parser.add_argument('--type', type=str, 
                       choices=['2d', '3d', 'stats', 'all', 'single'],
                       default='all',
                       help='可视化类型')
    parser.add_argument('--episodes', type=int, nargs='+', 
                       default=None,
                       help='指定要绘制的 episode 索引列表')
    parser.add_argument('--num-episodes', type=int, 
                       default=10,
                       help='默认绘制的 episodes 数量')
    parser.add_argument('--single-episode', type=int, 
                       default=0,
                       help='单独绘制的 episode 索引')
    
    args = parser.parse_args()
    
    # 检查数据文件是否存在
    if not os.path.exists(args.data):
        print(f"❌ 错误: 数据文件不存在: {args.data}")
        return
    
    print("=" * 60)
    print("【AUV 轨迹可视化工具】")
    print("=" * 60)
    print(f"数据路径: {args.data}")
    print(f"输出目录: {args.output_dir}")
    print()
    
    # 加载数据
    print("加载数据...")
    try:
        replay_buffer = ReplayBuffer.copy_from_path(args.data, keys=['state'])
    except Exception as e:
        print(f"❌ 错误: 无法加载数据: {e}")
        return
    
    print(f"✓ 数据加载成功")
    print(f"  Episode 数量: {replay_buffer.n_episodes}")
    print(f"  总步数: {replay_buffer.n_steps}")
    print(f"  State 数据形状: {replay_buffer['state'].shape}")
    print()
    
    # 提取所有 episodes 的轨迹
    print("提取轨迹数据...")
    trajectories = []
    episode_length_info = []
    
    for ep_idx in range(replay_buffer.n_episodes):
        ep_data = replay_buffer.get_episode(ep_idx)
        if 'state' in ep_data:
            state = ep_data['state']
            traj = extract_trajectory_from_state(state)
            trajectories.append(traj)
            episode_length_info.append((ep_idx, len(state), traj.shape if traj is not None else None))
    
    print(f"✓ 提取完成, 共 {len(trajectories)} 个 episodes")
    print()
    
    # 打印数据信息
    print("【数据信息】")
    for ep_idx, length, shape in episode_length_info[:5]:
        print(f"  Episode {ep_idx}: 长度 {length}, 轨迹形状 {shape}")
    if len(episode_length_info) > 5:
        print(f"  ... 还有 {len(episode_length_info) - 5} 个 episodes")
    print()
    
    # 确定要绘制的 episodes
    if args.episodes:
        plot_episodes = args.episodes
    else:
        plot_episodes = list(range(min(args.num_episodes, len(trajectories))))
    
    # 根据类型进行可视化
    print("生成可视化...")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.type in ['2d', 'all']:
        try:
            plot_trajectory_2d(trajectories, plot_episodes, 
                             f"{args.output_dir}/trajectories_2d.png")
        except Exception as e:
            print(f"⚠ 2D 绘图失败: {e}")
    
    if args.type in ['3d', 'all']:
        try:
            plot_trajectory_3d(trajectories, plot_episodes, 
                             f"{args.output_dir}/trajectories_3d.png")
        except Exception as e:
            print(f"⚠ 3D 绘图失败: {e}")
    
    if args.type in ['stats', 'all']:
        try:
            plot_trajectory_statistics(trajectories, 
                                      f"{args.output_dir}/trajectory_statistics.png")
        except Exception as e:
            print(f"⚠ 统计图绘制失败: {e}")
    
    if args.type in ['single', 'all']:
        if args.single_episode < len(trajectories):
            try:
                plot_single_trajectory_3d(trajectories[args.single_episode], 
                                        args.single_episode,
                                        f"{args.output_dir}/episode_{args.single_episode}_detailed.png")
            except Exception as e:
                print(f"⚠ 单个 episode 绘图失败: {e}")
    
    print()
    print("=" * 60)
    print("✓ 可视化完成!")
    print(f"  所有图像已保存到: {args.output_dir}")
    print("=" * 60)
    
    # 显示图表
    if args.type != 'all':
        print("\n提示: 按 Ctrl+C 关闭所有图表")
        try:
            plt.show()
        except KeyboardInterrupt:
            print("已关闭")

if __name__ == '__main__':
    main()
