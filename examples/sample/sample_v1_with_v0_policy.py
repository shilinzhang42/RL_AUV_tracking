"""
使用v0训练的SAC策略在AUVTracking_v1环境中进行episode采样

这个脚本展示了如何：
1. 加载v0训练的SAC模型
2. 在v1环境中使用该策略进行采样
3. 处理episode截断和数据收集
"""
import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import auv_env
from config_loader import load_config
import numpy as np
from stable_baselines3 import SAC,PPO
from auv_env.wrappers import StateOnlyWrapper
from auv_track_launcher.dataset.data_collector import AUVCollector

def get_model_class(alg_config_path):
    """根据配置文件路径或内容自动识别算法类"""
    config = load_config(alg_config_path)
    # 优先从配置文件的 'name' 字段读取，没有则从文件名判断
    alg_name = config.get('name', alg_config_path).lower()
    
    if 'ppo' in alg_name:
        print(f"检测到算法类型: PPO")
        return PPO
    elif 'sac' in alg_name:
        print(f"检测到算法类型: SAC")
        return SAC
    else:
        raise ValueError(f"无法识别算法类型: {alg_name}。请确保配置文件名包含 'sac' 或 'ppo'。")


def sample_episodes_v1_with_v0_policy(
    model_path: str,
    env_config_path: str,
    alg_config_path: str,
    n_episodes: int = 50,
    save_dir: str = "log/sample/v1_episodes",
    min_length: int = 300,
    truncate_tail: int = 100,
    show_viewport: bool = False,
    deterministic: bool = True
):
    """
    使用v0训练的SAC策略在v1环境中采样episodes
    
    Args:
        model_path: SAC模型路径（从sac.yml中的resume_path获取）
        env_config_path: v1环境配置文件路径
        alg_config_path: 算法配置文件路径（sac.yml）
        n_episodes: 要采样的episode数量
        save_dir: 数据保存目录
        min_length: episode最小长度，小于此值的episode将被舍弃
        truncate_tail: 截断尾部步数，有效episode会舍弃最后这么多步
        show_viewport: 是否显示可视化
        deterministic: 是否使用确定性策略（True=评估模式，False=探索模式）
    """
    # 1. 加载配置
    print("=" * 60)
    print("加载配置...")
    env_config = load_config(env_config_path)
    alg_config = load_config(alg_config_path)
    
    # 获取t_steps（episode最大步数）
    t_steps = env_config.get('t_steps', 1000)
    print(f"环境配置: {env_config['name']}")
    print(f"最大episode步数 (t_steps): {t_steps}")
    print(f"模型路径: {model_path}")
    print("=" * 60)
    ModelClass = get_model_class(alg_config_path)
    
    # 2. 创建环境
    print("\n创建环境...")
    env = auv_env.make(
        env_config['name'],
        config=env_config,
        eval=False,  # 评估模式，不进行训练
        t_steps=t_steps,
        show_viewport=show_viewport
    )
    
    # 3. 加载模型
    print(f"\n加载模型 ({ModelClass.__name__})...")
    # 注意：如果v0和v1的观察空间不同，需要使用StateOnlyWrapper
    # 因为SAC模型是在v0的state观察空间上训练的
    wrapped_env = StateOnlyWrapper(env)
    model = ModelClass.load(
        model_path,
        device='cuda',
        env=wrapped_env,
        custom_objects={
            'observation_space': wrapped_env.observation_space,
            'action_space': wrapped_env.action_space
        }
    )
    print("✓ 模型加载成功")
    
    # 4. 创建数据收集器
    print("\n初始化数据收集器...")
    collector = AUVCollector(
        save_dir=save_dir,
        exist_replay_path=None,
        min_length=min_length,
        truncate_tail=truncate_tail
    )
    print(f"  - 最小episode长度: {min_length}")
    print(f"  - 尾部截断步数: {truncate_tail}")
    
    # 5. 开始采样
    print(f"\n开始采样 {n_episodes} 个episodes...")
    print("=" * 60)
    
    valid_episodes = 0
    episode_lengths = []
    truncated_count = 0
    terminated_count = 0
    
    for episode in range(n_episodes):
        print(f"\nEpisode {episode + 1}/{n_episodes}")
        
        collector.start_episode()
        
        # 重置环境
        obs, info = env.reset()
        step = 0
        
        # 运行episode
        while True:
            # 使用模型预测动作（注意：需要使用state观察）
            if isinstance(obs, dict) and 'state' in obs:
                state_obs = obs['state']
            else:
                state_obs = obs
            
            action, _ = model.predict(state_obs, deterministic=deterministic)
            
            # 收集数据
            collector.add_step(obs, action)
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            
            # 检查episode是否结束
            if terminated:
                terminated_count += 1
                print(f"  Episode终止 (terminated=True) at step {step}")
                break
            elif truncated:
                truncated_count += 1
                print(f"  Episode截断 (truncated=True) at step {step} (达到最大步数 {t_steps})")
                break
        
        # 完成episode并检查是否有效
        episode_lengths.append(step)
        flag = collector.finish_episode()
        if flag:
            valid_episodes += 1
            print(f"  ✓ Episode有效，长度: {step} 步")
        else:
            print(f"  ✗ Episode被舍弃，长度: {step} 步 (小于最小长度 {min_length})")
        
        # 定期保存
        if valid_episodes > 0 and valid_episodes % 50 == 0:
            collector.save_data(f"auv_data_partial_{valid_episodes}.zarr")
            print(f"  → 已保存中间数据 ({valid_episodes} 个有效episodes)")
    
    # 6. 保存最终数据
    print("\n" + "=" * 60)
    print("采样完成，保存数据...")
    collector.save_data("auv_data_final.zarr")
    
    # 7. 统计信息
    print("\n" + "=" * 60)
    print("采样统计信息:")
    print(f"  总episodes: {n_episodes}")
    print(f"  有效episodes: {valid_episodes} ({valid_episodes/n_episodes*100:.1f}%)")
    print(f"  舍弃episodes: {n_episodes - valid_episodes}")
    print(f"  通过terminated结束: {terminated_count}")
    print(f"  通过truncated结束: {truncated_count}")
    if episode_lengths:
        print(f"  Episode长度统计:")
        print(f"    - 平均: {np.mean(episode_lengths):.1f} 步")
        print(f"    - 最小: {np.min(episode_lengths)} 步")
        print(f"    - 最大: {np.max(episode_lengths)} 步")
        print(f"    - 中位数: {np.median(episode_lengths):.1f} 步")
    
    print("=" * 60)
    env.close()
    
    return collector


def analyze_episode_truncation(
    env_config_path: str,
    model_path: str,
    alg_config_path: str,
    n_test_episodes: int = 10
):
    """
    分析episode截断的合理性
    
    这个函数会运行一些测试episodes，分析：
    1. 有多少episodes因为达到最大步数而被截断
    2. 平均episode长度
    3. 截断是否合理
    """
    print("\n" + "=" * 60)
    print("分析Episode截断合理性...")
    print("=" * 60)
    
    env_config = load_config(env_config_path)
    alg_config = load_config(alg_config_path)
    ModelClass = get_model_class(alg_config_path)
    t_steps = env_config.get('t_steps', 1000)
    
    env = auv_env.make(
        env_config['name'],
        config=env_config,
        eval=True,
        t_steps=t_steps,
        show_viewport=False
    )
    
    wrapped_env = StateOnlyWrapper(env)
    model = ModelClass.load(
        model_path,
        device='cuda',
        env=wrapped_env,
        custom_objects={
            'observation_space': wrapped_env.observation_space,
            'action_space': wrapped_env.action_space
        }
    )
    
    episode_lengths = []
    truncated_episodes = 0
    terminated_episodes = 0
    
    for episode in range(n_test_episodes):
        obs, info = env.reset()
        step = 0
        
        while True:
            if isinstance(obs, dict) and 'state' in obs:
                state_obs = obs['state']
            else:
                state_obs = obs
            
            action, _ = model.predict(state_obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            
            if terminated:
                terminated_episodes += 1
                break
            elif truncated:
                truncated_episodes += 1
                break
        
        episode_lengths.append(step)
        print(f"Episode {episode + 1}: {step} 步 ({'截断' if step >= t_steps else '正常结束'})")
    
    env.close()
    
    # 分析结果
    print("\n" + "-" * 60)
    print("截断分析结果:")
    print(f"  测试episodes: {n_test_episodes}")
    print(f"  平均长度: {np.mean(episode_lengths):.1f} 步")
    print(f"  最大步数限制: {t_steps} 步")
    print(f"  截断比例: {truncated_episodes/n_test_episodes*100:.1f}% ({truncated_episodes}/{n_test_episodes})")
    print(f"  正常结束比例: {terminated_episodes/n_test_episodes*100:.1f}% ({terminated_episodes}/{n_test_episodes})")
    
    # 建议
    avg_length = np.mean(episode_lengths)
    if truncated_episodes / n_test_episodes > 0.5:
        print(f"\n⚠️  警告: 超过50%的episodes被截断！")
        print(f"   建议: 考虑增加t_steps（当前{t_steps}）或检查策略性能")
    elif avg_length < t_steps * 0.3:
        print(f"\n💡 提示: 平均episode长度 ({avg_length:.1f}) 远小于最大步数 ({t_steps})")
        print(f"   建议: 可以考虑减小t_steps以加快训练/评估速度")
    else:
        print(f"\n✓ 截断设置合理: 平均长度 {avg_length:.1f} 步，最大步数 {t_steps} 步")
    
    print("=" * 60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='使用v0训练的SAC策略在v1环境中采样episodes')
    parser.add_argument('--model_path', type=str,
                       default='log/AUVTracking_v0/PID/SAC/12-20_01/rl_model_999990_steps.zip',
                       help='SAC模型路径（从sac.yml的resume_path获取）')
    parser.add_argument('--env_config', type=str,
                       default='configs/envs/v1_config.yml',
                       help='v1环境配置文件路径')
    parser.add_argument('--alg_config', type=str,
                       default='configs/algorithm/sac.yml',
                       help='算法配置文件路径')
    parser.add_argument('--n_episodes', type=int, default=50,
                       help='要采样的episode数量')
    parser.add_argument('--save_dir', type=str,
                       default='log/sample/v1_episodes',
                       help='数据保存目录')
    parser.add_argument('--min_length', type=int, default=300,
                       help='episode最小长度，小于此值的episode将被舍弃')
    parser.add_argument('--truncate_tail', type=int, default=100,
                       help='截断尾部步数，有效episode会舍弃最后这么多步')
    parser.add_argument('--show_viewport', action='store_true',
                       help='是否显示可视化')
    parser.add_argument('--analyze', action='store_true',
                       help='先分析episode截断合理性（运行少量测试episodes）')
    parser.add_argument('--deterministic', action='store_true', default=True,
                       help='是否使用确定性策略（默认True）')
    
    args = parser.parse_args()
    
    # 如果指定了analyze，先进行分析
    if args.analyze:
        analyze_episode_truncation(
            args.env_config,
            args.model_path,
            args.alg_config,
            n_test_episodes=10
        )
        print("\n是否继续采样？(y/n): ", end='')
        response = input().strip().lower()
        if response != 'y':
            exit(0)
    
    # 执行采样
    sample_episodes_v1_with_v0_policy(
        model_path=args.model_path,
        env_config_path=args.env_config,
        alg_config_path=args.alg_config,
        n_episodes=args.n_episodes,
        save_dir=args.save_dir,
        min_length=args.min_length,
        truncate_tail=args.truncate_tail,
        show_viewport=args.show_viewport,
        deterministic=args.deterministic
    )

