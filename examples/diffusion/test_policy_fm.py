import torch
import os
import sys
import pathlib


import numpy as np
import gymnasium as gym
import torchvision.transforms.functional as TF

# --- 1. 添加项目路径以导入模块 ---
ROOT = pathlib.Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from diffusion_policy.workspace.train_flow_matching_unet_image_workspace_custom import \
    TrainFlowMatchingUnetImageWorkspace
from auv_env.envs.tools import ImageBuffer
import auv_env

def load_fm_policy(ckpt_path, device="cuda"):
    """
    加载 Flow Matching 策略
    """
    print(f"Loading checkpoint: {ckpt_path}")
    # weights_only=False 是必须的，因为 checkpoint 包含 OmegaConf 配置对象
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = payload["cfg"]
    
    # 使用 Workspace 恢复模型，这样会自动处理网络结构参数
    workspace = TrainFlowMatchingUnetImageWorkspace(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # 获取策略模型
    policy = workspace.model
    if getattr(workspace, "ema_model", None) is not None:
        policy = workspace.ema_model
    
    policy.to(device)
    policy.eval()
    
    # 确保 Normalizer 被正确加载
    policy.set_normalizer(workspace.normalizer)
    
    # 如果之前遇到过 crop_shape 不匹配的问题，可以在这里读取配置
    crop_shape = None
    try:
        crop_shape = cfg.policy.obs_encoder.crop_shape
    except:
        pass
        
    return policy, crop_shape

def main():
    # --- 配置 ---
    # 替换为你训练好的 .ckpt 文件路径
    ckpt_path = "data/outputs/2025.12.24/01.27.34_train_flow_matching_unet_image_track_image/checkpoints/epoch=0005-val_loss=0.2480.ckpt"
    device = "cuda"
    
    # --- 2. 加载模型 ---
    policy, crop_shape = load_fm_policy(ckpt_path, device)
    
    # --- 3. 环境设置 ---
    # 确保环境名字正确，例如 'v1-Student-sample' 或 'AUVTracking-v1'
    # 注意：如果你的环境配置在 yaml 里，可能需要用 auv_env.make 加载 config
    env = gym.make('v1-Student-sample') 
    
    # 初始化 Buffer
    # 注意：Flow Matching 训练时通常有 n_obs_steps (例如 2 或 5)
    n_obs_steps = policy.n_obs_steps
    print(f"Policy expects n_obs_steps: {n_obs_steps}")
    
    # 假设图像大小是 (3, H, W)，这里根据你的环境实际输出调整
    # 如果环境输出是 (H, W, 3)，ImageBuffer 需要相应调整
    image_buffer = ImageBuffer(n_obs_steps, (3, 224, 224), time_gap=0.5)

    obs, _ = env.reset()
    # 假设 obs['images'] 是 (H, W, C) 或 (C, H, W)，确保放入 buffer 的格式一致
    # 这里假设环境返回的是 numpy array
    image_buffer.add_image(obs['images'], 0.0)

    # 预填充 buffer (如果需要)
    for _ in range(n_obs_steps - 1):
        image_buffer.add_image(obs['images'], 0.0)

    print("Start inference loop...")
    for step in range(500):
        # --- 4. 数据预处理 ---
        # 获取历史图像: (T, C, H, W) 或 (T, H, W, C)
        images_np = np.stack(image_buffer.get_buffer()) 
        
        # 转 Tensor
        obs_tensor = torch.from_numpy(images_np).float()
        
        # 如果环境输出是 (T, H, W, C)，转为 (T, C, H, W)
        if obs_tensor.shape[-1] == 3:
            obs_tensor = obs_tensor.permute(0, 3, 1, 2)
            
        # 调整尺寸 (Resize) - 只有当环境图像尺寸与模型训练时不一致时才需要
        # 注意：Flow Matching Policy 内部通常有 RandomCrop，推理时可能需要 CenterCrop
        # 如果之前遇到维度报错，这里手动 Crop 到训练尺寸 (例如 60x60)
        if crop_shape is not None:
             obs_tensor = TF.center_crop(obs_tensor, crop_shape)
        
        # 增加 Batch 维度: (1, T, C, H, W)
        obs_tensor = obs_tensor.unsqueeze(0).to(device)
        
        # 构建观测字典
        # 键名必须与训练时一致，通常是 'camera_image' 和 'state'
        obs_dict = {
            'camera_image': obs_tensor
        }
        
        # 如果模型还需要 state 输入
        if 'state' in obs:
            state_tensor = torch.from_numpy(obs['state']).float().to(device)
            # (1, T, D)
            state_tensor = state_tensor.unsqueeze(0).unsqueeze(0).repeat(1, n_obs_steps, 1)
            obs_dict['state'] = state_tensor

        # --- 5. 推理 (Flow Matching) ---
        with torch.no_grad():
            # 直接调用 predict_action，它会自动处理归一化和 ODE 求解
            result = policy.predict_action(obs_dict)
            action_pred = result['action'] # (B, Horizon, ActionDim)
            
            # 取出当前步的动作
            # 通常取第一个时间步的动作，或者根据 Receding Horizon Control 策略取值
            action = action_pred[0, 0, :].cpu().numpy()

        # --- 6. 环境交互 ---
        obs, reward, dones, _, inf = env.step(action)
        
        # 更新 Buffer
        image_buffer.add_image(obs['images'], 0.0)
        
        if dones:
            print(f"Episode finished at step {step}")
            break
            
    print("Done.")

if __name__ == "__main__":
    main()