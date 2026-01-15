import os
import sys
import pathlib
import argparse
import torch
import torch.nn as nn
import numpy as np
import zmq
import collections
from datetime import datetime
from tqdm import tqdm
import torchvision.transforms.functional as TF

# 1. 设置路径
ROOT = pathlib.Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# 2. 导入项目模块
from diffusion_policy.workspace.train_flow_matching_unet_image_workspace_custom import \
    TrainFlowMatchingUnetImageWorkspace
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer

def load_flow_matching_policy(ckpt_path: str, device: str = "cuda"):
    print(f"[Client] Loading checkpoint to RAM...")
    payload = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    # --- 1. 修复 'pickles' 缺失 ---
    if 'pickles' not in payload:
        payload['pickles'] = {}

    # --- 2. 修复 'state_dicts' 缺失 (核心修复) ---
    # 如果 payload 里没有 state_dicts，但有 model 或 ema_model
    # 说明这是个瘦身过的文件，我们需要手动包装一层
    if 'state_dicts' not in payload:
        print("[Client] Detected shrunk checkpoint, rebuilding state_dicts structure...")
        payload['state_dicts'] = {}
        
        # 寻找可能的权重键名并移入 state_dicts
        for key in ['model', 'ema_model', 'optimizer']:
            if key in payload:
                payload['state_dicts'][key] = payload.pop(key)
    
    # --- 3. 初始化并加载 ---
    cfg = payload["cfg"]
    workspace = TrainFlowMatchingUnetImageWorkspace(cfg)
    
    # 现在 payload 已经有了 pickles 和 state_dicts，load_payload 不会再报错
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    policy = workspace.model
    if getattr(workspace, "ema_model", None) is not None:
        policy = workspace.ema_model
        print("[Client] Using EMA weights.")
    
    print(f"[Client] Moving policy to {device}...")
    policy.to(device)
    policy.eval()
    
    # 保持原有的 encoder 模式
    policy.obs_encoder.train()
    
    # 禁用 CropRandomizer
    if hasattr(policy.obs_encoder, 'key_transform_map'):
        for key, transform in policy.obs_encoder.key_transform_map.items():
            if isinstance(transform, CropRandomizer):
                policy.obs_encoder.key_transform_map[key] = nn.Identity()
            elif isinstance(transform, nn.Sequential):
                for i, module in enumerate(transform):
                    if isinstance(module, CropRandomizer):
                        transform[i] = nn.Identity()

    print(f"[Client] Policy loaded successfully.")
    return policy, cfg

class RemoteEnv:
    def __init__(self, port=5555):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://localhost:{port}")
        print(f"[Client] Connected to HoloOcean Server on port {port}")

    def reset(self):
        self.socket.send_json({"cmd": "RESET"})
        resp = self.socket.recv_json()
        return self._parse_obs(resp["obs"])

    def step(self, action):
        self.socket.send_json({
            "cmd": "STEP",
            "action": action.tolist()
        })
        resp = self.socket.recv_json()
        obs = self._parse_obs(resp["obs"])
        return obs, resp["reward"], resp["done"]

    def close(self):
        self.socket.send_json({"cmd": "CLOSE"})
        self.socket.recv_json()

    def _parse_obs(self, obs_dict):
        parsed = {}
        img_raw = obs_dict.get("camera_image") or obs_dict.get("image")
        if img_raw is not None:
            parsed["camera_image"] = np.array(img_raw, dtype=np.uint8)
        if "state" in obs_dict:
            parsed["state"] = np.array(obs_dict["state"], dtype=np.float32)
        return parsed

def run_eval(ckpt_path, num_episodes=5, device="cuda", port=5555):
    policy, cfg = load_flow_matching_policy(ckpt_path, device)
    
    # 【关键】从配置获取 n_obs_steps，通常是 2
    n_obs_steps = cfg.n_obs_steps 
    env = RemoteEnv(port=port)

    # 【关键】使用队列维持 2 帧观察，解决 mat1/mat2 维度不匹配问题
    obs_deque = collections.deque(maxlen=n_obs_steps)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f"data/debug_images/{timestamp}"
    os.makedirs(save_path, exist_ok=True)

    for ep in tqdm(range(num_episodes), desc="Remote FM Eval"):
        obs = env.reset()
        # 初始化队列
        for _ in range(n_obs_steps):
            obs_deque.append(obs)
            
        done = False
        ep_ret = 0.0
        ep_len = 0

        while not done:
            obs_dict = {}
            
            # 1. 图像处理 (B, T, C, H, W)
            imgs = []
            for o in obs_deque:
                # 安全获取原始图像 (H, W, 3)
                img = o.get("camera_image") if "camera_image" in o else o.get("image")
                
                # numpy (H, W, 3) -> torch (H, W, 3)
                t = torch.from_numpy(img).float()
                
                # --- 核心自校正逻辑 ---
                # 如果通道维 (3) 在最后，则进行 permute
                if t.shape[-1] == 3 and t.shape[0] != 3:
                    t = t.permute(2, 0, 1) # 变为 (3, H, W)
                
                # 归一化到 [0, 1] (重要：ResNet 标准输入)
                if t.max() > 1.0:
                    t = t / 255.0
                
                # 缩放到训练尺寸 64x64
                t = TF.resize(t, [64, 64])
                imgs.append(t)
            
            # 堆叠并增加 Batch 维度 -> (1, 2, 3, 64, 64)
            img_batch = torch.stack(imgs).unsqueeze(0).to(device)
            
            # --- 最终形状保险锁 ---
            # 期望形状是 (Batch=1, Time=2, Channel=3, Height=64, Width=64)
            # 如果维度 2 不是 3，说明 Layout 依然是 HWC，强制纠正
            if img_batch.shape[2] != 3 and img_batch.shape[-1] == 3:
                # 假设当前是 (1, 2, 64, 64, 3) -> 转为 (1, 2, 3, 64, 64)
                img_batch = img_batch.permute(0, 1, 4, 2, 3)
            
            obs_dict["camera_image"] = img_batch

            # 2. 状态处理 (1, 2, D)
            states = [torch.from_numpy(o["state"]).float() for o in obs_deque]
            obs_dict["state"] = torch.stack(states).unsqueeze(0).to(device)



            with torch.no_grad():
                act_dict = policy.predict_action(obs_dict)
                
                # 每10步保存一次图像用于调试
                if ep_len % 5 == 1:
                    import torchvision.utils as vutils
                    img_to_save = obs_dict["camera_image"][0, -1] # 保存最新的一帧
                    vutils.save_image(img_to_save, f"{save_path}/ep{ep}_step{ep_len}.png")

                action_chunk = act_dict['action'][0, :4].cpu().numpy()

            # --- 3. 执行动作块 (Action Chunking) ---
            # 这里的逻辑是：预测一次，连续与环境交互 4次
            step_count = 0
            for i in range(len(action_chunk)):
                action = action_chunk[i]
                action = -action  # 反向动作修正
                # 执行当前这步动作
                obs, reward, done = env.step(action)
                
                # 重要：每走一步都要更新观察队列，确保下一次预测能拿到最新的 2 帧
                obs_deque.append(obs)
                
                ep_ret += reward
                step_count += 1
                
                # 如果在执行 8 步的过程中环境已经结束（如撞墙或跟丢），提前跳出
                if done:
                    break
            
            # 8 步走完后，回到循环开头，进行下一次模型推理
            obs_deque.append(obs)
            ep_ret += reward
            ep_len += 1

        print(f"Episode {ep} Return: {ep_ret:.2f}")

    env.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    run_eval(args.ckpt, port=args.port, device=args.device)

if __name__ == "__main__":
    main()