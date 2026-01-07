import os
import sys
import pathlib
import argparse
import torch
import torch.nn as nn  # 确保导入 nn
import numpy as np
import zmq
import json
from datetime import datetime
from tqdm import tqdm
import torchvision.transforms.functional as TF

# 1. 先设置路径
ROOT = pathlib.Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# 2. 再导入项目模块
from diffusion_policy.workspace.train_flow_matching_unet_image_workspace_custom import \
    TrainFlowMatchingUnetImageWorkspace
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer # 移到这里

def load_flow_matching_policy(ckpt_path: str, device: str = "cuda"):
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = payload["cfg"]
    workspace = TrainFlowMatchingUnetImageWorkspace(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    policy = workspace.model
    if getattr(workspace, "ema_model", None) is not None:
        policy = workspace.ema_model
    
    policy.to(device)
    
    # 1. 整体设为 eval
    policy.eval()
    
    # 2. 强制 obs_encoder 为 train 模式 (保持 Dropout/BN 行为一致)
    policy.obs_encoder.train()
    
    # 3. [修复] 禁用 CropRandomizer
    if hasattr(policy.obs_encoder, 'key_transform_map'):
        for key, transform in policy.obs_encoder.key_transform_map.items():
            if isinstance(transform, CropRandomizer):
                print(f"[Client] Disabling CropRandomizer for key: {key} (Direct)")
                policy.obs_encoder.key_transform_map[key] = nn.Identity()
            elif isinstance(transform, nn.Sequential):
                for i, module in enumerate(transform):
                    if isinstance(module, CropRandomizer):
                        print(f"[Client] Disabling CropRandomizer for key: {key} (Inside Sequential at index {i})")
                        transform[i] = nn.Identity()

    # 4. [关键修复] 强制移除 ResNet 的 Pooling 层
    # 这一步是为了解决 640 vs 8320 的维度不匹配问题
    if hasattr(policy.obs_encoder, 'key_model_map'):
        for key, model in policy.obs_encoder.key_model_map.items():
            print(f"[Client] Inspecting model for key: {key}")
            
            # 检查并替换 avgpool
            if hasattr(model, 'avgpool'):
                if not isinstance(model.avgpool, nn.Identity):
                    print(f"[Client] Found active avgpool in {key}, replacing with Identity to restore spatial features.")
                    model.avgpool = nn.Identity()
            
            # 检查并替换 fc
            if hasattr(model, 'fc'):
                if not isinstance(model.fc, nn.Identity):
                    print(f"[Client] Found active fc in {key}, replacing with Identity.")
                    model.fc = nn.Identity()

    print("[Client] Policy loaded with forced train mode, disabled random crop, and removed pooling.")
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
        return self._parse_obs(resp["obs"]), {}

    def step(self, action):
        self.socket.send_json({
            "cmd": "STEP",
            "action": action.tolist()
        })
        resp = self.socket.recv_json()
        obs = self._parse_obs(resp["obs"])
        return obs, resp["reward"], resp["done"], False, {} # Truncated=False

    def close(self):
        self.socket.send_json({"cmd": "CLOSE"})
        self.socket.recv_json()

    def _parse_obs(self, obs_dict):
        # 还原 numpy 数组
        parsed = {}
        # [修复] 同时处理 'camera_image' 和 'image'
        if "camera_image" in obs_dict:
            parsed["camera_image"] = np.array(obs_dict["camera_image"], dtype=np.uint8)
        elif "image" in obs_dict:
            parsed["image"] = np.array(obs_dict["image"], dtype=np.uint8)
            
        if "state" in obs_dict:
            parsed["state"] = np.array(obs_dict["state"], dtype=np.float32)
        return parsed

def run_eval(ckpt_path, num_episodes=5, device="cuda", port=5555):
    policy, cfg = load_flow_matching_policy(ckpt_path, device) # 获取 cfg
    
    # 从配置中读取 crop_shape
    # 假设 cfg 结构是 hydra 的 DictConfig
    try:
        crop_shape = cfg.policy.obs_encoder.crop_shape
        print(f"[Client] Using crop_shape from config: {crop_shape}")
    except:
        crop_shape = [60, 60] # 默认值
        print(f"[Client] Config not found, using default crop_shape: {crop_shape}")

    env = RemoteEnv(port=port)

    returns = []
    lengths = []

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(f"data/debug_images/{timestamp}", exist_ok=True)
    save_path = f"data/debug_images/{timestamp}"

    for ep in tqdm(range(num_episodes), desc="Remote FM Eval"):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        ep_len = 0

        while not done:
            obs_dict = {}
            
            # [修复] 检查 'image' 键并重命名为 'camera_image'
            # 环境返回的是 'image'，但模型需要 'camera_image'
            img = None
            if "camera_image" in obs:
                img = obs["camera_image"]
            elif "image" in obs:
                img = obs["image"]
            
            if img is not None:
                if img.ndim == 3 and img.shape[-1] in (1, 3):
                    img = np.transpose(img, (2, 0, 1))
                
                # [关键修复] 手动 CenterCrop 到 60x60
                # 先转 Tensor
                img_tensor = torch.from_numpy(img).float() # (C, H, W)
                
                # # 执行 CenterCrop
                # if crop_shape is not None:
                #     img_tensor = TF.center_crop(img_tensor, crop_shape)

                target_shape = [128, 128] 
                # print(f"[Client] Forcing crop shape to {target_shape} to match weight matrix (8320 dim)")
                img_tensor = TF.resize(img_tensor, target_shape)
                
                # 增加 Batch 和 Horizon 维度: (1, 1, C, H, W)
                img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)
                
                obs_dict["camera_image"] = img_tensor

            if "state" in obs:
                st = obs["state"]
                st = st[None, None, ...]
                obs_dict["state"] = torch.from_numpy(st).to(device).float()

            with torch.no_grad():
                act_dict = policy.predict_action(obs_dict)
                # 在 act_dict = policy.predict_action(obs_dict) 之后添加
                # 定时保存图片
                if ep_len % 10 == 1: # 每50步保存一张图
                    import torchvision.utils as vutils
                    # 提取并保存送入模型的图像
                    img_to_save = obs_dict["camera_image"][0, 0] # 取出 (C, H, W)
                    # 保存为 PNG 文件，到目录data/debug_images
                    
                    vutils.save_image(img_to_save / 255.0, f"{save_path}/_ep{ep}_step{ep_len}.png")

                action = act_dict["action"][0, 0].cpu().numpy()

            obs, reward, done, _, _ = env.step(action)
            ep_ret += reward
            ep_len += 1

        returns.append(ep_ret)
        lengths.append(ep_len)

    env.close()
    print(f"Mean Return: {np.mean(returns):.2f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()
    run_eval(args.ckpt, port=args.port)

if __name__ == "__main__":
    main()