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


def resolve_runtime_device(requested_device: str) -> str:
    req = (requested_device or "auto").strip().lower()
    if req == "auto":
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "cuda"
        return "cpu"
    if req.startswith("cuda"):
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return requested_device
        print("[Client] CUDA requested but unavailable. Fallback to CPU.")
        return "cpu"
    return requested_device

def load_flow_matching_policy(ckpt_path: str, device: str = "cuda"):
    resolved_device = resolve_runtime_device(device)
    print(f"[Client] Loading checkpoint to RAM...")
    payload = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'pickles' not in payload:
        payload['pickles'] = {}
    if 'state_dicts' not in payload:
        print("[Client] Detected shrunk checkpoint, rebuilding state_dicts structure...")
        payload['state_dicts'] = {}
        for key in ['model', 'ema_model', 'optimizer']:
            if key in payload:
                payload['state_dicts'][key] = payload.pop(key)

    cfg = payload["cfg"]
    workspace = TrainFlowMatchingUnetImageWorkspace(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    if hasattr(workspace.model, 'normalizer'):
        normalizer = workspace.model.normalizer
    elif hasattr(workspace, 'normalizer'):
        normalizer = workspace.normalizer
    else:
        print("[Client] Normalizer not found in workspace, creating a dummy or checking payload...")
        normalizer = None # 如果这里还是 None，后续 unnormalize 会报错，需检查训练代码
    
    policy = workspace.model
    if getattr(workspace, "ema_model", None) is not None:
        policy = workspace.ema_model

    print(f"[Client] Moving policy to {resolved_device}...")
    try:
        policy.to(resolved_device)
    except RuntimeError as e:
        if resolved_device.startswith("cuda"):
            print(f"[Client] Failed to init CUDA ({e}). Fallback to CPU.")
            resolved_device = "cpu"
            policy.to(resolved_device)
        else:
            raise
    policy.eval()
    # ------------------ 在这里插入检查代码 ------------------
    print("-" * 30)
    print("[Debug] Checking model weights...")
    try:
        # 1. 尝试获取第一层卷积权重
        conv1_weight = policy.obs_encoder.key_model_map.camera_image.conv1.weight
        
        # 打印前 5 个数值
        print("Conv1 weight sample (first 5):", conv1_weight.flatten()[:5].detach().cpu().numpy())
        
        # 打印均值和标准差
        print(f"Conv1 weight mean: {conv1_weight.mean().item():.6f}")
        print(f"Conv1 weight std: {conv1_weight.std().item():.6f}")

        # 检查是否为 NaN
        if torch.isnan(conv1_weight).any():
            print("!!! 警告：模型参数中存在 NaN !!!")
            
        # 2. 验证整体参数量
        total_params = sum(p.numel() for p in policy.parameters())
        print(f"Total parameters in policy: {total_params}")
        
    except Exception as e:
        print(f"[Debug] 访问权重时出错: {e}")
        # 如果路径不对，打印出所有的 key 供参考
        # print("Available keys:", policy.state_dict().keys())
    print("-" * 30)
    # ------------------------------------------------------
    policy.obs_encoder.eval()

    if hasattr(policy.obs_encoder, 'key_transform_map'):
        for key, transform in policy.obs_encoder.key_transform_map.items():
            if isinstance(transform, CropRandomizer):
                policy.obs_encoder.key_transform_map[key] = nn.Identity()
            elif isinstance(transform, nn.Sequential):
                for i, module in enumerate(transform):
                    if isinstance(module, CropRandomizer):
                        transform[i] = nn.Identity()

    if hasattr(policy.obs_encoder, 'key_model_map'):
        for key, model in policy.obs_encoder.key_model_map.items():
            if hasattr(model, 'avgpool') and not isinstance(model.avgpool, nn.Identity):
                model.avgpool = nn.Identity()
            if hasattr(model, 'fc') and not isinstance(model.fc, nn.Identity):
                model.fc = nn.Identity()
    # 统计总参数量
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"Total parameters in policy: {total_params}")

    # 统计实际加载的权重 key
    if total_params > 0:
        print("Weight keys example:", list(policy.state_dict().keys())[:5])
    else:
        print("!!! 错误：模型参数量为 0，请检查模型类定义 !!!")
        print(f"[Client] Policy loaded successfully.")
    return policy, cfg, normalizer, resolved_device


class RemoteEnv:
    def __init__(self, port=5555):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://localhost:{port}")
        print(f"[Client] Connected to HoloOcean Server on port {port}")

    def reset(self, warmup_steps=0, warmup_max_steps=0):
        self.socket.send_json({
            "cmd": "RESET",
            "warmup_steps": int(warmup_steps),
            "warmup_max_steps": int(warmup_max_steps),
        })
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

def run_eval(
    ckpt_path,
    num_episodes=5,
    device="cuda",
    port=5555,
    short_exec_steps=3,
    warmup_steps=5,
    warmup_max_steps=5,
):
    policy, cfg, _normalizer, runtime_device = load_flow_matching_policy(ckpt_path, device)
    n_obs_steps = cfg.n_obs_steps
    env = RemoteEnv(port=port)
    obs_deque = collections.deque(maxlen=n_obs_steps)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f"data/debug_images/{timestamp}"
    os.makedirs(save_path, exist_ok=True)

    for ep in tqdm(range(num_episodes), desc="Remote FM Eval"):
        obs = env.reset(
            warmup_steps=warmup_steps,
            warmup_max_steps=warmup_max_steps,
        )
        obs_deque.clear()
        for _ in range(n_obs_steps):
            obs_deque.append(obs)

        done = False
        ep_ret = 0.0
        ep_len = 0

        while not done:
            obs_dict = {}
            imgs = []
            for o in obs_deque:
                img = o.get("camera_image") if "camera_image" in o else o.get("image")
                t = torch.from_numpy(img).float()
                if t.shape[-1] == 3 and t.shape[0] != 3:
                    t = t.permute(2, 0, 1)
                if t.max() > 1.0:
                    t = t / 255.0          # 先到 [0, 1]
                    # t = t * 2.0 - 1.0      # 再到 [-1, 1] (这一步之前漏掉了！)
                t = TF.resize(t, [64, 64])
                imgs.append(t)
            img_batch = torch.stack(imgs).unsqueeze(0).to(runtime_device)
            if img_batch.shape[2] != 3 and img_batch.shape[-1] == 3:
                img_batch = img_batch.permute(0, 1, 4, 2, 3)
            obs_dict["camera_image"] = img_batch

            states = [torch.from_numpy(o["state"]).float() for o in obs_deque]
            obs_dict["state"] = torch.stack(states).unsqueeze(0).to(runtime_device)

            with torch.no_grad():
                # 在 policy.predict_action 之前执行
                # import cv2
                # img_for_save = img_batch[0, -1].cpu().permute(1, 2, 0).numpy()
                # img_for_save = (img_for_save * 255).astype(np.uint8)
                # cv2.imwrite("what_model_sees.png", cv2.cvtColor(img_for_save, cv2.COLOR_RGB2BGR))
                act_dict = policy.predict_action(obs_dict)
                if ep_len % 5 == 1:
                    import torchvision.utils as vutils
                    img_to_save = obs_dict["camera_image"][0, -1]
                    vutils.save_image(img_to_save, f"{save_path}/ep{ep}_step{ep_len}.png")

                # 短段执行：每次推理后，连续执行前 K 个动作，再重新感知并重规划
                action_chunk = act_dict['action'][0].cpu().numpy()

            exec_steps = min(short_exec_steps, action_chunk.shape[0])
            for k in range(exec_steps):
                action = action_chunk[k]
                print(f"Episode {ep} Step {ep_len} ChunkStep {k} Action: {action}")
                obs, reward, done = env.step(action)
                obs_deque.append(obs)
                ep_ret += reward
                ep_len += 1
                if done:
                    break

        print(f"Episode {ep} Return: {ep_ret:.2f}")

    env.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--device", type=str, default="auto",
                        help="推理设备: auto/cpu/cuda/cuda:0")
    parser.add_argument("--short_exec_steps", type=int, default=2,
                        help="每次推理后连续执行的动作步数")
    parser.add_argument("--warmup_steps", type=int, default=15,
                        help="reset 后先用零动作空转的步数")
    parser.add_argument("--warmup_max_steps", type=int, default=15,
                        help="预热允许的最大空转步数")
    args = parser.parse_args()
    run_eval(args.ckpt, port=args.port, device=args.device,
             short_exec_steps=max(1, args.short_exec_steps),
             warmup_steps=max(0, args.warmup_steps),
             warmup_max_steps=max(0, args.warmup_max_steps))

if __name__ == "__main__":
    main()