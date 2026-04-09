import os
import sys
import pathlib
import argparse
import numpy as np
import zmq  # 使用 ZeroMQ 进行通信
import json

# 添加项目路径
ROOT = pathlib.Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import auv_env
from config_loader import load_config


def _zero_action_for_env(env):
    shape = getattr(env.action_space, "shape", None)
    if shape is None or shape == ():
        return np.zeros((), dtype=np.float32)
    return np.zeros(shape, dtype=np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_config", type=str, required=True)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--show_viewport", type=int, default=1)
    args = parser.parse_args()

    # 1. 初始化 ZeroMQ Server
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{args.port}")
    print(f"[Server] HoloOcean Server started on port {args.port}")

    # 2. 初始化环境
    cfg = load_config(args.env_config)
    env = auv_env.make(
        cfg["name"],
        config=cfg,
        eval=True,
        show_viewport=bool(args.show_viewport),
    )
    print("[Server] Environment initialized")

    obs, info = env.reset()
    
    while True:
        # 3. 等待 Client 请求
        # 协议: 接收 "RESET" 或 动作数组
        message = socket.recv()
        
        try:
            request = json.loads(message)
        except:
            # 假设是二进制数据或其他格式，这里简化处理
            request = {"cmd": "UNKNOWN"}

        cmd = request.get("cmd")
        warmup_steps = int(request.get("warmup_steps", 0) or 0)
        warmup_max_steps = int(request.get("warmup_max_steps", 0) or 0)
        freeze_agent_steps_req = request.get("freeze_agent_steps", None)
        freeze_target_steps_req = request.get("freeze_target_steps", None)

        response = {}

        if cmd == "RESET":
            print("[Server] Resetting environment...")
            obs, info = env.reset()

            # 冻结步数默认与 warmup_steps 一致；也支持客户端显式传入覆盖
            if freeze_agent_steps_req is None:
                freeze_agent_steps = max(warmup_steps, 0)
            else:
                freeze_agent_steps = max(int(freeze_agent_steps_req), 0)
            if freeze_target_steps_req is None:
                freeze_target_steps = max(warmup_steps, 0)
            else:
                freeze_target_steps = max(int(freeze_target_steps_req), 0)

            env_base = getattr(env, "unwrapped", env)
            if hasattr(env_base, "set_freeze_steps"):
                env_base.set_freeze_steps(
                    freeze_agent_steps=freeze_agent_steps,
                    freeze_target_steps=freeze_target_steps,
                )
                print(
                    "[Server] Freeze control: "
                    f"agent_steps={freeze_agent_steps}, target_steps={freeze_target_steps}"
                )

            if warmup_steps > 0 or warmup_max_steps > 0:
                zero_action = _zero_action_for_env(env)
                warmup_count = 0
                target_steps = max(warmup_steps, 0)
                max_steps = max(warmup_max_steps, target_steps)
                print(
                    "[Server] Warmup: "
                    f"min_steps={target_steps}, max_steps={max_steps}"
                )

                while warmup_count < max_steps:
                    need_more = warmup_count < target_steps

                    if not need_more:
                        break

                    obs, reward, terminated, truncated, info = env.step(zero_action)
                    warmup_count += 1

                    if terminated or truncated:
                        print("[Server] Episode ended during warmup.")
                        break
                print(f"[Server] Warmup done at step {warmup_count}")
            response["status"] = "ok"
            response["done"] = False
            response["reward"] = 0.0
        
        elif cmd == "STEP":
            action = np.array(request["action"])
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            response["status"] = "ok"
            response["reward"] = float(reward)
            response["done"] = bool(done)
            response["info"] = str(info) # 简化 info
        
        elif cmd == "CLOSE":
            socket.send_json({"status": "closed"})
            break

        # 4. 打包观测数据 (Obs)
        obs_data = {}
        
        # [修复] 检查 'image' 键 (WorldAuvV1 使用 'image')
        if "camera_image" in obs:
            obs_data["camera_image"] = obs["camera_image"].tolist()
            obs_data["camera_shape"] = obs["camera_image"].shape
        elif "image" in obs:
            # 将 'image' 映射为 'camera_image' 发送，或者保留原名
            # 这里为了兼容性，我们保留原名，让 Client 去处理重命名
            obs_data["image"] = obs["image"].tolist()
            obs_data["camera_shape"] = obs["image"].shape
        
        if "state" in obs:
            obs_data["state"] = obs["state"].tolist()

        response["obs"] = obs_data
        
        # 发送回 Client
        socket.send_json(response)

    env.close()
    print("[Server] Closed.")

if __name__ == "__main__":
    main()