#!/bin/bash

# 配置路径
CKPT="data/outputs/2026-04-08/11-52-14/checkpoints/epoch=0010-train_loss=0.087.ckpt"
ENV_CONFIG="configs/envs/v1_config.yml"
PORT=5555

# 1. 启动 Server (HoloOcean 环境)
# 假设你的 holoocean 环境叫 'holoocean_env'，请替换为真实名称
echo "Starting HoloOcean Server..."
gnome-terminal -- /bin/bash -c "source ~/miniconda3/bin/activate auv_env; python examples/diffusion/eval_server_holoocean.py --env_config $ENV_CONFIG --port $PORT; exec bash"

# 等待几秒让 Server 启动
sleep 5

# 2. 启动 Client (Flow Matching 环境)
# 假设你的 FM 环境叫 'flow_matching'
echo "Starting Flow Matching Client..."
source ~/miniconda3/bin/activate flow_matching
python examples/diffusion/eval_client_fm.py --ckpt "$CKPT" --port $PORT --device auto