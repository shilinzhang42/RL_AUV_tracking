#!/bin/bash

# 配置路径
CKPT="/home/zsl/RL_AUV_tracking/data/outputs/06.28.52_train_flow_matching_unet_image_track_image/checkpoints/epoch=0011-val_loss=0.2445.ckpt"
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
python examples/diffusion/eval_client_fm.py --ckpt "$CKPT" --port $PORT