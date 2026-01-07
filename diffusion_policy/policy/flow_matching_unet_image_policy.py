from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.common.pytorch_util import dict_apply


class FlowMatchingUnetImagePolicy(BaseImagePolicy):
    def __init__(self,
                 shape_meta: dict,
                 noise_scheduler,             # 兼容 Hydra，实际未使用
                 obs_encoder: nn.Module,
                 horizon,
                 n_action_steps,
                 n_obs_steps,
                 num_inference_steps=10,
                 obs_as_global_cond=True,
                 diffusion_step_embed_dim=256,
                 down_dims=(256, 512, 1024),
                 kernel_size=5,
                 n_groups=8,
                 cond_predict_scale=True,
                 **kwargs):
        super().__init__()

        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.num_inference_steps = num_inference_steps
        self.kwargs = kwargs

        self.obs_encoder = obs_encoder
        self.normalizer = LinearNormalizer()

        action_dim = shape_meta['action']['shape'][0]
        self.action_dim = action_dim
        global_cond_dim = None
        if obs_as_global_cond:
            dummy_obs = self._build_dummy_obs(shape_meta)
            with torch.no_grad():
                encoder_out = self.obs_encoder(dummy_obs)
            global_cond_dim = encoder_out.shape[-1]
            print(f"[FM Policy] detected global_cond_dim: {global_cond_dim}")

        self.model = ConditionalUnet1D(
            input_dim=action_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

    def _build_dummy_obs(self, shape_meta: dict) -> Dict[str, torch.Tensor]:
        dummy_obs = {}
        for key, attr in shape_meta['obs'].items():
            shape = tuple(attr['shape'])
            tensor_shape = (1, self.horizon) + shape
            dummy_obs[key] = torch.zeros(tensor_shape, dtype=torch.float32)
        return dummy_obs

    # ================= 核心修改 1: 训练逻辑 =================
    def compute_loss(self, batch):
        # 1. 数据预处理
        norm_obs = self.normalizer.normalize(batch['obs'])
        norm_action = self.normalizer['action'].normalize(batch['action'])
        batch_size = norm_action.shape[0]

        # 2. 编码观测特征
        global_cond = self.obs_encoder(norm_obs)

        # 3. Flow Matching 核心逻辑
        # 3.1 采样 x0 (Source Noise) ~ N(0, I)
        x1 = norm_action
        x0 = torch.randn_like(x1)
        
        # 3.2 采样时间 t ~ Uniform[0, 1]
        t = torch.rand((batch_size,), device=x1.device)
        
        # 3.3 构造插值路径 x_t = (1 - t) * x0 + t * x1
        # 这里的 t 需要 reshape 以进行广播乘法
        t_expand = t.view(-1, 1, 1)
        x_t = (1 - t_expand) * x0 + t_expand * x1
        
        # 3.4 计算目标速度向量 (Target Velocity)
        # d(x_t)/dt = x1 - x0
        target_v = x1 - x0

        # 3.5 网络预测
        # 注意：U-Net 通常期望 timestep 是整数或特定范围。
        # 如果 ConditionalUnet1D 内部使用的是 Sinusoidal Embedding，传入 float [0,1] 也是可以的，
        # 但为了保持与原 Diffusion 训练分布一致，通常建议缩放到 [0, 1000] 或类似范围，或者直接传 float。
        # 这里我们直接传 t (float)，因为 Sinusoidal Embedding 对数值范围不敏感，只要一致即可。
        pred_v = self.model(x_t, t, global_cond=global_cond)

        # 3.6 计算 MSE Loss
        loss = F.mse_loss(pred_v, target_v)
        
        return loss

    # ================= 核心修改 2: 推理逻辑 (ODE Solver) =================
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 1. 数据预处理
        norm_obs = self.normalizer.normalize(obs_dict)
        batch_size = next(iter(norm_obs.values())).shape[0]
        global_cond = self.obs_encoder(norm_obs)

        # 2. 初始化 x (从标准高斯噪声开始，对应 t=0)
        action_dim = self.action_dim
        traj = torch.randn((batch_size, self.horizon, action_dim), device=self.device)

        # 3. ODE Solver (Euler Method)
        # 从 t=0 积分到 t=1
        steps = self.num_inference_steps
        dt = 1.0 / steps
        for i in range(steps):
            # 当前时间 t
            t_val = torch.full((batch_size,), i / steps, device=self.device)
            
            # 预测速度场 v
            v = self.model(traj, t_val, global_cond=global_cond)
            
            # Euler 更新: x_new = x_old + v * dt
            traj = traj + v * dt

        # 4. 后处理 (反归一化)
        traj = self.normalizer['action'].unnormalize(traj)
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        actions = traj[:, start:end]
        return {'action': actions, 'action_pred': actions}

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())