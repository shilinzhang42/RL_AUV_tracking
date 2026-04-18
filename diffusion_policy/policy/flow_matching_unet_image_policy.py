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

        # 记录 rgb 键名，用于训练/推理时的图像类型转换
        self.rgb_keys = [
            k for k, v in shape_meta['obs'].items()
            if v.get('type') == 'rgb'
        ]

        action_dim = shape_meta['action']['shape'][0]
        self.action_dim = action_dim
        global_cond_dim = None
        if obs_as_global_cond:
            # 建立 Dummy 时保持 float32 以确保初始化正常
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
            tensor_shape = (1, self.n_obs_steps) + shape 
            dummy_obs[key] = torch.zeros(tensor_shape, dtype=torch.float32)
        return dummy_obs

    # ================= 核心修改 1: 内存优化版训练逻辑 =================
    def compute_loss(self, batch):
        # 1. 立即将原始 batch 移至 GPU (此时图像还是 uint8，占用 RAM 极小)
        batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))

        # 2. 观测数据预处理
        obs = batch['obs']
        compact_obs = dict()
        for key, value in obs.items():
            # 【优化】：先切片减少元素数量，再进行类型转换
            # 维度从 (B, horizon, C, H, W) 变为 (B, n_obs_steps, C, H, W)
            sliced_obs = value[:, :self.n_obs_steps]
            
            # 用 shape_meta 中的 rgb 键名判断，避免依赖键名字符串
            if key in self.rgb_keys and sliced_obs.dtype == torch.uint8:
                compact_obs[key] = sliced_obs.float() / 255.0
            else:
                compact_obs[key] = sliced_obs
        
        # 3. 归一化与特征提取 (在 GPU 上并行)
        norm_obs = self.normalizer.normalize(compact_obs)
        global_cond = self.obs_encoder(norm_obs) 

        # 4. 动作处理
        # 确保动作也被正确归一化
        norm_action = self.normalizer['action'].normalize(batch['action'])
        batch_size = norm_action.shape[0]

        # 5. Flow Matching 核心逻辑
        # x1 是目标 (Expert Action), x0 是源噪声
        x1 = norm_action
        x0 = torch.randn_like(x1)
        
        # 采样时间 t ~ Uniform[0, 1]
        t = torch.rand((batch_size,), device=self.device)
        
        # 构造插值路径: x_t = (1 - t) \cdot x_0 + t \cdot x_1
        t_expand = t.view(-1, 1, 1)
        x_t = (1 - t_expand) * x0 + t_expand * x1
        
        # 目标速度向量: v_t = x_1 - x_0
        target_v = x1 - x0

        # 6. 网络预测并计算 MSE Loss
        pred_v = self.model(x_t, t, global_cond=global_cond)
        loss = F.mse_loss(pred_v, target_v)
        
        return loss

    # ================= 核心修改 2: 内存优化版推理逻辑 =================
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 1. 将输入移至 GPU
        obs_dict = dict_apply(obs_dict, lambda x: x.to(self.device, non_blocking=True))

        # 2. 同样的延迟类型转换策略
        processed_obs = dict()
        for key, value in obs_dict.items():
            if key in self.rgb_keys and value.dtype == torch.uint8:
                processed_obs[key] = value.float() / 255.0
            else:
                processed_obs[key] = value

        # 3. 预处理与特征提取
        norm_obs = self.normalizer.normalize(processed_obs)
        batch_size = next(iter(norm_obs.values())).shape[0]
        global_cond = self.obs_encoder(norm_obs)

        # 4. ODE Solver (Euler Method)
        action_dim = self.action_dim
        traj = torch.randn((batch_size, self.horizon, action_dim), device=self.device)

        steps = self.num_inference_steps
        dt = 1.0 / steps
        for i in range(steps):
            t_val = torch.full((batch_size,), i / steps, device=self.device)
            v = self.model(traj, t_val, global_cond=global_cond)
            traj = traj + v * dt

        # 5. 反归一化并提取动作步
        traj = self.normalizer['action'].unnormalize(traj)
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        actions = traj[:, start:end]
        return {'action': actions, 'action_pred': actions}

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())