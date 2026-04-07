from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.common.pytorch_util import dict_apply


class DDPMUnetImagePolicyAUV(BaseImagePolicy):
    """
    DDPM-based image policy for AUV control.
    结构参照 FlowMatchingUnetImagePolicy，但使用 DDPM 噪声预测方式。
    
    核心特点:
    - 使用 DDPMScheduler 进行时间步调度
    - 预测增加的噪声（epsilon）而非速度向量
    - 推理时使用反向扩散过程进行采样
    - 内存优化：延迟图像的uint8->float32转换到GPU上
    """
    
    def __init__(self,
                 shape_meta: dict,
                 noise_scheduler: DDPMScheduler,
                 obs_encoder: nn.Module,
                 horizon,
                 n_action_steps,
                 n_obs_steps,
                 num_inference_steps=None,
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
        self.kwargs = kwargs

        self.obs_encoder = obs_encoder
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()

        action_dim = shape_meta['action']['shape'][0]
        self.action_dim = action_dim
        
        # 设置推理步数
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        global_cond_dim = None
        if obs_as_global_cond:
            # 建立 Dummy 时保持 float32 以确保初始化正常
            dummy_obs = self._build_dummy_obs(shape_meta)
            with torch.no_grad():
                encoder_out = self.obs_encoder(dummy_obs)
            global_cond_dim = encoder_out.shape[-1]
            print(f"[DDPM Policy] detected global_cond_dim: {global_cond_dim}")

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
        """
        DDPM loss 计算: 预测采样的噪声
        
        Loss = MSE(model_output, noise_sampled)
        其中 x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        """
        # 1. 立即将原始 batch 移至 GPU (此时图像还是 uint8，占用 RAM 极小)
        batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))

        # 2. 观测数据预处理
        obs = batch['obs']
        compact_obs = dict()
        for key, value in obs.items():
            # 【优化】：先切片减少元素数量，再进行类型转换
            sliced_obs = value[:, :self.n_obs_steps]
            
            # 【关键修改】：仅在 GPU 上将图像从 uint8 转换为 float32
            if 'image' in key and sliced_obs.dtype == torch.uint8:
                compact_obs[key] = sliced_obs.float() / 255.0
            else:
                compact_obs[key] = sliced_obs
        
        # 3. 归一化与特征提取 (在 GPU 上并行)
        norm_obs = self.normalizer.normalize(compact_obs)
        global_cond = self.obs_encoder(norm_obs) 

        # 4. 动作处理
        norm_action = self.normalizer['action'].normalize(batch['action'])
        batch_size = norm_action.shape[0]

        # 5. DDPM 核心逻辑
        # x_0 是纯动作数据
        x_0 = norm_action
        
        # 采样噪声
        noise = torch.randn_like(x_0)
        
        # 采样随机时间步 t ~ Uniform[0, num_train_timesteps)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=self.device
        )
        
        # 使用调度器的 alpha_cumprod 计算加噪后的 x_t
        # x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        sqrt_alphas_cumprod = self.noise_scheduler.alphas_cumprod[timesteps].sqrt()
        sqrt_one_minus_alphas_cumprod = (1.0 - self.noise_scheduler.alphas_cumprod[timesteps]).sqrt()
        
        # 展开维度以支持矩阵乘法
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.view(-1, 1, 1)
        
        x_t = sqrt_alphas_cumprod * x_0 + sqrt_one_minus_alphas_cumprod * noise

        # 6. 网络预测噪声并计算 MSE Loss
        pred_noise = self.model(x_t, timesteps, global_cond=global_cond)
        loss = F.mse_loss(pred_noise, noise)
        
        return loss

    # ================= 核心修改 2: 内存优化版推理逻辑 =================
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        使用 DDPMScheduler 进行反向扩散采样
        """
        # 1. 将输入移至 GPU
        obs_dict = dict_apply(obs_dict, lambda x: x.to(self.device, non_blocking=True))

        # 2. 同样的延迟类型转换策略
        processed_obs = dict()
        for key, value in obs_dict.items():
            if 'image' in key and value.dtype == torch.uint8:
                processed_obs[key] = value.float() / 255.0
            else:
                processed_obs[key] = value

        # 3. 预处理与特征提取
        norm_obs = self.normalizer.normalize(processed_obs)
        batch_size = next(iter(norm_obs.values())).shape[0]
        global_cond = self.obs_encoder(norm_obs)

        # 4. 反向扩散采样 (从纯噪声到清晰动作)
        action_dim = self.action_dim
        x_t = torch.randn((batch_size, self.horizon, action_dim), device=self.device)

        # 设置推理步数
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            # 预测该步的噪声
            pred_noise = self.model(x_t, t.unsqueeze(0).expand(batch_size).to(self.device), 
                                   global_cond=global_cond)
            
            # 使用调度器进行反向扩散步骤
            scheduler_output = self.noise_scheduler.step(pred_noise, t, x_t)
            x_t = scheduler_output.prev_sample

        # 5. 反归一化并提取动作步
        x_t = self.normalizer['action'].unnormalize(x_t)
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        actions = x_t[:, start:end]
        return {'action': actions, 'action_pred': actions}

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
