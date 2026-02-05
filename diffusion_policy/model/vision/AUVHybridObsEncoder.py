import torch
import torch.nn as nn
import copy
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder

class AUVHybridObsEncoder(nn.Module):
    def __init__(self, shape_meta, **kwargs):
        super().__init__()
        
        # 1. 提取配置
        internal_meta = copy.deepcopy(shape_meta)
        self.rgb_keys = []
        self.low_dim_keys = []
        self.n_obs_steps = kwargs.get('n_obs_steps', 2)
        
        # 过滤 shape_meta 供内部 vision_encoder 使用
        for k, v in shape_meta['obs'].items():
            if v.get('type') == 'rgb':
                self.rgb_keys.append(k)
            else:
                self.low_dim_keys.append(k)
                if k in internal_meta['obs']:
                    del internal_meta['obs'][k]

        # 2. 实例化视觉编码器
        self.vision_encoder = MultiImageObsEncoder(shape_meta=internal_meta, **kwargs)
        
        # 3. 【核心优化 1】视觉投影器 (Vision Projector)
        # 自动获取视觉编码器的原始输出维度 (如 12288)
        raw_vision_dim = self.vision_encoder.output_shape()[0] * 6
        self.vision_cond_dim = 512  # 强制压缩到 512
        
        self.vision_projector = nn.Sequential(
            nn.Linear(raw_vision_dim, self.vision_cond_dim),
            nn.LayerNorm(self.vision_cond_dim),
            nn.ReLU()
        )
        
        # 4. 【核心优化 2】状态投影器 (State Projector)
        total_low_dim_input = 0
        for k in self.low_dim_keys:
            dim = shape_meta['obs'][k]['shape'][0]
            total_low_dim_input += dim * self.n_obs_steps
            
        self.state_cond_dim = 256  # 投影到 256
        self.state_projector = nn.Sequential(
            nn.Linear(total_low_dim_input, self.state_cond_dim),
            nn.LayerNorm(self.state_cond_dim),
            nn.ReLU(),
            nn.Linear(self.state_cond_dim, self.state_cond_dim),
            nn.ReLU()
        )
        
        self.full_shape_meta = shape_meta

    def forward(self, obs_dict):
        batch_size = None
        
        # --- 视觉路径 ---
        vision_obs_dict = {k: obs_dict[k] for k in self.rgb_keys}
        raw_vision_feat = self.vision_encoder(vision_obs_dict)
        batch_size = raw_vision_feat.shape[0]
        # 压缩到 512 维
        vision_feat = self.vision_projector(raw_vision_feat)

        # --- 状态路径 ---
        low_dim_features = []
        for key in self.low_dim_keys:
            data = obs_dict[key][:, :self.n_obs_steps, :] # 确保步数对齐
            data_flat = data.reshape(batch_size, -1)
            # 投影到 256 维
            projected_state = self.state_projector(data_flat)
            low_dim_features.append(projected_state)

        # --- 最终拼接：512 + 256 = 768 ---
        return torch.cat([vision_feat] + low_dim_features, dim=-1)

    @torch.no_grad()
    def output_shape(self):
        # 明确返回 768
        return (int(self.vision_cond_dim + self.state_cond_dim),)