import torch
import torch.nn as nn
import copy
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder

class AUVHybridObsEncoder(nn.Module):
    def __init__(self, shape_meta, **kwargs):
        super().__init__()
        
        # 1. 深度拷贝一份配置，并移除所有非图像键
        # 这样内部的 MultiImageObsEncoder 就不会去寻找 'state' 了
        internal_meta = copy.deepcopy(shape_meta)
        obs_meta = internal_meta['obs']
        
        self.rgb_keys = []
        self.low_dim_keys = []
        
        # 分类存储键名
        for k, v in shape_meta['obs'].items():
            if v.get('type') == 'rgb':
                self.rgb_keys.append(k)
            else:
                self.low_dim_keys.append(k)
                # 从内部配置中删除低维键
                if k in obs_meta:
                    del obs_meta[k]

        # 2. 实例化内部编码器（它现在只处理图像）
        self.vision_encoder = MultiImageObsEncoder(
            shape_meta=internal_meta,
            **kwargs
        )
        
        # 记录原始配置用于维度推算
        self.full_shape_meta = shape_meta

    def forward(self, obs_dict):
        batch_size = None
        features = list()

        # 1. 视觉特征提取 (只把图像键传进去)
        vision_obs_dict = {k: obs_dict[k] for k in self.rgb_keys}
        vision_feat = self.vision_encoder(vision_obs_dict)
        features.append(vision_feat)

        # 2. 手动接管低维状态 (state)
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            
            # 将 (B, T, D) 展平为 (B, T*D)
            if len(data.shape) == 3:
                features.append(data.reshape(batch_size, -1))
            else:
                features.append(data)

        # 3. 拼接视觉与状态特征
        return torch.cat(features, dim=-1)

    @torch.no_grad()
    def output_shape(self):
        # 自动推算总维度：视觉维度 + (状态维度 * 观测步数)
        v_shape = self.vision_encoder.output_shape()
        v_dim = v_shape[0]
        
        # 假设 n_obs_steps 为 2 (需与你的配置同步)
        n_obs = 2 
        
        ld_dim = 0
        for k in self.low_dim_keys:
            # 取得 [8]
            shape = self.full_shape_meta['obs'][k]['shape']
            import numpy as np
            ld_dim += np.prod(shape) * n_obs
            
        return (int(v_dim + ld_dim),)