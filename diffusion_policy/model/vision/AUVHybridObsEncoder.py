import torch
import torch.nn as nn
import copy
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder

class AUVHybridObsEncoder(nn.Module):
    def __init__(self, shape_meta, **kwargs):
        super().__init__()
        
        # 1. 提取配置
        internal_meta = copy.deepcopy(shape_meta)
        obs_meta = internal_meta['obs']
        self.rgb_keys = []
        self.low_dim_keys = []
        
        # 获取 n_obs_steps，默认为 2，最好从外部传入或从某处读取
        # 在 diffusion_policy 中，通常可以从 kwargs 获取
        self.n_obs_steps = kwargs.get('n_obs_steps', 2) 

        for k, v in shape_meta['obs'].items():
            if v.get('type') == 'rgb':
                self.rgb_keys.append(k)
            else:
                self.low_dim_keys.append(k)
                if k in obs_meta:
                    del obs_meta[k]

        # 2. 实例化内部视觉编码器
        self.vision_encoder = MultiImageObsEncoder(shape_meta=internal_meta, **kwargs)
        
        # 3. 【核心修正】自动计算输入维度
        # 遍历所有 low_dim 键，计算总维度
        total_low_dim_input = 0
        for k in self.low_dim_keys:
            dim = shape_meta['obs'][k]['shape'][0]
            total_low_dim_input += dim * self.n_obs_steps
        
        print(f"[Encoder] 自动计算 Low-Dim 输入总维度: {total_low_dim_input} (Steps: {self.n_obs_steps})")

        self.project_dim = 256 
        self.state_projector = nn.Sequential(
            nn.Linear(total_low_dim_input, self.project_dim),
            nn.LayerNorm(self.project_dim),
            nn.ReLU(),
            nn.Linear(self.project_dim, self.project_dim),
            nn.ReLU()
        )
        
        self.full_shape_meta = shape_meta

    def forward(self, obs_dict):
        batch_size = None
        features = list()

        # 1. 提取视觉特征 (B, 2048)
        vision_obs_dict = {k: obs_dict[k] for k in self.rgb_keys}
        vision_feat = self.vision_encoder(vision_obs_dict)
        batch_size = vision_feat.shape[0]
        features.append(vision_feat)

        # 2. 提取并投影状态特征
        for key in self.low_dim_keys:
            data = obs_dict[key]  # 可能形状是 (B, 16, 8) 或 (B, 6, 8) 或 (B, 2, 8)
            
            # 【核心修正】强制截断，只取前 self.n_obs_steps 步
            # 这样无论 dummy_obs 给多少，我们只处理定义的 2 步 (16维)
            if len(data.shape) == 3:
                data = data[:, :self.n_obs_steps, :] 
                data_flat = data.reshape(batch_size, -1) # 结果一定是 (B, 16)
            else:
                data_flat = data

            # 现在的 data_flat 一定是 (B, 16)，可以安全通过 (16x256) 的线性层
            projected_state = self.state_projector(data_flat)
            features.append(projected_state)

        # 3. 拼接：[B, 2048 + 256]
        combined_feat = torch.cat(features, dim=-1)
        return combined_feat

    @torch.no_grad()
    def output_shape(self):
        v_shape = self.vision_encoder.output_shape()
        # 现在的总维度是：视觉输出 + 投影后的状态维度
        return (int(v_shape[0] + self.project_dim),)