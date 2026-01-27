import torch
import torch.nn as nn
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder

class AUVHybridObsEncoder(nn.Module):
    def __init__(self, shape_meta, rgb_model, resize_shape=None, crop_shape=None, **kwargs):
        super().__init__()
        
        # 1. 视觉部分：继续使用框架成熟的 MultiImageObsEncoder
        # 它会自动根据 shape_meta 处理 'camera_image'
        self.vision_encoder = MultiImageObsEncoder(
            shape_meta=shape_meta,
            rgb_model=rgb_model,
            resize_shape=resize_shape,
            crop_shape=crop_shape,
            **kwargs
        )
        
        # 2. 确定低维状态维度
        # 根据你的 Zarr 结构，state 维度为 8
        self.state_key = 'state'
        self.state_dim = 8
        self.n_obs_steps = 2 # 对应你的 n_obs_steps 配置
        
        # 3. 计算最终输出维度
        # vision_encoder 的输出维度通常是 512 (ResNet18) * 空间池化系数
        # 假设 ResNet 输出 2048 维，加上 2 帧 * 8 维 state
        self.output_features_dim = self.vision_encoder.output_shape()[0] + (self.state_dim * self.n_obs_steps)

    def forward(self, obs_dict):
        # 提取视觉特征: [B, Vision_Dim]
        vision_feat = self.vision_encoder(obs_dict)
        
        # 提取低维状态特征
        # obs_dict['state'] 形状为 [B, Horizon, 8]，我们只取前 n_obs_steps 帧
        state_feat = obs_dict[self.state_key][:, :self.n_obs_steps, :]
        
        # 将 [B, 2, 8] 展平为 [B, 16]
        state_feat_flat = state_feat.reshape(state_feat.shape[0], -1)
        
        # 特征拼接: [B, Vision_Dim + 16]
        combined_feat = torch.cat([vision_feat, state_feat_flat], dim=-1)
        
        return combined_feat

    def output_shape(self):
        return (self.output_features_dim,)