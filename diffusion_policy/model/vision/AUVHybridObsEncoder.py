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
        # 与 policy/dataset 显式对齐观测窗口长度
        encoder_kwargs = copy.deepcopy(kwargs)
        self.n_obs_steps = encoder_kwargs.pop('n_obs_steps', None)
        if self.n_obs_steps is not None:
            self.n_obs_steps = int(self.n_obs_steps)
        
        # 过滤 shape_meta 供内部 vision_encoder 使用
        for k, v in shape_meta['obs'].items():
            if v.get('type') == 'rgb':
                self.rgb_keys.append(k)
            else:
                self.low_dim_keys.append(k)
                if k in internal_meta['obs']:
                    del internal_meta['obs'][k]

        # 2. 实例化视觉编码器
        self.vision_encoder = MultiImageObsEncoder(shape_meta=internal_meta, **encoder_kwargs)

        # 3. 【核心优化 1】视觉投影器 (Vision Projector)
        # 使用 LazyLinear 自动对齐时间维展开后的输入维度
        self.vision_cond_dim = 512  # 强制压缩到 512

        self.vision_projector = nn.Sequential(
            nn.LazyLinear(self.vision_cond_dim),
            nn.LayerNorm(self.vision_cond_dim),
            nn.ReLU()
        )

        # 4. 【核心优化 2】状态投影器 (State Projector)
        # 将全部 low-dim 键拼接后统一投影，避免多键时输出维度膨胀
        self.state_cond_dim = 256  # 投影到 256
        self.state_projector = nn.Sequential(
            nn.LazyLinear(self.state_cond_dim),
            nn.LayerNorm(self.state_cond_dim),
            nn.ReLU(),
            nn.Linear(self.state_cond_dim, self.state_cond_dim),
            nn.ReLU()
        )

        self.full_shape_meta = shape_meta

    def _resolve_obs_steps(self, obs_dict):
        # 推荐由 config 显式传入，保证与 policy 的 n_obs_steps 对齐
        if self.n_obs_steps is not None:
            return int(self.n_obs_steps)

        # 兼容兜底：若未配置则按输入推断，避免硬编码
        for key in self.rgb_keys:
            x = obs_dict.get(key, None)
            if x is not None and x.ndim >= 5:
                return int(x.shape[1])

        for key in self.low_dim_keys:
            x = obs_dict.get(key, None)
            if x is not None and x.ndim >= 3:
                return int(x.shape[1])

        return 1

    def forward(self, obs_dict):
        batch_size = None
        effective_obs_steps = self._resolve_obs_steps(obs_dict)

        # --- 视觉路径 ---
        vision_obs_dict = {k: obs_dict[k] for k in self.rgb_keys}
        raw_vision_feat = self.vision_encoder(vision_obs_dict)
        batch_size = raw_vision_feat.shape[0]
        # 压缩到 512 维
        vision_feat = self.vision_projector(raw_vision_feat)

        # --- 状态路径 ---
        low_dim_features = []
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if data.ndim == 3:
                data = data[:, :effective_obs_steps, :]
                data_flat = data.reshape(batch_size, -1)
            else:
                data_flat = data.reshape(batch_size, -1)
            low_dim_features.append(data_flat)

        if len(low_dim_features) > 0:
            low_dim_concat = torch.cat(low_dim_features, dim=-1)
            state_feat = self.state_projector(low_dim_concat)
            return torch.cat([vision_feat, state_feat], dim=-1)

        return vision_feat

    @torch.no_grad()
    def output_shape(self):
        if len(self.low_dim_keys) > 0:
            return (int(self.vision_cond_dim + self.state_cond_dim),)
        return (int(self.vision_cond_dim),)