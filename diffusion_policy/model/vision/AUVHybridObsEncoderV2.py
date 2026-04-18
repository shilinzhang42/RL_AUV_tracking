"""
AUVHybridObsEncoderV2 — 增强版混合观测编码器

相比 V1 的改进：
  - 状态分支：支持 GRU / TCN / MLP 三种时序编码器（通过 state_temporal_encoder 切换）
  - 视觉分支：支持逐帧 ResNet18 编码 + TCN 时序聚合（通过 vision_use_tcn 开关）
  - 融合模块：支持 FiLM / Cross-Attention / cat 三种融合方式（通过 fusion_type 切换）
"""

import copy
import torch
import torch.nn as nn
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder

# 复用项目内已有的 TCN 实现
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../auv_track_launcher'))
from networks.tcn import TemporalConvNet


# ---------------------------------------------------------------------------
# 辅助模块
# ---------------------------------------------------------------------------

class FiLMFusion(nn.Module):
    """
    Feature-wise Linear Modulation:
        output = gamma(state) * vision + beta(state)
    state_feat → Linear(256, vision_dim*2) → split → (gamma, beta)
    """
    def __init__(self, vision_dim: int, state_dim: int):
        super().__init__()
        self.generator = nn.Linear(state_dim, vision_dim * 2)
        nn.init.zeros_(self.generator.weight)
        nn.init.ones_(self.generator.bias[:vision_dim])   # gamma 初始化为 1
        nn.init.zeros_(self.generator.bias[vision_dim:])  # beta  初始化为 0

    def forward(self, vision_feat: torch.Tensor, state_feat: torch.Tensor) -> torch.Tensor:
        params = self.generator(state_feat)
        gamma, beta = params.chunk(2, dim=-1)
        return gamma * vision_feat + beta


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention 融合：视觉特征作为 Query，状态特征作为 Key/Value。
    输出维度与 vision_dim 相同，并加残差连接。
    """
    def __init__(self, vision_dim: int, state_dim: int, num_heads: int = 8):
        super().__init__()
        self.kv_proj = nn.Linear(state_dim, vision_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=vision_dim, num_heads=num_heads,
            batch_first=True, dropout=0.0
        )
        self.norm = nn.LayerNorm(vision_dim)

    def forward(self, vision_feat: torch.Tensor, state_feat: torch.Tensor) -> torch.Tensor:
        # vision_feat: (B, vision_dim)  state_feat: (B, state_dim)
        q = vision_feat.unsqueeze(1)                    # (B, 1, vision_dim)
        kv = self.kv_proj(state_feat).unsqueeze(1)      # (B, 1, vision_dim)
        attn_out, _ = self.attn(q, kv, kv)              # (B, 1, vision_dim)
        out = self.norm(vision_feat + attn_out.squeeze(1))
        return out


# ---------------------------------------------------------------------------
# 主编码器
# ---------------------------------------------------------------------------

class AUVHybridObsEncoderV2(nn.Module):
    """
    增强版 AUV 混合观测编码器。

    参数
    ----
    shape_meta : dict
        与 policy/dataset 共享的观测元信息。
    n_obs_steps : int
        观测窗口长度（时序步数）。
    state_temporal_encoder : str
        状态分支时序编码器类型：'gru' | 'tcn' | 'none'。
        'none' 退化为原始 flatten+MLP 行为。
    vision_use_tcn : bool
        是否在视觉分支启用逐帧编码 + TCN 时序聚合。
        False 时退化为原始 MultiImageObsEncoder 行为。
    fusion_type : str
        融合方式：'film' | 'cross_attn' | 'cat'。
        'cat' 退化为原始简单拼接。
    vision_cond_dim : int
        视觉特征投影维度，默认 512。
    state_cond_dim : int
        状态特征投影维度，默认 256。
    **kwargs
        透传给 MultiImageObsEncoder 的参数（rgb_model, resize_shape 等）。
    """

    def __init__(
        self,
        shape_meta: dict,
        n_obs_steps: int = 6,
        state_temporal_encoder: str = 'gru',
        vision_use_tcn: bool = True,
        fusion_type: str = 'film',
        vision_cond_dim: int = 512,
        state_cond_dim: int = 256,
        **kwargs,
    ):
        super().__init__()

        self.n_obs_steps = int(n_obs_steps)
        self.state_temporal_encoder = state_temporal_encoder
        self.vision_use_tcn = vision_use_tcn
        self.fusion_type = fusion_type
        self.vision_cond_dim = vision_cond_dim
        self.state_cond_dim = state_cond_dim

        # ---- 分离 rgb / low_dim 键 ----
        internal_meta = copy.deepcopy(shape_meta)
        self.rgb_keys = []
        self.low_dim_keys = []
        for k, v in shape_meta['obs'].items():
            if v.get('type') == 'rgb':
                self.rgb_keys.append(k)
            else:
                self.low_dim_keys.append(k)
                del internal_meta['obs'][k]

        # ---- 视觉分支 ----
        self._build_vision_branch(internal_meta, kwargs)

        # ---- 状态分支 ----
        self._build_state_branch(shape_meta)

        # ---- 融合模块 ----
        self._build_fusion()

        self.full_shape_meta = shape_meta

    # ------------------------------------------------------------------
    # 构建子模块
    # ------------------------------------------------------------------

    def _build_vision_branch(self, internal_meta: dict, kwargs: dict):
        """构建视觉编码分支。"""
        encoder_kwargs = copy.deepcopy(kwargs)
        encoder_kwargs.pop('n_obs_steps', None)

        # 基础 ResNet 编码器（始终需要，用于逐帧编码或直接编码）
        self.vision_encoder = MultiImageObsEncoder(
            shape_meta=internal_meta, **encoder_kwargs
        )

        if self.vision_use_tcn:
            # 推断单帧特征维度：构造单帧 dummy（shape_meta 中存的是单帧 shape）
            with torch.no_grad():
                dummy_single = {}
                for k, v in internal_meta['obs'].items():
                    # shape_meta 中 rgb 的 shape 是单帧 (C, H, W)
                    dummy_single[k] = torch.zeros(
                        (1,) + tuple(v['shape']), dtype=torch.float32
                    )
                single_feat = self.vision_encoder(dummy_single)
            per_frame_dim = single_feat.shape[-1]

            # TCN: (B, per_frame_dim, T) → (B, vision_cond_dim, T)
            self.vision_tcn = TemporalConvNet(
                num_inputs=per_frame_dim,
                num_channels=[self.vision_cond_dim, self.vision_cond_dim],
                kernel_size=3,
                dropout=0.1,
            )
            self.vision_proj = nn.Sequential(
                nn.LayerNorm(self.vision_cond_dim),
                nn.ReLU(),
            )
        else:
            # 退化路径：直接用 LazyLinear 投影
            self.vision_projector = nn.Sequential(
                nn.LazyLinear(self.vision_cond_dim),
                nn.LayerNorm(self.vision_cond_dim),
                nn.ReLU(),
            )

    def _build_state_branch(self, shape_meta: dict):
        """构建状态时序编码分支。"""
        # 推断单步状态维度（shape_meta 中 low_dim 的 shape 是单步的，如 [6]）
        state_input_dim = 0
        for k in self.low_dim_keys:
            shape = tuple(shape_meta['obs'][k]['shape'])
            state_input_dim += shape[-1]  # 取最后一维（单步特征维度）

        self.state_input_dim = state_input_dim

        if self.state_temporal_encoder == 'gru':
            self.state_gru = nn.GRU(
                input_size=state_input_dim,
                hidden_size=self.state_cond_dim,
                num_layers=2,
                batch_first=True,
                dropout=0.1,
            )
        elif self.state_temporal_encoder == 'tcn':
            self.state_tcn = TemporalConvNet(
                num_inputs=state_input_dim,
                num_channels=[128, self.state_cond_dim],
                kernel_size=3,
                dropout=0.1,
            )
        else:
            # 'none'：退化为原始 flatten+MLP
            self.state_projector = nn.Sequential(
                nn.LazyLinear(self.state_cond_dim),
                nn.LayerNorm(self.state_cond_dim),
                nn.ReLU(),
                nn.Linear(self.state_cond_dim, self.state_cond_dim),
                nn.ReLU(),
            )

    def _build_fusion(self):
        """构建融合模块。"""
        if self.fusion_type == 'film':
            self.fusion = FiLMFusion(
                vision_dim=self.vision_cond_dim,
                state_dim=self.state_cond_dim,
            )
        elif self.fusion_type == 'cross_attn':
            self.fusion = CrossAttentionFusion(
                vision_dim=self.vision_cond_dim,
                state_dim=self.state_cond_dim,
                num_heads=8,
            )
        # 'cat' 不需要额外模块

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _encode_vision(self, obs_dict: dict) -> torch.Tensor:
        """视觉分支前向。返回 (B, vision_cond_dim)。"""
        vision_obs = {k: obs_dict[k] for k in self.rgb_keys}

        if self.vision_use_tcn:
            # 高效逐帧编码：将 (B, T, C, H, W) reshape 为 (B*T, C, H, W)
            # 一次 forward 完成所有帧，充分利用 GPU 并行
            B = None
            flat_obs = {}
            for k in self.rgb_keys:
                img = vision_obs[k]  # (B, T, C, H, W)
                if B is None:
                    B, T = img.shape[0], img.shape[1]
                flat_obs[k] = img.reshape(B * T, *img.shape[2:])  # (B*T, C, H, W)

            flat_feat = self.vision_encoder(flat_obs)              # (B*T, D_frame)
            frame_feats = flat_feat.reshape(B, T, -1)              # (B, T, D_frame)

            # (B, T, D_frame) → (B, D_frame, T) for TCN
            frame_feats = frame_feats.permute(0, 2, 1)             # (B, D_frame, T)
            tcn_out = self.vision_tcn(frame_feats)                 # (B, vision_cond_dim, T)
            vision_feat = tcn_out[:, :, -1]                        # (B, vision_cond_dim)
            vision_feat = self.vision_proj(vision_feat)
        else:
            raw = self.vision_encoder(vision_obs)
            vision_feat = self.vision_projector(raw)

        return vision_feat

    def _encode_state(self, obs_dict: dict, B: int) -> torch.Tensor:
        """状态分支前向。返回 (B, state_cond_dim)。"""
        # 收集各 low_dim 键的单步特征，拼接为 (B, T, state_input_dim)
        step_feats = []
        for k in self.low_dim_keys:
            data = obs_dict[k]  # (B, T, D) 或 (B, D)
            if data.ndim == 3:
                data = data[:, :self.n_obs_steps, :]  # (B, T, D)
            else:
                data = data.unsqueeze(1).expand(-1, self.n_obs_steps, -1)
            step_feats.append(data)

        state_seq = torch.cat(step_feats, dim=-1)  # (B, T, state_input_dim)

        if self.state_temporal_encoder == 'gru':
            _, h_n = self.state_gru(state_seq)      # h_n: (num_layers, B, hidden)
            state_feat = h_n[-1]                    # (B, state_cond_dim)
        elif self.state_temporal_encoder == 'tcn':
            x = state_seq.permute(0, 2, 1)          # (B, state_input_dim, T)
            tcn_out = self.state_tcn(x)             # (B, state_cond_dim, T)
            state_feat = tcn_out[:, :, -1]          # (B, state_cond_dim)
        else:
            flat = state_seq.reshape(B, -1)
            state_feat = self.state_projector(flat)

        return state_feat

    def forward(self, obs_dict: dict) -> torch.Tensor:
        # 推断 batch size
        B = None
        for k in self.rgb_keys + self.low_dim_keys:
            if k in obs_dict:
                B = obs_dict[k].shape[0]
                break

        vision_feat = self._encode_vision(obs_dict)   # (B, vision_cond_dim)

        if len(self.low_dim_keys) == 0:
            return vision_feat

        state_feat = self._encode_state(obs_dict, B)  # (B, state_cond_dim)

        # 融合
        if self.fusion_type in ('film', 'cross_attn'):
            # FiLM/CrossAttn 已将 state 信息编码进 vision，直接输出融合结果
            # 不再重复拼接原始 state_feat，避免 state 信息双重计数
            return self.fusion(vision_feat, state_feat)
        else:
            # 'cat'：直接拼接
            return torch.cat([vision_feat, state_feat], dim=-1)

    @torch.no_grad()
    def output_shape(self):
        if len(self.low_dim_keys) == 0:
            return (self.vision_cond_dim,)
        if self.fusion_type in ('film', 'cross_attn'):
            # 融合后输出维度等于 vision_cond_dim
            return (self.vision_cond_dim,)
        return (self.vision_cond_dim + self.state_cond_dim,)
