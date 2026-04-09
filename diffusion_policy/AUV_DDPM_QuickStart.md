# AUV DDPM 脚本 - 快速启动指南

## 📋 概述

本指南帮助你快速启动用 DDPM 训练 AUV 控制策略。核心是根据你现有的 Flow Matching 形式改写，保持相同的架构和内存优化。

---

## 📁 新增文件清单

已为你创建的三个核心文件：

### 1. **DDPM 策略文件** 
```
路径: diffusion_policy/policy/ddpm_unet_image_policy_auv.py
作用: 定义 DDPM 扩散模型的前向传播、训练损失、推理采样
核心类: DDPMUnetImagePolicyAUV
```

### 2. **DDPM 训练脚本**
```
路径: diffusion_policy/workspace/train_ddpm_unet_image_workspace_auv.py
作用: 完整的训练循环、验证、日志、checkpoint 管理
核心类: TrainDDPMUnetImageWorkspaceAUV
```

### 3. **DDPM 配置文件**
```
路径: diffusion_policy/config/train_ddpm_unet_image_workspace_auv.yaml
作用: Hydra 配置，定义所有超参数
```

---

## 🚀 快速开始（3 步）

### 步骤 1: 准备环境
```bash
cd /home/zsl/RL_AUV_tracking

# 确保 conda 环境激活
conda activate auv_env

# 验证依赖已安装
python -c "import diffusers; import torch; print('✅ Dependencies OK')"
```

### 步骤 2: 准备数据
```bash
# 确保你的数据集已准备好
# 配置文件中默认使用 track_image 任务
# 检查 diffusion_policy/config/task/track_image.yaml
```

### 步骤 3: 启动训练
```bash
# 最简单的方式（使用默认配置）
python diffusion_policy/workspace/train_ddpm_unet_image_workspace_auv.py \
  --config-name=train_ddpm_unet_image_workspace_auv

# 或者带自定义参数
python diffusion_policy/workspace/train_ddpm_unet_image_workspace_auv.py \
  --config-name=train_ddpm_unet_image_workspace_auv \
  training.num_epochs=20 \
  training.device="cuda:0" \
  dataloader.batch_size=128
```

---

## ✨ 主要改进与特性

### 1. 基于 Flow Matching 的结构
```python
# 共享相同的对象编码和内存优化
- 相同的 obs_encoder (AUVHybridObsEncoder)
- 相同的 UNet 架构 (ConditionalUnet1D)
- 相同的归一化方案 (LinearNormalizer)
```

### 2. 内存优化（保留自 FM）
```python
# 延迟类型转换 - 图像在 GPU 上从 uint8 → float32
if 'image' in key and sliced_obs.dtype == torch.uint8:
    compact_obs[key] = sliced_obs.float() / 255.0

# Non-blocking 数据传输
batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

# 动态垃圾回收
del batch, obs_dict, gt_action
```

### 3. DDPM 核心算法
```python
# 训练：预测高斯噪声
Loss = MSE(model_output, noise_sampled)

# 推理：反向扩散去噪
for t in [T, T-1, ..., 0]:
    x_{t-1} = scheduler.step(model_output, t, x_t)
```

---

## 🔧 配置修改示例

### 示例 1: 使用更多推理步数（提高质量）
```yaml
# 直接编辑 train_ddpm_unet_image_workspace_auv.yaml
policy:
  num_inference_steps: 100  # 默认 50，增加到 100
```

或命令行：
```bash
python train_ddpm_unet_image_workspace_auv.py \
  --config-name=train_ddpm_unet_image_workspace_auv \
  policy.num_inference_steps=100
```

### 示例 2: 减少 batch size 节省内存
```bash
python train_ddpm_unet_image_workspace_auv.py \
  --config-name=train_ddpm_unet_image_workspace_auv \
  dataloader.batch_size=128 \
  training.gradient_accumulate_every=2
```

### 示例 3: 修改学习率和 warmup
```bash
python train_ddpm_unet_image_workspace_auv.py \
  --config-name=train_ddpm_unet_image_workspace_auv \
  optimizer.lr=1e-4 \
  training.lr_warmup_steps=1000
```

### 示例 4: 使用更好的 beta schedule
```yaml
# 修改调度器，尝试不同的 schedule
noise_scheduler:
  beta_schedule: linear        # 改为 linear
  # 其他选项: cosine, sqrt, squaredcos_cap_v2
```


## 🔄 与 Flow Matching 的关键差异

| 方面 | Flow Matching | DDPM (AUV) |
|------|--------------|-----------|
| 推理步数 | 20 | 50 ⚠️ |
| 推理速度 | 快 ⚡ | 中等 ⏱️ |
| 训练收敛 | 快 | 中等 |
| 输出质量 | 良好 | 更好 ✨ |
| 超参敏感度 | 低 | 低-中 |

---

## 🐛 常见问题

### Q1: 报错 `ModuleNotFoundError: No module named 'ddpm_unet_image_policy_auv'`

**解决:**
```bash
# 确保文件位置正确
ls -la diffusion_policy/policy/ddpm_unet_image_policy_auv.py

# 清除 Python 缓存
find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null

# 重新运行
python train_ddpm_unet_image_workspace_auv.py ...
```

### Q2: CUDA 内存溢出

**解决:**
```bash
# 方案 1: 减少 batch size
python train_ddpm_unet_image_workspace_auv.py \
  --config-name=train_ddpm_unet_image_workspace_auv \
  dataloader.batch_size=128 \
  training.gradient_accumulate_every=2

# 方案 2: 进一步减小模型
# 编辑配置中的 down_dims: [256, 512, 1024] → [128, 256, 512]

# 方案 3: 启用混合精度（高级）
# 修改 train_ddpm_unet_image_workspace_auv.py 第 XXX 行
# 使用 autocast 或 torch.cuda.amp
```

### Q3: 训练 Loss 不下降

**排查清单:**
```python
# 1. 检查数据加载
# 添加调试代码查看 batch

# 2. 验证时间步采样
# 确保 timesteps ∈ [0, num_train_timesteps-1]

# 3. 降低学习率
python train_ddpm_unet_image_workspace_auv.py \
  --config-name=train_ddpm_unet_image_workspace_auv \
  optimizer.lr=5e-5

# 4. 增加 warmup
training.lr_warmup_steps=2000

# 5. 检查标准化器
# 确保 normalizer 正确应用
```

### Q4: 推理输出不合理

**排查:**
```python
# 1. 验证反归一化
# 检查 predict_action() 最后的 unnormalize 步骤

# 2. 增加推理步数
policy.num_inference_steps=100

# 3. 使用 EMA 模型
policy = ema_model  # 而非 model

# 4. 检查条件编码
# 确保 global_cond 正确计算
```

---

## 📈 监测训练进度

### WandB 仪表板查看
```bash
# 自动上传到 WandB（需要配置 API key）
# 在配置中设置：logging.project 和 logging.entity

# 查看实时指标：
# - train_loss (每个 batch)
# - val_loss (每个 epoch)
# - train_action_mse_error (采样质量)
# - rollout_success_rate (环境验证)
```

### 本地日志查看
```bash
# 查看 JSON 日志
tail -f log_dir/logs.json.txt

# 解析日志
python -c "
import json
with open('logs.json.txt') as f:
    lines = f.readlines()
    for line in lines[-10:]:
        print(json.loads(line))
"
```

---

## 🎯 下一步

### 阶段 1: 验证基础功能 (1-2 小时)
- [ ] 成功启动训练脚本
- [ ] 观察 loss 下降趋势
- [ ] 生成第一个 checkpoint

### 阶段 2: 超参调优 (1-2 天)
- [ ] 尝试不同 batch sizes
- [ ] 测试不同 num_inference_steps (20, 50, 100)
- [ ] 对比 Flow Matching vs DDPM 输出

### 阶段 3: 生产部署 (可选)
- [ ] 选择最佳方法 (FM 速度 vs DDPM 质量)
- [ ] 模型蒸馏或量化
- [ ] 集成到 AUV 硬件系统

---

## 📚 相关文档

- 详细实现对比: `DDPM_vs_FlowMatching_CodeComparison.md`
- 理论对比: `DDPM_vs_FlowMatching_Comparison.md`
- Flow Matching 原版: `examples/diffusion/train.py`
- AUV 任务配置: `diffusion_policy/config/task/track_image.yaml`

---

## 💡 技巧与最佳实践

### 技巧 1: 快速调试模式
```bash
python train_ddpm_unet_image_workspace_auv.py \
  --config-name=train_ddpm_unet_image_workspace_auv \
  training.debug=True
```
这会自动设置:
- `num_epochs=2`
- `max_train_steps=3`
- `max_val_steps=3`
- `rollout_every=1`

### 技巧 2: 恢复训练
```bash
# 自动从最新 checkpoint 恢复
python train_ddpm_unet_image_workspace_auv.py \
  --config-name=train_ddpm_unet_image_workspace_auv \
  training.resume=True
```

### 技巧 3: 直接加载预训练权重 (混合初始化)
```python
# 在 workspace 的 __init__ 中添加
state_dict = torch.load('path/to/pretrained.pt')
# 加载支持的权重
self.model.load_state_dict(state_dict, strict=False)
```

### 技巧 4: 多 GPU 训练 (进阶)
```bash
# 修改配置使用多 GPU（需要 DistributedDataParallel）
training.device="cuda:0"  # 修改为使用更多 GPU
# 注：目前脚本单 GPU，如需多 GPU 支持请反馈
```

---

## 🔗 资源链接

- Diffusers 文档: https://huggingface.co/docs/diffusers
- DDPM 原始论文: https://arxiv.org/abs/2006.11239
- Flow Matching 论文: https://arxiv.org/abs/2210.02747
- PyTorch 官方文档: https://pytorch.org/docs/stable/index.html

---

## ✅ 验收清单

使用本脚本前确保：

- [ ] Python ≥ 3.8
- [ ] PyTorch ≥ 1.12 with CUDA support
- [ ] diffusers ≥ 0.20.0
- [ ] 8GB+ GPU 显存
- [ ] 数据集已准备 (图像 + 动作标注)
- [ ] 环境变量已配置 (如需 WandB)

---

完成上述步骤后，你就可以：

✨ **用 DDPM 算法训练 AUV 控制策略**

Enjoy! 🚀
