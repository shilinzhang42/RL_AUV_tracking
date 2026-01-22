if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import tqdm
import numpy as np
from torch.amp import GradScaler, autocast # 确保导入路径正确

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.flow_matching_unet_image_policy import FlowMatchingUnetImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainFlowMatchingUnetImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # 设置随机种子
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # 实例化策略
        self.model: FlowMatchingUnetImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: FlowMatchingUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # 配置优化器
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # 断点续传逻辑
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # 配置学习率调度器
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs) // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step-1
        )

        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # 配置日志 (SwanLab)
        if cfg.logging.repo == 'swanlab':
            import swanlab
            run_logger = swanlab.init(
                project=cfg.logging.project,
                experiment_name=cfg.logging.name,
                logdir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
            )
        else:
            import wandb
            run_logger = wandb.init(dir=str(self.output_dir), config=OmegaConf.to_container(cfg, resolve=True), **cfg.logging)

        topk_manager = TopKCheckpointManager(save_dir=os.path.join(self.output_dir, 'checkpoints'), **cfg.checkpoint.topk)

        # 设备迁移
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # 【新增】混合精度缩放器 (AMP)
        scaler = GradScaler(device='cuda')

        train_sampling_batch = None
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                self.model.train()
                
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # 【重要】Batch 进入 GPU 时保持 uint8
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        
                        # 采样用的样本存放在 CPU 以节省显存
                        if train_sampling_batch is None:
                            train_sampling_batch = dict_apply(batch, lambda x: x.cpu())

                        # 【核心优化】使用混合精度计算
                        with autocast(device_type='cuda'):
                            raw_loss = self.model.compute_loss(batch)
                            loss = raw_loss / cfg.training.gradient_accumulate_every
                        
                        # 反向传播缩放
                        scaler.scale(loss).backward()

                        if (self.global_step + 1) % cfg.training.gradient_accumulate_every == 0:
                            scaler.step(self.optimizer)
                            scaler.update()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        
                        # 记录全局步数日志
                        if self.global_step % 10 == 0:
                            step_log = {
                                'train_loss': raw_loss_cpu,
                                'global_step': self.global_step,
                                'epoch': self.epoch,
                                'lr': lr_scheduler.get_last_lr()[0]
                            }
                            run_logger.log(step_log, step=self.global_step)

                        self.global_step += 1

                # ========= Epoch 级验证与采样 (内存保护逻辑) ==========
                policy = self.ema_model if cfg.training.use_ema else self.model
                policy.eval()

                # 1. 验证集评估
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad(), autocast(device_type='cuda'):
                        val_losses = []
                        for val_batch in val_dataloader:
                            val_batch = dict_apply(val_batch, lambda x: x.to(device, non_blocking=True))
                            val_losses.append(policy.compute_loss(val_batch).item())
                        if val_losses:
                            step_log['val_loss'] = np.mean(val_losses)


                # 2. 推理采样 (最吃内存的环节)
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # 从 CPU 样本中切片并推入 GPU
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = {k: v[:, :self.model.n_obs_steps] for k, v in batch['obs'].items()}
                        
                        # 执行推理
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        
                        # 计算 MSE 误差
                        start = self.model.n_obs_steps - 1
                        gt_action_slice = batch['action'][:, start:start + self.model.n_action_steps]
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action_slice)
                        step_log['train_action_mse_error'] = mse.item()
                        
                        # 【核心修改】显式释放大对象并清理显存缓存，防止下个 Epoch 初始内存抖动
                        del batch, obs_dict, result, pred_action, gt_action_slice
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect() # 显式触发系统垃圾回收 (非常重要)
                        
                run_logger.log(step_log, step=self.global_step)
                # 3. 保存 Checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    self.save_checkpoint()
                    topk_manager.get_ckpt_path(step_log)

                self.epoch += 1

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    # [修改] 实例化新的类
    workspace = TrainFlowMatchingUnetImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()