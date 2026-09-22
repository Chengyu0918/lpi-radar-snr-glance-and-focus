"""
CWD去噪器 - 基于JiT的条件扩散模型
用于将含噪CWD图去噪为干净CWD图
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_jit_cwd import JiT_CWD_models


class CWDDenoiser(nn.Module):
    """
    CWD图像去噪器
    
    训练时：
    - 输入：干净CWD图 (clean_cwd) 和 含噪CWD图 (noisy_cwd)
    - 过程：对clean_cwd添加扩散噪声得到z_t，以noisy_cwd为条件预测clean_cwd
    - 损失：v-prediction loss
    
    推理时：
    - 输入：含噪CWD图 (noisy_cwd)
    - 过程：从随机噪声开始，以noisy_cwd为条件，通过ODE求解器逐步去噪
    - 输出：去噪后的CWD图
    """
    def __init__(self, args):
        super().__init__()
        
        # 创建模型
        self.net = JiT_CWD_models[args.model](
            input_size=args.img_size,
            in_channels=args.in_channels,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
        )
        
        self.img_size = args.img_size
        self.in_channels = args.in_channels
        
        # 扩散参数
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale
        
        # EMA参数
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None
        
        # 采样参数
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps

    def sample_t(self, n: int, device=None):
        """采样时间步 t，使用logit-normal分布"""
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, clean_cwd, noisy_cwd):
        """
        训练前向传播
        
        Args:
            clean_cwd: 干净CWD图 [B, C, H, W]，作为目标x
            noisy_cwd: 含噪CWD图 [B, C, H, W]，作为条件cond
        
        Returns:
            loss: v-prediction损失
        """
        B = clean_cwd.size(0)
        device = clean_cwd.device
        
        # 1) 采样时间步 t (0, 1)
        t = self.sample_t(B, device=device).view(-1, 1, 1, 1)  # [B, 1, 1, 1]
        
        # 2) 采样噪声 eps
        eps = torch.randn_like(clean_cwd) * self.noise_scale
        
        # 3) 构造 z_t = t * x + (1 - t) * eps
        z = t * clean_cwd + (1 - t) * eps
        
        # 4) 真实速度 v = x - eps = (x - z) / (1 - t)
        #    或者直接用 v = clean_cwd - eps
        v = clean_cwd - eps
        
        # 5) 模型预测 x_pred，以noisy_cwd为条件
        x_pred = self.net(z, t.flatten(), cond=noisy_cwd)
        
        # 6) 转成 v_pred = (x_pred - z) / (1 - t)
        one_minus_t = (1 - t).clamp_min(self.t_eps)
        v_pred = (x_pred - z) / one_minus_t
        
        # 7) 损失：MSE(v_pred, v)
        loss = F.mse_loss(v_pred, v)
        
        return loss

    @torch.no_grad()
    def denoise(self, noisy_cwd):
        """
        推理：对含噪CWD图进行去噪
        
        Args:
            noisy_cwd: 含噪CWD图 [B, C, H, W]
        
        Returns:
            denoised: 去噪后的CWD图 [B, C, H, W]
        """
        device = noisy_cwd.device
        B = noisy_cwd.size(0)
        
        # 初始化 z 为随机噪声
        z = self.noise_scale * torch.randn(
            B, self.in_channels, self.img_size, self.img_size, 
            device=device
        )
        
        # 时间步序列
        timesteps = torch.linspace(0.0, 1.0, self.steps + 1, device=device)
        
        # 选择ODE求解器
        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError(f"Unknown sampling method: {self.method}")
        
        # ODE求解
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, noisy_cwd)
        
        # 最后一步用Euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], noisy_cwd)
        
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, noisy_cwd):
        """
        单步前向采样，计算速度场v
        
        Args:
            z: 当前状态 [B, C, H, W]
            t: 时间步 (标量)
            noisy_cwd: 条件图 [B, C, H, W]
        
        Returns:
            v_pred: 预测的速度场 [B, C, H, W]
        """
        B = z.size(0)
        t_batch = t.expand(B)  # [B]
        
        # 模型预测
        x_pred = self.net(z, t_batch, cond=noisy_cwd)
        
        # 转换为速度
        one_minus_t = (1.0 - t).clamp_min(self.t_eps)
        v_pred = (x_pred - z) / one_minus_t
        
        return v_pred

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, noisy_cwd):
        """Euler方法ODE步进"""
        v_pred = self._forward_sample(z, t, noisy_cwd)
        dt = t_next - t
        z_next = z + dt * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, noisy_cwd):
        """Heun方法ODE步进（二阶精度）"""
        # 第一步：Euler预测
        v_pred_t = self._forward_sample(z, t, noisy_cwd)
        dt = t_next - t
        z_next_euler = z + dt * v_pred_t
        
        # 第二步：在预测点计算速度
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, noisy_cwd)
        
        # 平均速度
        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + dt * v_pred
        
        return z_next

    @torch.no_grad()
    def update_ema(self):
        """更新EMA参数"""
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
