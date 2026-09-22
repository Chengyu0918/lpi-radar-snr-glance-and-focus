"""
带类别条件的CWD去噪器封装类
"""
import torch
import torch.nn as nn
import numpy as np

from model_jit_cwd_class import JiT_CWD_Class_models


class CWDDenoiserWithClass(nn.Module):
    """
    带类别条件的CWD去噪器
    封装JiT模型，提供训练和推理接口
    """
    def __init__(self, args):
        super().__init__()
        
        # 创建模型
        model_name = getattr(args, 'model', 'JiT-CWD-Class-Tiny')
        if model_name not in JiT_CWD_Class_models:
            model_name = 'JiT-CWD-Class-Tiny'
        
        self.model = JiT_CWD_Class_models[model_name](
            img_size=getattr(args, 'img_size', 224),
            in_channels=getattr(args, 'in_channels', 1),
            num_classes=getattr(args, 'num_classes', 16),
            class_dropout_prob=getattr(args, 'class_dropout_prob', 0.1),
            attn_dropout=getattr(args, 'attn_dropout', 0.0),
            proj_dropout=getattr(args, 'proj_dropout', 0.0),
        )
        
        # 扩散参数
        self.P_mean = getattr(args, 'P_mean', -0.8)
        self.P_std = getattr(args, 'P_std', 0.8)
        self.noise_scale = getattr(args, 'noise_scale', 1.0)
        self.t_eps = getattr(args, 't_eps', 0.05)
        
        # 采样参数
        self.sampling_method = getattr(args, 'sampling_method', 'heun')
        self.num_sampling_steps = getattr(args, 'num_sampling_steps', 50)
        
        # 类别名称映射
        self.class_names = [
            'BPSK', 'Costas', 'CP', 'Frank', 
            'FSK4_Baker5', 'FSK4_LFM', 'LFM', 'NLFM',
            'P1', 'P2', 'P3', 'P4', 
            'T1', 'T2', 'T3', 'T4'
        ]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
    def get_class_idx(self, class_name):
        """从类别名称获取索引"""
        return self.class_to_idx.get(class_name, 0)
    
    def forward(self, clean_cwd, noisy_cwd, class_labels):
        """
        训练前向传播
        
        Args:
            clean_cwd: [B, 1, H, W] 干净CWD图
            noisy_cwd: [B, 1, H, W] 含噪CWD图（作为条件）
            class_labels: [B] 类别标签
        
        Returns:
            loss: 训练损失
        """
        B = clean_cwd.shape[0]
        device = clean_cwd.device
        
        # 采样时间步 t ~ LogNormal(P_mean, P_std)
        log_t = torch.randn(B, device=device) * self.P_std + self.P_mean
        t = torch.sigmoid(log_t)  # 映射到 (0, 1)
        t = t.clamp(self.t_eps, 1 - self.t_eps)  # 避免边界
        
        # 采样噪声
        noise = torch.randn_like(clean_cwd) * self.noise_scale
        
        # 构造 z_t = t * clean + (1 - t) * noise
        t_expand = t.view(B, 1, 1, 1)
        z_t = t_expand * clean_cwd + (1 - t_expand) * noise
        
        # 目标速度场 v = clean - noise
        v_target = clean_cwd - noise
        
        # 模型预测
        v_pred = self.model(z_t, t, noisy_cwd, class_labels)
        
        # MSE损失
        loss = ((v_pred - v_target) ** 2).mean()
        
        return loss
    
    @torch.no_grad()
    def denoise(self, noisy_cwd, class_labels, cfg_scale=1.0):
        """
        去噪推理
        
        Args:
            noisy_cwd: [B, 1, H, W] 含噪CWD图
            class_labels: [B] 类别标签
            cfg_scale: classifier-free guidance强度
        
        Returns:
            denoised: [B, 1, H, W] 去噪后的图像
        """
        B = noisy_cwd.shape[0]
        device = noisy_cwd.device
        
        # 从纯噪声开始
        z = torch.randn_like(noisy_cwd) * self.noise_scale
        
        # 时间步
        ts = torch.linspace(self.t_eps, 1 - self.t_eps, self.num_sampling_steps, device=device)
        dt = 1.0 / self.num_sampling_steps
        
        # ODE求解
        for i, t in enumerate(ts):
            t_batch = torch.full((B,), t.item(), device=device)
            
            if cfg_scale > 1.0:
                # Classifier-free guidance
                # 有条件预测
                v_cond = self.model(z, t_batch, noisy_cwd, class_labels)
                # 无条件预测 (使用num_classes作为无条件标签)
                uncond_labels = torch.full_like(class_labels, self.model.num_classes)
                v_uncond = self.model(z, t_batch, noisy_cwd, uncond_labels)
                # 引导
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = self.model(z, t_batch, noisy_cwd, class_labels)
            
            if self.sampling_method == 'euler':
                z = z + v * dt
            elif self.sampling_method == 'heun':
                # Heun方法（二阶）
                z_next = z + v * dt
                if i < len(ts) - 1:
                    t_next = ts[i + 1]
                    t_next_batch = torch.full((B,), t_next.item(), device=device)
                    
                    if cfg_scale > 1.0:
                        v_next_cond = self.model(z_next, t_next_batch, noisy_cwd, class_labels)
                        v_next_uncond = self.model(z_next, t_next_batch, noisy_cwd, uncond_labels)
                        v_next = v_next_uncond + cfg_scale * (v_next_cond - v_next_uncond)
                    else:
                        v_next = self.model(z_next, t_next_batch, noisy_cwd, class_labels)
                    
                    z = z + (v + v_next) * dt / 2
                else:
                    z = z_next
        
        return z
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path, device='cuda'):
        """
        从检查点加载模型
        
        Args:
            checkpoint_path: 检查点文件路径
            device: 设备
        
        Returns:
            加载好的去噪器
        """
        import argparse
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 从检查点获取参数，或使用默认值
        if 'args' in checkpoint:
            args = checkpoint['args']
        else:
            # 创建默认参数
            args = argparse.Namespace(
                model='JiT-CWD-Class-Tiny',
                img_size=224,
                in_channels=1,
                num_classes=16,
                class_dropout_prob=0.1,
                attn_dropout=0.0,
                proj_dropout=0.0,
                P_mean=-0.8,
                P_std=0.8,
                noise_scale=1.0,
                t_eps=0.05,
                sampling_method='heun',
                num_sampling_steps=50,
            )
        
        # 创建模型
        denoiser = cls(args)
        
        # 加载权重
        if 'model_state_dict' in checkpoint:
            denoiser.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'ema_state_dict' in checkpoint:
            denoiser.model.load_state_dict(checkpoint['ema_state_dict'])
        elif 'state_dict' in checkpoint:
            denoiser.model.load_state_dict(checkpoint['state_dict'])
        else:
            # 尝试直接加载
            denoiser.model.load_state_dict(checkpoint)
        
        denoiser = denoiser.to(device)
        denoiser.eval()
        
        return denoiser
