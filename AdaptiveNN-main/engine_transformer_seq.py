"""
Transformer时序建模版本的训练引擎
支持轨迹多样性约束L_div
"""
import math
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy
import torch.nn.functional as F

import utils


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None,
                    args=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    # 添加多样性损失指标
    metric_logger.add_meter('loss_div', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_cls', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_total', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    # 添加位置统计指标
    for focus_step_index in range(args.seq_l):
        metric_logger.add_meter(f'pos_x_mean_{focus_step_index}', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter(f'pos_y_mean_{focus_step_index}', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter(f'pos_x_std_{focus_step_index}', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter(f'pos_y_std_{focus_step_index}', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if epoch < param_group['fix_step']:
                    param_group["lr"] = 0.
                elif lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            # 前向传播
            expected_outputs = model(samples, seq_l=args.seq_l)

            # ============ 计算分类损失 ============
            # 1. Focus网络正则化损失
            outputs_reg_focus_net = expected_outputs['outputs_reg_focus_net']
            loss_reg_focus_net = criterion(outputs_reg_focus_net, targets)
            loss_reg_focus_net = args.loss_reg_focus_net_weight * loss_reg_focus_net

            # 2. Glance损失
            loss_glance = criterion(expected_outputs['x_glance'][-1], targets)
            
            # 3. Focus损失（所有steps）
            loss_focus = sum(
                [criterion(_x_focus, targets) for _x_focus in expected_outputs['x_focus']]
            )
            
            # 4. 最终分类损失
            if expected_outputs['x_final'] is not None:
                loss_final = criterion(expected_outputs['x_final'], targets)
            else:
                loss_final = 0
            
            # 5. KD损失（可选）
            if expected_outputs['x_final'] is not None:
                out_teacher = expected_outputs['x_final'].detach()
            else:
                out_teacher = expected_outputs['x_focus'][-1].detach()
                
            loss_KD = F.kl_div(
                F.log_softmax(expected_outputs['x_glance'][-1] / args.kd_temp, dim=1),
                F.softmax(out_teacher / args.kd_temp, dim=1), 
                reduction='batchmean'
            ) * (args.kd_temp ** 2)

            loss_KD = loss_KD + sum(
                [
                    F.kl_div(
                        F.log_softmax(_x_focus / args.kd_temp, dim=1),
                        F.softmax(out_teacher / args.kd_temp, dim=1), 
                        reduction='batchmean'
                    ) * (args.kd_temp ** 2)
                    for _x_focus in expected_outputs['x_focus'][:-1]
                ]
            )

            # 总分类损失
            loss_cls = loss_reg_focus_net + loss_focus + loss_glance + loss_final + loss_KD * args.kd_alpha

            # ============ 计算轨迹多样性损失 ============
            diversity_loss = expected_outputs.get('diversity_loss', torch.tensor(0.0, device=device))
            
            # 获取多样性权重（从args或使用默认值）
            lambda_div = getattr(args, 'diversity_weight', 0.1)
            
            loss_div = lambda_div * diversity_loss

            # ============ 注意力熵正则化（可选，鼓励探索） ============
            if hasattr(args, 'attention_entropy_weight') and args.attention_entropy_weight > 0:
                attention_entropy = 0
                for prob in expected_outputs['attention_probs']:
                    # 计算熵: -sum(p * log(p))
                    entropy = -(prob * torch.log(prob + 1e-8)).sum(dim=-1).mean()
                    attention_entropy += entropy
                attention_entropy = attention_entropy / len(expected_outputs['attention_probs'])
                loss_entropy = -args.attention_entropy_weight * attention_entropy  # 负号因为我们要最大化熵
            else:
                loss_entropy = 0

            # ============ 总损失 ============
            loss = loss_cls + loss_div + loss_entropy
            loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        # 梯度更新
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss /= update_freq
        grad_norm = loss_scaler(loss, optimizer, 
                                clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(data_iter_step + 1) % update_freq == 0)
        if (data_iter_step + 1) % update_freq == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        # 更新指标
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_cls=loss_cls.item())
        metric_logger.update(loss_div=diversity_loss.item() if torch.is_tensor(diversity_loss) else diversity_loss)
        metric_logger.update(loss_total=loss_value)
        
        for focus_step_index in range(args.seq_l):
            if focus_step_index < len(expected_outputs['pos_std']):
                metric_logger.update(**{f'pos_x_std_{focus_step_index}': expected_outputs['pos_std'][focus_step_index][1].item()})
                metric_logger.update(**{f'pos_y_std_{focus_step_index}': expected_outputs['pos_std'][focus_step_index][0].item()})
                metric_logger.update(**{f'pos_x_mean_{focus_step_index}': expected_outputs['pos_mean'][focus_step_index][1].item()})
                metric_logger.update(**{f'pos_y_mean_{focus_step_index}': expected_outputs['pos_mean'][focus_step_index][0].item()})
        
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_cls=loss_cls.item(), head="loss")
            log_writer.update(loss_div=diversity_loss.item() if torch.is_tensor(diversity_loss) else diversity_loss, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if wandb_logger:
            wandb_logger._wandb.log({
                'Rank-0 Batch Wise/train_loss': loss_value,
                'Rank-0 Batch Wise/train_loss_cls': loss_cls.item(),
                'Rank-0 Batch Wise/train_loss_div': diversity_loss.item() if torch.is_tensor(diversity_loss) else diversity_loss,
                'Rank-0 Batch Wise/train_max_lr': max_lr,
                'Rank-0 Batch Wise/train_min_lr': min_lr
            }, commit=False)
            wandb_logger._wandb.log({'Rank-0 Batch Wise/train_grad_norm': grad_norm}, commit=False)
            wandb_logger._wandb.log({'Rank-0 Batch Wise/global_train_step': it})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False, args=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    # 用于统计位置分布
    all_positions = [[] for _ in range(args.seq_l)]
    
    for batch in metric_logger.log_every(data_loader, 10, header):
        
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            expected_outputs = model(images, seq_l=args.seq_l)
            
            # 使用最终分类输出或最后一个focus输出
            if expected_outputs['x_final'] is not None:
                output = expected_outputs['x_final']
            else:
                output = expected_outputs['x_focus'][-1]
            
            output_glance = expected_outputs['x_glance'][-1]
            loss = criterion(output, target)
            loss_glance = criterion(output_glance, target)
            
            # 收集位置信息
            for t, pos in enumerate(expected_outputs['actions']):
                all_positions[t].append(pos.cpu())

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        acc1_glance, acc5_glance = accuracy(output_glance, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        # 每个step的准确率
        for focus_step_index in range(args.seq_l):
            if focus_step_index < len(expected_outputs['x_focus']):
                acc1_this_step, acc5_this_step = accuracy(
                    expected_outputs['x_focus'][focus_step_index], target, topk=(1, 5)
                )
                metric_logger.meters[f'acc1_step_{focus_step_index}'].update(acc1_this_step.item(), n=batch_size)
                metric_logger.meters[f'acc5_step_{focus_step_index}'].update(acc5_this_step.item(), n=batch_size)
        
        metric_logger.update(loss_glance=loss_glance.item())
        metric_logger.meters['acc1_glance'].update(acc1_glance.item(), n=batch_size)
        metric_logger.meters['acc5_glance'].update(acc5_glance.item(), n=batch_size)
        
        # 多样性损失
        if expected_outputs['diversity_loss'] is not None:
            metric_logger.meters['diversity_loss'].update(
                expected_outputs['diversity_loss'].item(), n=batch_size
            )
    
    # 计算位置分布统计
    print("\n=== Position Distribution Statistics ===")
    for t in range(args.seq_l):
        if all_positions[t]:
            positions_t = torch.cat(all_positions[t], dim=0)  # (N, 2)
            mean_pos = positions_t.mean(dim=0)
            std_pos = positions_t.std(dim=0)
            print(f"Step {t}: Mean position = ({mean_pos[0]:.4f}, {mean_pos[1]:.4f}), "
                  f"Std = ({std_pos[0]:.4f}, {std_pos[1]:.4f})")
    
    # 计算步间位置多样性
    if args.seq_l > 1 and all(len(p) > 0 for p in all_positions):
        all_pos_stacked = [torch.cat(p, dim=0) for p in all_positions]  # List of (N, 2)
        mean_positions = torch.stack([p.mean(dim=0) for p in all_pos_stacked])  # (T, 2)
        
        # 计算平均步间距离
        total_dist = 0
        count = 0
        for i in range(len(mean_positions)):
            for j in range(i+1, len(mean_positions)):
                dist = torch.norm(mean_positions[i] - mean_positions[j])
                total_dist += dist
                count += 1
        if count > 0:
            avg_inter_step_dist = total_dist / count
            print(f"Average inter-step distance: {avg_inter_step_dist:.4f}")
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    for focus_step_index in range(args.seq_l):
        if f'acc1_step_{focus_step_index}' in metric_logger.meters:
            print(f'Step {focus_step_index} Acc@1: {metric_logger.meters[f"acc1_step_{focus_step_index}"].global_avg:.3f}')

    print('* Glance Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1_glance, top5=metric_logger.acc5_glance, losses=metric_logger.loss_glance))
    
    if 'diversity_loss' in metric_logger.meters:
        print(f'* Diversity Loss: {metric_logger.meters["diversity_loss"].global_avg:.6f}')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def visualize_attention(model, images, device, args, save_path=None):
    """
    可视化注意力分布和选择的位置
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    model.eval()
    images = images.to(device)
    
    with torch.cuda.amp.autocast():
        expected_outputs = model(images, seq_l=args.seq_l)
    
    B = images.size(0)
    num_samples = min(B, 4)  # 最多显示4个样本
    
    fig, axes = plt.subplots(num_samples, args.seq_l + 1, figsize=(4*(args.seq_l+1), 4*num_samples))
    
    for b in range(num_samples):
        # 原图
        img = images[b].cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())
        
        axes[b, 0].imshow(img)
        axes[b, 0].set_title('Original')
        axes[b, 0].axis('off')
        
        # 每个step的注意力
        for t in range(args.seq_l):
            attn = expected_outputs['attention_probs'][t][b].cpu().numpy()
            H = W = int(np.sqrt(len(attn)))
            attn = attn.reshape(H, W)
            
            # 叠加注意力到原图
            axes[b, t+1].imshow(img)
            axes[b, t+1].imshow(attn, alpha=0.5, cmap='hot', 
                               extent=[0, img.shape[1], img.shape[0], 0])
            
            # 标记选择的位置
            pos = expected_outputs['actions'][t][b].cpu().numpy()
            y_pos = pos[0] * img.shape[0]
            x_pos = pos[1] * img.shape[1]
            axes[b, t+1].scatter([x_pos], [y_pos], c='blue', s=100, marker='x')
            
            axes[b, t+1].set_title(f'Step {t+1}')
            axes[b, t+1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Attention visualization saved to {save_path}")
    
    plt.close()
    
    return fig
