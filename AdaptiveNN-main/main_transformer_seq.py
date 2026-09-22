"""
Transformer时序建模版本的主训练脚本
支持轨迹多样性约束L_div
"""
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os

from pathlib import Path

from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from optim_factory import create_optimizer, LayerDecayValueAssigner

from dataset_utils import build_dataset
from engine_transformer_seq import train_one_epoch, evaluate, visualize_attention

from utils import NativeScalerWithGradNormCount as NativeScaler
import utils

# 导入模型
from models.dynamic_deitS_transformer_seq import dynamic_deitS_transformer_seq


def load_pretrained_weights(model, pretrained_path, device='cpu'):
    """
    从PPO版本的checkpoint加载glance_net和focus_net的预训练权重
    跳过不匹配的键（如sequence_transformer相关的新模块）
    """
    print(f"\n{'='*60}")
    print(f"加载预训练权重: {pretrained_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(pretrained_path):
        print(f"⚠ 预训练文件不存在: {pretrained_path}")
        return model, 0
    
    checkpoint = torch.load(pretrained_path, map_location=device)
    
    # 尝试不同的键名获取state_dict
    pretrained_state_dict = None
    for key in ['model', 'model_ema', 'state_dict', 'module']:
        if key in checkpoint:
            pretrained_state_dict = checkpoint[key]
            print(f"✓ 使用键名 '{key}' 获取预训练权重")
            break
    
    if pretrained_state_dict is None:
        pretrained_state_dict = checkpoint
        print("✓ 直接使用checkpoint作为state_dict")
    
    # 处理DDP模型的键名（去除module.前缀）
    new_pretrained_state_dict = {}
    for k, v in pretrained_state_dict.items():
        if k.startswith('module.'):
            new_pretrained_state_dict[k[7:]] = v
        else:
            new_pretrained_state_dict[k] = v
    pretrained_state_dict = new_pretrained_state_dict
    
    # 获取当前模型的state_dict
    model_state_dict = model.state_dict()
    
    # 选择性加载权重
    loaded_keys = []
    skipped_keys = []
    mismatched_keys = []
    
    # 定义要加载的模块前缀
    modules_to_load = ['glance_net', 'focus_net', 'multi_cls']
    
    for name, param in pretrained_state_dict.items():
        # 检查是否是要加载的模块
        should_load = any(module in name for module in modules_to_load)
        
        if should_load and name in model_state_dict:
            if param.shape == model_state_dict[name].shape:
                model_state_dict[name] = param
                loaded_keys.append(name)
            else:
                mismatched_keys.append((name, param.shape, model_state_dict[name].shape))
        else:
            skipped_keys.append(name)
    
    # 加载更新后的state_dict
    model.load_state_dict(model_state_dict, strict=False)
    
    # 打印加载统计
    print(f"\n加载统计:")
    print(f"  ✓ 成功加载: {len(loaded_keys)} 个参数")
    print(f"  ○ 跳过: {len(skipped_keys)} 个参数 (新模块或不需要)")
    print(f"  ✗ 形状不匹配: {len(mismatched_keys)} 个参数")
    
    if loaded_keys:
        print(f"\n成功加载的模块:")
        loaded_modules = set()
        for key in loaded_keys:
            module_name = key.split('.')[0]
            loaded_modules.add(module_name)
        for module in sorted(loaded_modules):
            count = sum(1 for k in loaded_keys if k.startswith(module))
            print(f"  - {module}: {count} 个参数")
    
    if mismatched_keys:
        print(f"\n形状不匹配的参数:")
        for name, pre_shape, cur_shape in mismatched_keys:
            print(f"  - {name}: 预训练{pre_shape} vs 当前{cur_shape}")
    
    print(f"{'='*60}\n")
    
    return model, len(loaded_keys)


def freeze_modules(model, modules_to_freeze):
    """冻结指定模块的参数"""
    frozen_count = 0
    for name, param in model.named_parameters():
        if any(module in name for module in modules_to_freeze):
            param.requires_grad = False
            frozen_count += 1
    return frozen_count


def unfreeze_modules(model, modules_to_unfreeze):
    """解冻指定模块的参数"""
    unfrozen_count = 0
    for name, param in model.named_parameters():
        if any(module in name for module in modules_to_unfreeze):
            param.requires_grad = True
            unfrozen_count += 1
    return unfrozen_count


def get_args_parser():
    parser = argparse.ArgumentParser('Transformer-Seq AdaptiveNN training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--update_freq', default=1, type=int)

    # Model parameters
    parser.add_argument('--model', default='dynamic_deitS_transformer_seq', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # EMA related parameters
    parser.add_argument('--model_ema', type=utils.str2bool, default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', type=utils.str2bool, default=False, help='')
    parser.add_argument('--model_ema_eval', type=utils.str2bool, default=False, help='Using ema to eval during training.')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--layer_decay', type=float, default=1.0)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". (default: rand-m9-mstd0.5-inc1)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', type=utils.str2bool, default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Dataset parameters
    parser.add_argument('--data_path', default='./CWDtrain', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default='./CWDtest', type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=16, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', type=utils.str2bool, default=True)
    parser.add_argument('--data_set', default='IMNET', choices=['CIFAR', 'IMNET', 'image_folder'],
                        type=str, help='ImageNet dataset path')
    parser.add_argument('--output_dir', default='./output_transformer_seq',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', type=utils.str2bool, default=True)
    parser.add_argument('--save_ckpt', type=utils.str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_num', default=3, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', type=utils.str2bool, default=False,
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', type=utils.str2bool, default=True,
                        help='Enabling distributed evaluation')
    parser.add_argument('--disable_eval', type=utils.str2bool, default=False,
                        help='Disabling evaluation during training')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', type=utils.str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=utils.str2bool, default=False)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--use_amp', type=utils.str2bool, default=True, 
                        help="Use PyTorch's AMP (Automatic Mixed Precision) or not")

    # Weights and Biases arguments
    parser.add_argument('--enable_wandb', type=utils.str2bool, default=False,
                        help="enable logging to Weights and Biases")
    parser.add_argument('--project', default='adaptivenn', type=str,
                        help="The name of the W&B project where you're sending the new run.")
    parser.add_argument('--wandb_ckpt', type=utils.str2bool, default=False,
                        help="Save model checkpoints as W&B Artifacts.")

    # ============ AdaptiveNN 特定参数 ============
    parser.add_argument('--seq_l', default=4, type=int,
                        help='Number of focus steps')
    parser.add_argument('--feature_in_chans', default=384, type=int,
                        help='Feature channels (384 for DeiT-S)')
    parser.add_argument('--recover_n', default=3, type=int,
                        help='Recovery neighborhood size')
    parser.add_argument('--remaining_blocks', default=2, type=int,
                        help='Remaining blocks for classification')
    
    # Glance网络参数
    parser.add_argument('--glance_input_size', default=112, type=int,
                        help='Glance network input size')
    parser.add_argument('--glance_net_depth', default=6, type=int,
                        help='Glance network depth')
    parser.add_argument('--glance_net_mlp_ratio', default=4.0, type=float,
                        help='Glance network MLP ratio')
    parser.add_argument('--glance_net_drop_path', default=0.1, type=float,
                        help='Glance network drop path rate')
    
    # Focus网络参数
    parser.add_argument('--focus_patch_size', default=96, type=int,
                        help='Focus patch size')
    parser.add_argument('--focus_net_reg_size', default=96, type=int,
                        help='Focus network regularization size')
    parser.add_argument('--focus_net_depth', default=6, type=int,
                        help='Focus network depth')
    parser.add_argument('--focus_net_mlp_ratio', default=4.0, type=float,
                        help='Focus network MLP ratio')
    parser.add_argument('--focus_net_drop_path', default=0.1, type=float,
                        help='Focus network drop path rate')
    
    # 分类头参数
    parser.add_argument('--multi_cls_drop_path', default=0.1, type=float,
                        help='Multi-classifier drop path rate')
    
    # 策略网络参数
    parser.add_argument('--policy_net_hidden_chans', default=256, type=int,
                        help='Policy network hidden channels')
    parser.add_argument('--policy_net_kernel_size', default=7, type=int,
                        help='Policy network kernel size')
    
    # ============ Transformer时序建模参数 ============
    parser.add_argument('--transformer_num_layers', default=3, type=int,
                        help='Number of Transformer encoder layers')
    parser.add_argument('--transformer_nhead', default=8, type=int,
                        help='Number of attention heads in Transformer')
    parser.add_argument('--transformer_dim_feedforward', default=1536, type=int,
                        help='Feedforward dimension in Transformer')
    parser.add_argument('--transformer_dropout', default=0.1, type=float,
                        help='Dropout rate in Transformer')
    parser.add_argument('--transformer_drop_path', default=0.1, type=float,
                        help='Drop path rate in Transformer')
    
    # ============ 轨迹多样性约束参数 ============
    parser.add_argument('--diversity_weight', default=0.1, type=float,
                        help='Weight for diversity regularization loss (lambda_div)')
    parser.add_argument('--diversity_sigma', default=0.3, type=float,
                        help='Sigma for RBF kernel in diversity loss')
    
    # ============ 其他训练参数 ============
    parser.add_argument('--loss_reg_focus_net_weight', default=0.5, type=float,
                        help='Weight for focus network regularization loss')
    parser.add_argument('--kd_alpha', default=0.5, type=float,
                        help='Weight for knowledge distillation loss')
    parser.add_argument('--kd_temp', default=4.0, type=float,
                        help='Temperature for knowledge distillation')
    parser.add_argument('--attention_entropy_weight', default=0.0, type=float,
                        help='Weight for attention entropy regularization (0 to disable)')
    
    # 学习率缩放
    parser.add_argument('--glance_net_lr_scale', default=1.0, type=float,
                        help='Learning rate scale for glance network')
    parser.add_argument('--focus_net_lr_scale', default=1.0, type=float,
                        help='Learning rate scale for focus network')
    parser.add_argument('--transformer_lr_scale', default=1.0, type=float,
                        help='Learning rate scale for sequence transformer')
    parser.add_argument('--policy_lr_scale', default=1.0, type=float,
                        help='Learning rate scale for policy head')
    
    # 冻结epoch
    parser.add_argument('--glance_net_fix_step', default=0, type=int,
                        help='Epochs to freeze glance network')
    parser.add_argument('--focus_net_fix_step', default=0, type=int,
                        help='Epochs to freeze focus network')
    parser.add_argument('--transformer_fix_step', default=0, type=int,
                        help='Epochs to freeze sequence transformer')
    parser.add_argument('--policy_fix_step', default=0, type=int,
                        help='Epochs to freeze policy head')
    
    # 可视化
    parser.add_argument('--visualize_freq', default=10, type=int,
                        help='Frequency of attention visualization (0 to disable)')
    
    # ============ 迁移学习参数 ============
    parser.add_argument('--pretrained_ppo', default='', type=str,
                        help='Path to pretrained PPO checkpoint for transfer learning')
    parser.add_argument('--freeze_pretrained_epochs', default=20, type=int,
                        help='Number of epochs to freeze pretrained modules (glance_net, focus_net)')
    parser.add_argument('--pretrained_lr_scale', default=0.1, type=float,
                        help='Learning rate scale for pretrained modules after unfreezing')

    return parser


def main(args):
    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = build_dataset(is_train=True, args=args)
    if args.disable_eval:
        args.dist_eval = False
        dataset_val = None
    else:
        dataset_val = build_dataset(is_train=False, args=args)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
    )
    print("Sampler_train = %s" % str(sampler_train))
    
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                  'This will slightly alter validation results as extra duplicate entries are added to achieve '
                  'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    if global_rank == 0 and args.enable_wandb:
        wandb_logger = utils.WandbLogger(args)
    else:
        wandb_logger = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_val = None

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    # 创建模型
    print(f"Creating model: {args.model}")
    model = dynamic_deitS_transformer_seq(
        seq_len=args.seq_l,
        feature_in_chans=args.feature_in_chans,
        recover_n=args.recover_n,
        remaining_blocks=args.remaining_blocks,
        glance_input_size=args.glance_input_size,
        glance_net_depth=args.glance_net_depth,
        glance_net_mlp_ratio=args.glance_net_mlp_ratio,
        glance_net_drop_path=args.glance_net_drop_path,
        focus_patch_size=args.focus_patch_size,
        focus_net_reg_size=args.focus_net_reg_size,
        focus_net_depth=args.focus_net_depth,
        focus_net_mlp_ratio=args.focus_net_mlp_ratio,
        focus_net_drop_path=args.focus_net_drop_path,
        multi_cls_drop_path=args.multi_cls_drop_path,
        policy_net_hidden_chans=args.policy_net_hidden_chans,
        policy_net_kernel_size=args.policy_net_kernel_size,
        # Transformer参数
        transformer_num_layers=args.transformer_num_layers,
        transformer_nhead=args.transformer_nhead,
        transformer_dim_feedforward=args.transformer_dim_feedforward,
        transformer_dropout=args.transformer_dropout,
        transformer_drop_path=args.transformer_drop_path,
        # 多样性参数
        diversity_sigma=args.diversity_sigma,
        # 其他
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
    )

    model.to(device)

    # ============ 迁移学习：加载预训练权重 ============
    pretrained_loaded = False
    # 标记是否已解冻（用于断点续训时的智能判断）
    modules_unfrozen = False
    
    if args.pretrained_ppo:
        model, loaded_count = load_pretrained_weights(model, args.pretrained_ppo, device)
        if loaded_count > 0:
            pretrained_loaded = True
            
            # ============ 智能冻结逻辑：考虑断点续训场景 ============
            # 判断是否处于需要冻结的阶段
            should_freeze = False
            if args.freeze_pretrained_epochs > 0 and args.start_epoch < args.freeze_pretrained_epochs:
                should_freeze = True
            
            if should_freeze:
                # 只有在当前 epoch 小于设定阈值时才冻结
                frozen_count = freeze_modules(model, ['glance_net', 'focus_net'])
                print(f"✓ 冻结 {frozen_count} 个预训练参数 (当前epoch {args.start_epoch} < {args.freeze_pretrained_epochs})")
                print(f"  将在第 {args.freeze_pretrained_epochs} 个epoch后解冻，开始联合微调")
                modules_unfrozen = False  # 标记为未解冻
            else:
                # 如果是从后面 epoch 恢复，或者不需要冻结，则确保参数是解冻的
                if args.freeze_pretrained_epochs > 0:
                    # 从解冻阶段恢复训练，确保参数 requires_grad = True
                    # （防止加载的模型状态里记录了 requires_grad=False）
                    unfrozen_count = unfreeze_modules(model, ['glance_net', 'focus_net'])
                    print(f"✓ 跳过冻结步骤 (当前epoch {args.start_epoch} >= {args.freeze_pretrained_epochs})")
                    print(f"  已确保 {unfrozen_count} 个预训练参数处于解冻状态")
                    modules_unfrozen = True  # 标记为已解冻（防止循环中再次重建优化器）
                else:
                    print("✓ 预训练权重已加载，不冻结任何模块")
                    modules_unfrozen = True

    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training steps per epoch = %d" % num_training_steps_per_epoch)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # 构建参数组
    param_groups = []
    
    # Glance网络参数
    glance_params = [p for n, p in model_without_ddp.named_parameters() if 'glance_net' in n and p.requires_grad]
    if glance_params:
        param_groups.append({
            'params': glance_params,
            'lr_scale': args.glance_net_lr_scale,
            'fix_step': args.glance_net_fix_step,
            'name': 'glance_net'
        })
    
    # Focus网络参数
    focus_params = [p for n, p in model_without_ddp.named_parameters() if 'focus_net' in n and p.requires_grad]
    if focus_params:
        param_groups.append({
            'params': focus_params,
            'lr_scale': args.focus_net_lr_scale,
            'fix_step': args.focus_net_fix_step,
            'name': 'focus_net'
        })
    
    # Transformer参数
    transformer_params = [p for n, p in model_without_ddp.named_parameters() 
                         if 'sequence_transformer' in n and p.requires_grad]
    if transformer_params:
        param_groups.append({
            'params': transformer_params,
            'lr_scale': args.transformer_lr_scale,
            'fix_step': args.transformer_fix_step,
            'name': 'sequence_transformer'
        })
    
    # 策略头参数
    policy_params = [p for n, p in model_without_ddp.named_parameters() 
                    if ('policy_head' in n or 'token_generator' in n) and p.requires_grad]
    if policy_params:
        param_groups.append({
            'params': policy_params,
            'lr_scale': args.policy_lr_scale,
            'fix_step': args.policy_fix_step,
            'name': 'policy'
        })
    
    # 其他参数
    other_param_names = set()
    for n, p in model_without_ddp.named_parameters():
        if p.requires_grad:
            if not any(key in n for key in ['glance_net', 'focus_net', 'sequence_transformer', 'policy_head', 'token_generator']):
                other_param_names.add(n)
    
    other_params = [p for n, p in model_without_ddp.named_parameters() if n in other_param_names]
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr_scale': 1.0,
            'fix_step': 0,
            'name': 'other'
        })

    optimizer = create_optimizer(args, param_groups, filter_bias_and_bn=True)
    loss_scaler = NativeScaler()

    print("Use Cosine LR scheduler")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )

    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    if args.eval:
        print(f"Eval only mode")
        test_stats = evaluate(data_loader_val, model, device, args=args)
        print(f"Accuracy of the model on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    max_accuracy = 0.0
    if args.model_ema and args.model_ema_eval:
        max_accuracy_ema = 0.0

    print("Start training for %d epochs" % args.epochs)
    start_time = time.time()
    
    # 注意：modules_unfrozen 变量已在迁移学习部分初始化
    # 如果没有使用预训练权重，则默认为 True（无需解冻操作）
    if not pretrained_loaded:
        modules_unfrozen = True
    
    for epoch in range(args.start_epoch, args.epochs):
        # ============ 迁移学习：在指定epoch解冻预训练模块 ============
        if pretrained_loaded and args.freeze_pretrained_epochs > 0 and not modules_unfrozen:
            if epoch >= args.freeze_pretrained_epochs:
                # 解冻预训练模块
                unfrozen_count = unfreeze_modules(model_without_ddp, ['glance_net', 'focus_net'])
                modules_unfrozen = True
                print(f"\n{'='*60}")
                print(f"Epoch {epoch}: 解冻预训练模块")
                print(f"  ✓ 解冻 {unfrozen_count} 个参数")
                print(f"  开始联合微调所有模块")
                print(f"{'='*60}\n")
                
                # 重建优化器以包含解冻的参数，并为预训练模块使用较小的学习率
                param_groups = []
                
                # Glance网络参数（使用较小学习率）
                glance_params = [p for n, p in model_without_ddp.named_parameters() if 'glance_net' in n and p.requires_grad]
                if glance_params:
                    param_groups.append({
                        'params': glance_params,
                        'lr_scale': args.pretrained_lr_scale,  # 使用较小的学习率
                        'fix_step': 0,
                        'name': 'glance_net'
                    })
                
                # Focus网络参数（使用较小学习率）
                focus_params = [p for n, p in model_without_ddp.named_parameters() if 'focus_net' in n and p.requires_grad]
                if focus_params:
                    param_groups.append({
                        'params': focus_params,
                        'lr_scale': args.pretrained_lr_scale,  # 使用较小的学习率
                        'fix_step': 0,
                        'name': 'focus_net'
                    })
                
                # Transformer参数
                transformer_params = [p for n, p in model_without_ddp.named_parameters() 
                                     if 'sequence_transformer' in n and p.requires_grad]
                if transformer_params:
                    param_groups.append({
                        'params': transformer_params,
                        'lr_scale': args.transformer_lr_scale,
                        'fix_step': 0,
                        'name': 'sequence_transformer'
                    })
                
                # 策略头参数
                policy_params = [p for n, p in model_without_ddp.named_parameters() 
                                if ('policy_head' in n or 'token_generator' in n) and p.requires_grad]
                if policy_params:
                    param_groups.append({
                        'params': policy_params,
                        'lr_scale': args.policy_lr_scale,
                        'fix_step': 0,
                        'name': 'policy'
                    })
                
                # 其他参数
                other_param_names = set()
                for n, p in model_without_ddp.named_parameters():
                    if p.requires_grad:
                        if not any(key in n for key in ['glance_net', 'focus_net', 'sequence_transformer', 'policy_head', 'token_generator']):
                            other_param_names.add(n)
                
                other_params = [p for n, p in model_without_ddp.named_parameters() if n in other_param_names]
                if other_params:
                    param_groups.append({
                        'params': other_params,
                        'lr_scale': 1.0,
                        'fix_step': 0,
                        'name': 'other'
                    })
                
                # 重建优化器
                optimizer = create_optimizer(args, param_groups, filter_bias_and_bn=True)
                print(f"  ✓ 优化器已重建，预训练模块学习率缩放: {args.pretrained_lr_scale}")
        
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        
        if wandb_logger:
            wandb_logger.set_steps()
        
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, mixup_fn,
            log_writer=log_writer, wandb_logger=wandb_logger, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
            args=args
        )
        
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)
        
        if data_loader_val is not None and not args.disable_eval:
            test_stats = evaluate(data_loader_val, model, device, args=args)
            print(f"Accuracy of the model on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                # 最佳模型保存逻辑独立于 save_ckpt 参数
                # 即使 save_ckpt=False，也保存最佳模型
                if args.output_dir:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
            
            print(f'Max accuracy: {max_accuracy:.2f}%')

            if log_writer is not None:
                log_writer.update(test_acc1=test_stats['acc1'], head="perf", step=epoch)
                log_writer.update(test_acc5=test_stats['acc5'], head="perf", step=epoch)
                log_writer.update(test_loss=test_stats['loss'], head="perf", step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

            # EMA评估
            if args.model_ema and args.model_ema_eval:
                test_stats_ema = evaluate(data_loader_val, model_ema.ema, device, args=args)
                print(f"Accuracy of the model EMA on the {len(dataset_val)} test images: {test_stats_ema['acc1']:.1f}%")
                if max_accuracy_ema < test_stats_ema["acc1"]:
                    max_accuracy_ema = test_stats_ema["acc1"]
                    if args.output_dir and args.save_ckpt:
                        utils.save_model(
                            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch="best-ema", model_ema=model_ema)
                print(f'Max EMA accuracy: {max_accuracy_ema:.2f}%')
                log_stats.update({f'test_ema_{k}': v for k, v in test_stats_ema.items()})
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        # 可视化注意力
        if args.visualize_freq > 0 and (epoch + 1) % args.visualize_freq == 0:
            if data_loader_val is not None:
                # 获取一个batch用于可视化
                for batch in data_loader_val:
                    images = batch[0][:8].to(device)
                    break
                save_path = os.path.join(args.output_dir, f'attention_epoch_{epoch+1}.png')
                visualize_attention(model, images, device, args, save_path=save_path)

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if wandb_logger:
            wandb_logger.log_epoch_metrics(log_stats)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Transformer-Seq AdaptiveNN training', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
