@echo off
chcp 65001
echo ============================================
echo Transformer时序建模版本 - 迁移学习训练
echo 使用PPO预训练权重进行渐进式训练
echo ============================================

cd /d "%~dp0"

REM 设置Python环境
set PYTHONIOENCODING=utf-8

REM 检查预训练文件是否存在
if not exist "output_radar\checkpoint-best.pth" (
    echo 错误: 预训练文件 output_radar\checkpoint-best.pth 不存在!
    echo 请先运行PPO版本训练，或指定正确的预训练文件路径
    pause
    exit /b 1
)

echo.
echo 训练策略:
echo   第1阶段 (0-19 epoch): 冻结glance_net和focus_net，只训练Transformer
echo   第2阶段 (20+ epoch): 解冻所有模块，联合微调（预训练模块使用0.1x学习率）
echo.

REM 运行训练
REM SNR范围：-20dB 到 0dB，步长2dB
python main_transformer_seq.py ^
    --data_path "d:/project/测试coatnet (2)/CWDtrain" ^
    --eval_data_path "d:/project/测试coatnet (2)/CWDtest" ^
    --data_set image_folder ^
    --nb_classes 16 ^
    --output_dir ./output_transformer_transfer ^
    --batch_size 32 ^
    --epochs 100 ^
    --lr 5e-4 ^
    --min_lr 1e-6 ^
    --warmup_epochs 5 ^
    --weight_decay 0.05 ^
    --seq_l 4 ^
    --transformer_num_layers 3 ^
    --transformer_nhead 8 ^
    --transformer_dim_feedforward 1536 ^
    --transformer_dropout 0.1 ^
    --diversity_weight 0.1 ^
    --diversity_sigma 0.3 ^
    --mixup 0.0 ^
    --cutmix 0.0 ^
    --smoothing 0.1 ^
    --drop_path 0.1 ^
    --pretrained_ppo ./output_radar/checkpoint-best.pth ^
    --freeze_pretrained_epochs 20 ^
    --pretrained_lr_scale 0.1 ^
    --transformer_lr_scale 1.0 ^
    --save_ckpt_freq 5 ^
    --visualize_freq 10 ^
    --num_workers 4 ^
    --use_amp True

echo.
echo ============================================
echo 训练完成!
echo 模型保存在: output_transformer_transfer
echo ============================================
pause
