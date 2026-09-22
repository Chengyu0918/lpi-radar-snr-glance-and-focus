@echo off
chcp 65001 >nul
REM ========================================
REM AdaptiveNN系列模型 - 综合训练脚本
REM 包含：PPO版本、Differentiable版本、Transformer版本
REM SNR范围：-20dB 到 10dB
REM ========================================

cd /d "%~dp0"

echo ========================================
echo AdaptiveNN系列模型 - 综合训练
echo ========================================
echo.
echo 将按顺序训练以下模型：
echo   1. AdaptiveNN (PPO) - 基础强化学习版本
echo   2. AdaptiveNN (Differentiable) - 可微分软注意力版本
echo   3. AdaptiveNN (Transformer) - Transformer时序版本
echo.
echo 数据集：
echo   训练集: d:/project/测试coatnet (2)/CWDtrain
echo   测试集: d:/project/测试coatnet (2)/CWDtest
echo   SNR范围: -20dB ~ 10dB
echo.
echo ========================================
echo.

REM 激活conda环境
call conda activate adaptivenn

REM ========================================
REM 1. 训练 AdaptiveNN (PPO) 版本
REM ========================================
echo.
echo ========================================
echo [1/3] 训练 AdaptiveNN (PPO) 版本
echo ========================================
echo.

python -m torch.distributed.launch --use-env --nproc_per_node=1 --master_port=12345 main_ppo.py ^
--data_path "d:/project/测试coatnet (2)/CWDtrain" ^
--eval_data_path "d:/project/测试coatnet (2)/CWDtest" ^
--output_dir ./output_radar_fresh ^
--data_set image_folder ^
--model dynamic_deitS ^
--nb_classes 16 ^
--finetune AdaptiveNN_ckpt.pth ^
--lr 3e-4 ^
--batch_size 128 ^
--epochs 50 ^
--clip_grad 0.5 ^
--seq_l 4 ^
--input_size 224 ^
--glance_input_size 112 ^
--focus_patch_size 112 ^
--focus_net_reg_size 160 ^
--loss_reg_focus_net_weight 2.0 ^
--gamma 0.5 ^
--ppo_lam 0.84 ^
--ppo_update_steps 5 ^
--update_policy_freq 10 ^
--num_workers 8 ^
--pin_mem True ^
--auto_resume True ^
--resume "" ^
--weight_decay 0.02

echo.
echo [1/3] AdaptiveNN (PPO) 训练完成！
echo.

REM ========================================
REM 2. 训练 AdaptiveNN (Differentiable) 版本
REM ========================================
echo.
echo ========================================
echo [2/3] 训练 AdaptiveNN (Differentiable) 版本
echo ========================================
echo.

python main_differentiable.py ^
--data_path "d:/project/测试coatnet (2)/CWDtrain" ^
--eval_data_path "d:/project/测试coatnet (2)/CWDtest" ^
--output_dir ./output_radar_differentiable ^
--data_set image_folder ^
--model dynamic_deitS_diff ^
--nb_classes 16 ^
--finetune AdaptiveNN_ckpt.pth ^
--lr 3e-4 ^
--batch_size 128 ^
--epochs 50 ^
--clip_grad 0.5 ^
--seq_l 4 ^
--input_size 224 ^
--glance_input_size 112 ^
--focus_patch_size 112 ^
--focus_net_reg_size 160 ^
--loss_reg_focus_net_weight 2.0 ^
--kd_temp 1.0 ^
--kd_alpha 1.0 ^
--attention_entropy_weight 0.0 ^
--num_workers 8 ^
--pin_mem True ^
--auto_resume True ^
--resume "" ^
--weight_decay 0.02 ^
--device cuda ^
--world_size 1 ^
--rank 0 ^
--dist_url ""

echo.
echo [2/3] AdaptiveNN (Differentiable) 训练完成！
echo.

REM ========================================
REM 3. 训练 AdaptiveNN (Transformer) 版本
REM ========================================
echo.
echo ========================================
echo [3/3] 训练 AdaptiveNN (Transformer) 版本
echo ========================================
echo.

set CUDA_VISIBLE_DEVICES=0

if not exist "output_transformer_seq" mkdir output_transformer_seq

python main_transformer_seq.py ^
    --data_path "d:/project/测试coatnet (2)/CWDtrain" ^
    --eval_data_path "d:/project/测试coatnet (2)/CWDtest" ^
    --output_dir ./output_transformer_seq ^
    --nb_classes 16 ^
    --batch_size 32 ^
    --epochs 100 ^
    --lr 5e-4 ^
    --min_lr 1e-6 ^
    --warmup_epochs 5 ^
    --weight_decay 0.05 ^
    --input_size 224 ^
    --seq_l 4 ^
    --glance_input_size 112 ^
    --glance_net_depth 6 ^
    --focus_patch_size 96 ^
    --focus_net_depth 6 ^
    --remaining_blocks 2 ^
    --recover_n 3 ^
    --policy_net_hidden_chans 256 ^
    --transformer_num_layers 3 ^
    --transformer_nhead 8 ^
    --transformer_dim_feedforward 1536 ^
    --transformer_dropout 0.1 ^
    --diversity_weight 0.1 ^
    --diversity_sigma 0.3 ^
    --loss_reg_focus_net_weight 0.5 ^
    --kd_alpha 0.5 ^
    --kd_temp 4.0 ^
    --mixup 0.0 ^
    --cutmix 0.0 ^
    --smoothing 0.1 ^
    --drop_path 0.1 ^
    --num_workers 4 ^
    --save_ckpt_freq 5 ^
    --visualize_freq 10 ^
    --use_amp True ^
    --dist_eval False

echo.
echo [3/3] AdaptiveNN (Transformer) 训练完成！
echo.

REM ========================================
REM 4. 运行综合评估
REM ========================================
echo.
echo ========================================
echo [4/4] 运行综合评估并生成图表
echo ========================================
echo.

python train_evaluate_all_extended.py

echo.
echo ========================================
echo 所有训练和评估完成！
echo ========================================
echo.
echo 训练结果保存位置：
echo   - PPO版本: ./output_radar_fresh
echo   - Differentiable版本: ./output_radar_differentiable
echo   - Transformer版本: ./output_transformer_seq
echo.
echo 评估结果保存位置：
echo   - ./output_evaluation_extended
echo.
echo 生成的图表包括：
echo   - 各模型16类信号分别的SNR-准确率曲线
echo   - 各模型所有类别在一张图的曲线
echo   - 各模型SNR-类别准确率热力图
echo   - 各模型混淆矩阵
echo   - 各模型不同SNR下的混淆矩阵
echo   - 三个模型的对比曲线
echo ========================================
pause
