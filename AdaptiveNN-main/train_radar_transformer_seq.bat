@echo off
chcp 65001 >nul
REM Transformer时序建模版本训练脚本
REM 支持轨迹多样性约束L_div
REM SNR范围：-20dB 到 0dB，步长2dB

cd /d "%~dp0"

REM 设置CUDA设备
set CUDA_VISIBLE_DEVICES=0

REM 设置数据路径（使用绝对路径）
set DATA_PATH=d:/project/测试coatnet (2)/CWDtrain
set EVAL_DATA_PATH=d:/project/测试coatnet (2)/CWDtest

REM 创建输出目录
if not exist "output_transformer_seq" mkdir output_transformer_seq

echo ========================================
echo Transformer-Seq AdaptiveNN Training
echo ========================================
echo Data path: %DATA_PATH%
echo Eval path: %EVAL_DATA_PATH%
echo Output: output_transformer_seq
echo ========================================

python main_transformer_seq.py ^
    --data_path %DATA_PATH% ^
    --eval_data_path %EVAL_DATA_PATH% ^
    --data_set image_folder ^
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

echo ========================================
echo Training completed!
echo ========================================
pause
