@echo off
REM Transformer时序建模版本评估脚本

cd /d "%~dp0"

REM 设置CUDA设备
set CUDA_VISIBLE_DEVICES=0

REM 设置数据路径
set DATA_PATH=../../CWDtest
set CHECKPOINT=./output_transformer_seq/checkpoint-best.pth
set OUTPUT_DIR=./eval_results_transformer_seq

echo ========================================
echo Transformer-Seq AdaptiveNN Evaluation
echo ========================================
echo Data path: %DATA_PATH%
echo Checkpoint: %CHECKPOINT%
echo Output: %OUTPUT_DIR%
echo ========================================

python evaluate_transformer_seq.py ^
    --data_path %DATA_PATH% ^
    --checkpoint %CHECKPOINT% ^
    --output_dir %OUTPUT_DIR% ^
    --nb_classes 16 ^
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
    --diversity_sigma 0.3 ^
    --visualize ^
    --num_visualize 16

echo ========================================
echo Evaluation completed!
echo ========================================
pause
