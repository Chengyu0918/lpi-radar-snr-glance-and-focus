@echo off
chcp 65001 >nul
echo ================================================
echo JiT-CWD 去噪模型训练脚本
echo 针对单卡 RTX 4080 优化
echo ================================================

REM 激活conda环境（如果需要）
REM call conda activate jit

REM 设置参数
set DATA_PATH=../../CWDtrain
set OUTPUT_DIR=./output_cwd
set MODEL=JiT-CWD-Tiny
set IMG_SIZE=224
set BATCH_SIZE=8
set EPOCHS=100

echo.
echo 训练参数:
echo   数据路径: %DATA_PATH%
echo   输出目录: %OUTPUT_DIR%
echo   模型: %MODEL%
echo   图像尺寸: %IMG_SIZE%
echo   批次大小: %BATCH_SIZE%
echo   训练轮数: %EPOCHS%
echo.

REM 开始训练
python train_cwd.py ^
    --model %MODEL% ^
    --data_path %DATA_PATH% ^
    --output_dir %OUTPUT_DIR% ^
    --img_size %IMG_SIZE% ^
    --batch_size %BATCH_SIZE% ^
    --epochs %EPOCHS% ^
    --lr 1e-4 ^
    --warmup_epochs 5 ^
    --train_mode self_supervised ^
    --added_noise_level 0.1 ^
    --sampling_method heun ^
    --num_sampling_steps 50 ^
    --save_freq 10 ^
    --eval_freq 20 ^
    --log_freq 50 ^
    --num_workers 4

echo.
echo 训练完成！
pause
