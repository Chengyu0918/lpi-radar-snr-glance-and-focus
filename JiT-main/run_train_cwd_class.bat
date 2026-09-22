@echo off
chcp 65001 >nul
echo ========================================
echo 带类别条件的CWD去噪模型训练
echo ========================================

REM 激活conda环境
call conda activate adaptivenn

REM 设置数据路径
set NOISY_PATH=../../../CWDtrain_noisy
set CLEAN_PATH=../../../CWDtrain_clean
set OUTPUT_DIR=./output_cwd_class

REM 训练参数
set EPOCHS=100
set BATCH_SIZE=4
set LR=1e-4
set IMG_SIZE=224
set MODEL=JiT-CWD-Class-Base

echo.
echo 数据路径:
echo   含噪数据: %NOISY_PATH%
echo   干净数据: %CLEAN_PATH%
echo   输出目录: %OUTPUT_DIR%
echo.
echo 训练参数:
echo   模型: %MODEL%
echo   图像尺寸: %IMG_SIZE%
echo   批次大小: %BATCH_SIZE%
echo   学习率: %LR%
echo   训练轮数: %EPOCHS%
echo.

REM 开始训练
python train_cwd_class.py ^
    --data_path %NOISY_PATH% ^
    --clean_path %CLEAN_PATH% ^
    --output_dir %OUTPUT_DIR% ^
    --model %MODEL% ^
    --img_size %IMG_SIZE% ^
    --batch_size %BATCH_SIZE% ^
    --lr %LR% ^
    --epochs %EPOCHS% ^
    --num_workers 4 ^
    --save_every 10 ^
    --class_dropout_prob 0.1

echo.
echo ========================================
echo 训练完成！
echo 模型保存在: %OUTPUT_DIR%
echo ========================================
pause
