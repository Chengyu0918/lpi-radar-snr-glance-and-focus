@echo off
REM CWD去噪配对训练脚本
REM 使用配对数据集（含噪图像 + 干净图像）

echo ============================================
echo CWD去噪配对训练
echo ============================================
echo.
echo 数据集信息:
echo   含噪图像: ../../../CWDtrain_noisy (SNR: -20dB 到 0dB)
echo   干净图像: ../../../CWDtrain_clean (SNR: 10dB)
echo   总配对数: 140,800
echo.

REM 激活conda环境（如果需要）
REM call conda activate your_env

REM 配对训练
python train_cwd.py ^
    --model JiT-CWD-Tiny ^
    --img_size 224 ^
    --batch_size 8 ^
    --epochs 100 ^
    --lr 1e-4 ^
    --data_path ../../../CWDtrain_noisy ^
    --clean_path ../../../CWDtrain_clean ^
    --train_mode paired ^
    --output_dir ./output_cwd_paired ^
    --save_freq 10 ^
    --eval_freq 10 ^
    --log_freq 100 ^
    --num_workers 4

echo.
echo 训练完成！
pause
