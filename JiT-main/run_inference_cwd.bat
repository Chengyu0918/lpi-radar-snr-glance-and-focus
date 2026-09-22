@echo off
chcp 65001 >nul
echo ================================================
echo JiT-CWD 去噪模型推理脚本
echo ================================================

REM 激活conda环境（如果需要）
REM call conda activate jit

REM 设置参数
set CHECKPOINT=./output_cwd/checkpoint-final.pth
set INPUT_DIR=../../CWDtest
set OUTPUT_DIR=./denoised_output
set MODEL=JiT-CWD-Tiny
set IMG_SIZE=224
set BATCH_SIZE=8

echo.
echo 推理参数:
echo   模型检查点: %CHECKPOINT%
echo   输入目录: %INPUT_DIR%
echo   输出目录: %OUTPUT_DIR%
echo   模型: %MODEL%
echo   图像尺寸: %IMG_SIZE%
echo.

REM 开始推理
python inference_cwd.py ^
    --checkpoint %CHECKPOINT% ^
    --input %INPUT_DIR% ^
    --output %OUTPUT_DIR% ^
    --model %MODEL% ^
    --img_size %IMG_SIZE% ^
    --batch_size %BATCH_SIZE% ^
    --sampling_method heun ^
    --num_sampling_steps 50 ^
    --use_ema

echo.
echo 推理完成！去噪结果保存在: %OUTPUT_DIR%
pause
