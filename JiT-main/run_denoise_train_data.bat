@echo off
chcp 65001 >nul
echo ================================================
echo JiT-CWD 批量去噪 CWDtrain_noisy 数据集
echo ================================================

REM 激活conda环境（如果需要）
REM call conda activate adaptivenn

REM 设置参数
set CHECKPOINT=./output_cwd_paired/checkpoint-last.pth
set INPUT_DIR=../../../CWDtrain_noisy
set OUTPUT_DIR=../../../CWDtrain_denoised
set MODEL=JiT-CWD-Tiny
set IMG_SIZE=224
set BATCH_SIZE=8

REM 检查checkpoint是否存在
if not exist "%CHECKPOINT%" (
    echo 错误: 找不到模型文件 %CHECKPOINT%
    echo 请确保已完成训练或指定正确的checkpoint路径
    pause
    exit /b 1
)

REM 检查输入目录是否存在
if not exist "%INPUT_DIR%" (
    echo 错误: 找不到输入目录 %INPUT_DIR%
    pause
    exit /b 1
)

echo.
echo 推理参数:
echo   模型检查点: %CHECKPOINT%
echo   输入目录: %INPUT_DIR%
echo   输出目录: %OUTPUT_DIR%
echo   模型: %MODEL%
echo   图像尺寸: %IMG_SIZE%
echo.
echo 开始批量去噪...
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
echo ================================================
echo 去噪完成！
echo 去噪后的图像保存在: %OUTPUT_DIR%
echo ================================================
pause
