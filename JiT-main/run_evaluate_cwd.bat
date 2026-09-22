@echo off
chcp 65001 >nul
echo ============================================
echo JiT-CWD 去噪模型评估
echo ============================================

REM 激活conda环境（如果需要）
REM call conda activate jit

REM 设置路径
set CHECKPOINT=./output_cwd_paired/checkpoint-last.pth
set NOISY_DIR=../../../CWDtrain_noisy
set CLEAN_DIR=../../../CWDtrain_clean
set OUTPUT_DIR=./evaluation_results

REM 检查checkpoint是否存在
if not exist "%CHECKPOINT%" (
    echo 错误: 找不到模型文件 %CHECKPOINT%
    echo 请确保已完成训练或指定正确的checkpoint路径
    pause
    exit /b 1
)

REM 检查数据目录是否存在
if not exist "%NOISY_DIR%" (
    echo 错误: 找不到含噪图像目录 %NOISY_DIR%
    pause
    exit /b 1
)

if not exist "%CLEAN_DIR%" (
    echo 错误: 找不到干净图像目录 %CLEAN_DIR%
    pause
    exit /b 1
)

echo.
echo 模型路径: %CHECKPOINT%
echo 含噪图像目录: %NOISY_DIR%
echo 干净图像目录: %CLEAN_DIR%
echo 输出目录: %OUTPUT_DIR%
echo.

REM 运行评估
python evaluate_cwd.py ^
    --checkpoint %CHECKPOINT% ^
    --noisy_dir %NOISY_DIR% ^
    --clean_dir %CLEAN_DIR% ^
    --output %OUTPUT_DIR% ^
    --model JiT-CWD-Tiny ^
    --img_size 224 ^
    --num_samples 50 ^
    --num_vis_samples 3 ^
    --num_sampling_steps 50 ^
    --save_images

echo.
echo ============================================
echo 评估完成！
echo 结果保存在: %OUTPUT_DIR%
echo ============================================
pause
