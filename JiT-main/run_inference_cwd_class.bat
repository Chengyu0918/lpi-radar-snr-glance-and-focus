@echo off
chcp 65001 >nul
echo ========================================
echo 带类别条件的CWD去噪推理
echo ========================================

REM 激活conda环境
call conda activate adaptivenn

REM 设置路径
set CHECKPOINT=./output_cwd_class/best_model.pth
set INPUT_DIR=../../../CWDtrain_noisy
set OUTPUT_DIR=../../../CWDtrain_denoised_class

REM 推理参数
set CFG_SCALE=2.0
set IMG_SIZE=224

echo.
echo 配置:
echo   模型检查点: %CHECKPOINT%
echo   输入目录: %INPUT_DIR%
echo   输出目录: %OUTPUT_DIR%
echo   CFG强度: %CFG_SCALE%
echo   图像尺寸: %IMG_SIZE%
echo.

REM 检查模型文件是否存在
if not exist %CHECKPOINT% (
    echo 错误: 模型文件不存在: %CHECKPOINT%
    echo 请先运行 run_train_cwd_class.bat 训练模型
    pause
    exit /b 1
)

REM 开始推理
python inference_cwd_class.py ^
    --checkpoint %CHECKPOINT% ^
    --input %INPUT_DIR% ^
    --output %OUTPUT_DIR% ^
    --cfg_scale %CFG_SCALE% ^
    --image_size %IMG_SIZE%

echo.
echo ========================================
echo 推理完成！
echo 结果保存在: %OUTPUT_DIR%
echo ========================================
pause
