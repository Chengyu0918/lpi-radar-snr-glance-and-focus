@echo off
chcp 65001 >nul
echo ============================================
echo JiT-CWD-Class 模型评估
echo ============================================

cd /d "%~dp0"

REM 激活conda环境
call conda activate adaptivenn

echo.
echo 开始评估带类别条件的去噪模型...
echo.

python evaluate_cwd_class.py ^
    --model JiT-CWD-Class-Base ^
    --checkpoint ./output_cwd_class/checkpoint-last.pth ^
    --noisy_dir ../../../CWDtrain_noisy ^
    --clean_dir ../../../CWDtrain_clean ^
    --output ./evaluation_results_class ^
    --cfg_scale 2.0 ^
    --num_sampling_steps 50 ^
    --num_samples 30 ^
    --num_vis_samples 3

echo.
echo ============================================
echo 评估完成！
echo 结果保存在: evaluation_results_class/
echo ============================================
pause
