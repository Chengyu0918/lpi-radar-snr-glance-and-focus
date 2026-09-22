@echo off
chcp 65001 >nul

echo ========================================
echo AdaptiveNN模型 - SNR评估
echo ========================================

call conda activate adaptivenn

python evaluate_adaptivenn_snr.py

echo.
echo ========================================
echo 评估完成！
echo ========================================
pause
