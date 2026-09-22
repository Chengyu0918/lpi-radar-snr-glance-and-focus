@echo off
chcp 65001 >nul
REM 可微分AdaptiveNN评估脚本

call conda activate adaptivenn

echo ========================================
echo 可微分AdaptiveNN模型评估
echo ========================================
echo.
echo 评估配置:
echo - 模型: dynamic_deitS_diff (可微分版本)
echo - Checkpoint: output_radar_differentiable/checkpoint-best.pth
echo - 测试集: CWDtest
echo - 类别数: 16
echo - 按SNR级别评估: 是
echo.
echo 开始评估...
echo ========================================
echo.

python evaluate_differentiable.py ^
--eval_data_path "d:/project/测试coatnet (2)/CWDtest" ^
--checkpoint ./output_radar_differentiable/checkpoint-best.pth ^
--model dynamic_deitS_diff ^
--nb_classes 16 ^
--data_set image_folder ^
--batch_size 128 ^
--num_workers 8 ^
--seq_l 4 ^
--input_size 224 ^
--glance_input_size 112 ^
--focus_patch_size 112 ^
--focus_net_reg_size 160 ^
--evaluate_by_snr True

echo.
echo ========================================
echo 评估完成！
echo ========================================
pause
