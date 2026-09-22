@echo off
chcp 65001 >nul
REM 可微分AdaptiveNN训练脚本 - 雷达信号识别

call conda activate adaptivenn

echo ========================================
echo 可微分AdaptiveNN训练 - 软注意力版本
echo GPU: RTX 2080Ti (11GB)
echo ========================================
echo.
echo 训练配置:
echo - 模型: dynamic_deitS_diff (可微分版本)
echo - 类别数: 16
echo - Batch Size: 128
echo - Epochs: 50
echo - Learning Rate: 3e-4
echo - 梯度裁剪: 0.5
echo - 数据加载线程: 8
echo.
echo 技术改进:
echo - 用软注意力替代PPO强化学习
echo - 端到端可微分训练
echo - 更稳定的训练过程
echo - 无需PPO超参数调优
echo.
echo 开始训练...
echo ========================================
echo.

python main_differentiable.py ^
--data_path "d:/project/测试coatnet (2)/CWDtrain" ^
--eval_data_path "d:/project/测试coatnet (2)/CWDtest" ^
--output_dir ./output_radar_differentiable ^
--data_set image_folder ^
--model dynamic_deitS_diff ^
--nb_classes 16 ^
--finetune AdaptiveNN_ckpt.pth ^
--lr 3e-4 ^
--batch_size 128 ^
--epochs 50 ^
--clip_grad 0.5 ^
--seq_l 4 ^
--input_size 224 ^
--glance_input_size 112 ^
--focus_patch_size 112 ^
--focus_net_reg_size 160 ^
--loss_reg_focus_net_weight 2.0 ^
--kd_temp 1.0 ^
--kd_alpha 1.0 ^
--attention_entropy_weight 0.0 ^
--num_workers 8 ^
--pin_mem True ^
--auto_resume True ^
--resume "" ^
--weight_decay 0.02 ^
--device cuda ^
--world_size 1 ^
--rank 0 ^
--dist_url ""

echo.
echo ========================================
echo 训练完成！
echo 结果保存在: ./output_radar_differentiable
echo ========================================
pause
