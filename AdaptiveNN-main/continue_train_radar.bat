@echo off
chcp 65001 >nul
REM AdaptiveNN继续训练脚本 - 从best checkpoint恢复

call conda activate adaptivenn

echo ========================================
echo AdaptiveNN继续训练 (从Epoch 50开始)
echo GPU: RTX 2080Ti (11GB)
echo ========================================
echo.
echo 训练配置:
echo - 起始Checkpoint: output_radar_fresh/checkpoint-best.pth
echo - 额外训练: 50 epochs (总共50 epochs)
echo - Batch Size: 128
echo - Learning Rate: 1e-4 (降低以微调)
echo - 输出目录: output_radar_continue
echo.
echo 开始继续训练...
echo ========================================
echo.

python -m torch.distributed.launch --use-env --nproc_per_node=1 --master_port=12346 main_ppo.py ^
--data_path "d:/project/测试coatnet (2)/CWDtrain" ^
--eval_data_path "d:/project/测试coatnet (2)/CWDtest" ^
--output_dir ./output_radar_continue ^
--data_set image_folder ^
--model dynamic_deitS ^
--nb_classes 16 ^
--resume output_radar_fresh/checkpoint-best.pth ^
--lr 1e-4 ^
--batch_size 128 ^
--epochs 50 ^
--clip_grad 0.5 ^
--seq_l 4 ^
--input_size 224 ^
--glance_input_size 112 ^
--focus_patch_size 112 ^
--focus_net_reg_size 160 ^
--loss_reg_focus_net_weight 2.0 ^
--gamma 0.5 ^
--ppo_lam 0.84 ^
--ppo_update_steps 5 ^
--update_policy_freq 10 ^
--num_workers 8 ^
--pin_mem True ^
--auto_resume True ^
--weight_decay 0.02

echo.
echo ========================================
echo 继续训练完成！
echo 结果保存在: ./output_radar_continue
echo ========================================
pause
