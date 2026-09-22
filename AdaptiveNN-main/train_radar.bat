@echo off
chcp 65001 >nul
REM Radar Signal Recognition Training Script - RTX 2080Ti Optimized

REM Activate conda environment
call conda activate adaptivenn

echo ========================================
echo Radar Signal Recognition - AdaptiveNN Training (Fresh Start)
echo GPU: RTX 2080Ti (11GB)
echo ========================================
echo.
echo Training Configuration:
echo - Classes: 16
echo - Batch Size: 128 (optimized for 2080Ti)
echo - Epochs: 50
echo - Learning Rate: 3e-4 (lowered for stability)
echo - Gradient Clipping: 0.5
echo - Data Loading Workers: 8
echo.
echo Starting training from scratch (no resume)...
echo ========================================
echo.

python -m torch.distributed.launch --use-env --nproc_per_node=1 --master_port=12345 main_ppo.py ^
--data_path "d:/project/测试coatnet (2)/CWDtrain" ^
--eval_data_path "d:/project/测试coatnet (2)/CWDtest" ^
--output_dir ./output_radar_fresh ^
--data_set image_folder ^
--model dynamic_deitS ^
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
--gamma 0.5 ^
--ppo_lam 0.84 ^
--ppo_update_steps 5 ^
--update_policy_freq 10 ^
--num_workers 8 ^
--pin_mem True ^
--auto_resume True ^
--resume "" ^
--weight_decay 0.02

echo.
echo ========================================
echo Training completed!
echo Results saved in: ./output_radar_fresh
echo ========================================
pause