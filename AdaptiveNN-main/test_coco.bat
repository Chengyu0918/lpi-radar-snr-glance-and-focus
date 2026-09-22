@echo off
chcp 65001 >nul
REM COCO Dataset Test Script - RTX 2080Ti Optimized

echo ========================================
echo COCO Dataset Test - AdaptiveNN
echo GPU: RTX 2080Ti (11GB)
echo ========================================
echo.
echo Test Configuration:
echo - Dataset: COCO (80 classes)
echo - Batch Size: 128
echo - Epochs: 10 (quick test)
echo - Learning Rate: 4e-3
echo.
echo Starting test...
echo ========================================
echo.

python -m torch.distributed.launch --use-env --nproc_per_node=1 --master_port=12345 main_ppo.py ^
    --data_path coco_imagefolder ^
    --nb_classes 80 ^
    --lr 4e-3 ^
    --batch_size 128 ^
    --epochs 10 ^
    --seq_l 4 ^
    --clip_grad 5.0 ^
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
    --output_dir ./output_coco_test

echo.
echo ========================================
echo Test completed!
echo Results saved in: ./output_coco_test
echo ========================================
pause
