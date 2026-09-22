@echo off
chcp 65001 >nul
echo ========================================
echo COCO Dataset Test - AdaptiveNN
echo Single GPU Mode (RTX 2080Ti)
echo ========================================
echo.

REM Activate conda environment
call conda activate adaptivenn
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate conda environment 'adaptivenn'
    pause
    exit /b 1
)

echo [OK] Conda environment activated
echo.

REM Check if COCO data exists
if not exist "coco_imagefolder\train" (
    echo [ERROR] COCO training data not found
    echo Please run download_coco.py and convert_coco_to_imagefolder.py first
    pause
    exit /b 1
)

if not exist "coco_imagefolder\val" (
    echo [ERROR] COCO validation data not found
    echo Please run download_coco.py and convert_coco_to_imagefolder.py first
    pause
    exit /b 1
)

echo [OK] COCO data found
echo.

echo ========================================
echo Test Configuration
echo ========================================
echo Dataset: COCO (80 classes)
echo Batch Size: 128
echo Epochs: 10 (quick test)
echo Learning Rate: 4e-3
echo.
echo Starting test...
echo ========================================
echo.

REM Run training (single GPU, no distributed)
python main_ppo.py ^
    --data_path ./coco_imagefolder/train ^
    --eval_data_path ./coco_imagefolder/val ^
    --data_set image_folder ^
    --nb_classes 80 ^
    --model dynamic_deitS ^
    --batch_size 128 ^
    --epochs 10 ^
    --lr 4e-3 ^
    --num_workers 8 ^
    --pin_mem True ^
    --input_size 224 ^
    --output_dir ./output_coco_test ^
    --log_dir ./output_coco_test/logs ^
    --save_ckpt True ^
    --dist_eval False ^
    --world_size 1 ^
    --device cuda:0

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo [SUCCESS] Test completed!
    echo ========================================
    echo Results saved in: ./output_coco_test
) else (
    echo.
    echo ========================================
    echo [ERROR] Test failed!
    echo ========================================
)

echo.
pause
