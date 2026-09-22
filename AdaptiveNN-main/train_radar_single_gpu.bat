@echo off
chcp 65001 >nul
echo ========================================
echo Radar Signal Recognition Training
echo Single GPU Mode (RTX 2080Ti)
echo ========================================
echo.

REM Activate conda environment
call conda activate adaptivenn
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate conda environment 'adaptivenn'
    echo Please run setup_env.bat first
    pause
    exit /b 1
)

echo [OK] Conda environment activated
echo.

REM Check if data exists
set "TRAIN_DATA=..\..\CWDtrain"
set "TEST_DATA=..\..\CWDtest"

if not exist "%TRAIN_DATA%" (
    echo [ERROR] Training data not found: %TRAIN_DATA%
    echo Please run MATLAB script to generate data first
    pause
    exit /b 1
)

if not exist "%TEST_DATA%" (
    echo [ERROR] Test data not found: %TEST_DATA%
    echo Please run MATLAB script to generate data first
    pause
    exit /b 1
)

echo [OK] Data folders found
echo.

echo ========================================
echo Training Configuration
echo ========================================
echo Model: AdaptiveNN (dynamic_deitS)
echo Dataset: Radar Signals (16 classes)
echo Training data: %TRAIN_DATA%
echo Test data: %TEST_DATA%
echo.
echo GPU: RTX 2080Ti (11GB)
echo Batch Size: 128
echo Epochs: 300
echo Learning Rate: 4e-3
echo Workers: 8
echo.
echo Output: ./output_radar
echo ========================================
echo.

REM Run training (single GPU, no distributed)
python main_ppo.py ^
    --data_path %TRAIN_DATA% ^
    --eval_data_path %TEST_DATA% ^
    --data_set image_folder ^
    --nb_classes 16 ^
    --model dynamic_deitS ^
    --batch_size 128 ^
    --epochs 300 ^
    --lr 4e-3 ^
    --num_workers 8 ^
    --pin_mem True ^
    --input_size 224 ^
    --output_dir ./output_radar ^
    --log_dir ./output_radar/logs ^
    --save_ckpt True ^
    --save_ckpt_freq 10 ^
    --dist_eval False ^
    --world_size 1 ^
    --device cuda:0

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo [SUCCESS] Training completed!
    echo ========================================
    echo.
    echo Results saved in: ./output_radar
    echo Logs saved in: ./output_radar/logs
    echo.
    echo Check the best model: ./output_radar/checkpoint-best.pth
) else (
    echo.
    echo ========================================
    echo [ERROR] Training failed!
    echo ========================================
    echo Please check the error messages above
)

echo.
pause
