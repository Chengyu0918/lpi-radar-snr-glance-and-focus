@echo off
REM RTX 2080Ti性能优化环境变量设置脚本

echo ========================================
echo RTX 2080Ti 性能优化环境设置
echo ========================================
echo.

REM 禁用CUDA同步，提升性能
set CUDA_LAUNCH_BLOCKING=0

REM 优化显存分配策略
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

REM 启用cuDNN自动调优（首次运行会慢，后续会快）
set CUDNN_BENCHMARK=1

REM 设置数据加载线程数
set OMP_NUM_THREADS=8

echo 环境变量设置完成！
echo.
echo 已设置的优化项：
echo - CUDA_LAUNCH_BLOCKING=0 (异步执行)
echo - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 (显存优化)
echo - CUDNN_BENCHMARK=1 (cuDNN自动调优)
echo - OMP_NUM_THREADS=8 (多线程优化)
echo.
echo ========================================
echo 现在可以运行训练脚本：
echo train_radar.bat
echo ========================================
echo.
