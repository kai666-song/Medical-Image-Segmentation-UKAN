@echo off
echo ========================================
echo Training UKAN_CBAM on BUSI dataset
echo ========================================

cd /d %~dp0\..\Seg_UKAN

python train.py ^
    --arch UKAN_CBAM ^
    --dataset BUSI_processed ^
    --data_dir ../datasets ^
    --input_w 256 --input_h 256 ^
    --batch_size 4 ^
    --epochs 200 ^
    --lr 1e-4 ^
    --name busi_ukan_cbam ^
    --output_dir ../outputs ^
    --num_workers 4

echo ========================================
echo Training completed!
echo ========================================
pause
