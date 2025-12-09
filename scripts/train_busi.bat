@echo off
REM BUSI U-KAN 训练脚本 (RTX 4060)

cd /d F:\homework\medicalimage\coursedesign\U-KAN\Seg_UKAN

python train.py ^
    --arch UKAN ^
    --dataset BUSI_processed ^
    --data_dir ../datasets ^
    --input_w 256 ^
    --input_h 256 ^
    --batch_size 4 ^
    --epochs 200 ^
    --lr 1e-4 ^
    --name busi_ukan ^
    --output_dir ../outputs ^
    --num_workers 4

pause
