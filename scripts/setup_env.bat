@echo off
REM BUSI U-KAN 环境配置脚本 (RTX 4060 / CUDA 12.x)

echo ========================================
echo 配置 medicalimage conda 环境
echo ========================================

REM 激活 conda 环境
call conda activate medicalimage

REM 安装 PyTorch (CUDA 12.1 - 适配 RTX 4060)
echo [1/3] 安装 PyTorch (CUDA 12.1)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

REM 安装核心依赖
echo [2/3] 安装核心依赖...
pip install numpy pillow opencv-python scikit-image scipy
pip install albumentations tqdm tensorboardX pyyaml pandas
pip install timm addict yapf

REM 安装数据处理脚本依赖 (已包含在上面)
echo [3/3] 验证安装...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ========================================
echo 环境配置完成！
echo ========================================
pause
