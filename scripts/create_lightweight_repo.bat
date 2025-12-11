@echo off
chcp 65001 >nul
echo ========================================
echo 创建轻量化仓库脚本
echo ========================================

set SOURCE_DIR=%~dp0..
set TARGET_DIR=%~dp0..\..\U-KAN-Lightweight

echo 源目录: %SOURCE_DIR%
echo 目标目录: %TARGET_DIR%
echo.

:: 创建目标目录
if exist "%TARGET_DIR%" (
    echo 目标目录已存在，正在删除...
    rmdir /s /q "%TARGET_DIR%"
)
mkdir "%TARGET_DIR%"

echo [1/8] 复制 Seg_UKAN 代码...
xcopy "%SOURCE_DIR%\Seg_UKAN\*.py" "%TARGET_DIR%\Seg_UKAN\" /E /I /Y >nul
xcopy "%SOURCE_DIR%\Seg_UKAN\*.txt" "%TARGET_DIR%\Seg_UKAN\" /E /I /Y >nul
xcopy "%SOURCE_DIR%\Seg_UKAN\*.yml" "%TARGET_DIR%\Seg_UKAN\" /E /I /Y >nul
xcopy "%SOURCE_DIR%\Seg_UKAN\*.sh" "%TARGET_DIR%\Seg_UKAN\" /E /I /Y >nul
copy "%SOURCE_DIR%\Seg_UKAN\LICENSE" "%TARGET_DIR%\Seg_UKAN\" >nul 2>&1

echo [2/8] 复制 Diffusion_UKAN 代码...
mkdir "%TARGET_DIR%\Diffusion_UKAN"
xcopy "%SOURCE_DIR%\Diffusion_UKAN\*.py" "%TARGET_DIR%\Diffusion_UKAN\" /Y >nul
xcopy "%SOURCE_DIR%\Diffusion_UKAN\*.txt" "%TARGET_DIR%\Diffusion_UKAN\" /Y >nul
xcopy "%SOURCE_DIR%\Diffusion_UKAN\*.md" "%TARGET_DIR%\Diffusion_UKAN\" /Y >nul
xcopy "%SOURCE_DIR%\Diffusion_UKAN\Diffusion\*.py" "%TARGET_DIR%\Diffusion_UKAN\Diffusion\" /E /I /Y >nul 2>&1
xcopy "%SOURCE_DIR%\Diffusion_UKAN\tools\*.py" "%TARGET_DIR%\Diffusion_UKAN\tools\" /E /I /Y >nul 2>&1

echo [3/8] 复制 scripts 脚本...
xcopy "%SOURCE_DIR%\scripts\*.py" "%TARGET_DIR%\scripts\" /E /I /Y >nul
xcopy "%SOURCE_DIR%\scripts\*.bat" "%TARGET_DIR%\scripts\" /E /I /Y >nul

echo [4/8] 复制 docs 文档和图表...
xcopy "%SOURCE_DIR%\docs\*.*" "%TARGET_DIR%\docs\" /E /I /Y >nul

echo [5/8] 复制 assets 资源...
xcopy "%SOURCE_DIR%\assets\*.*" "%TARGET_DIR%\assets\" /E /I /Y >nul

echo [6/8] 复制训练日志 (不含模型权重)...
mkdir "%TARGET_DIR%\outputs\busi_ukan"
mkdir "%TARGET_DIR%\outputs\busi_ukan_cbam"
copy "%SOURCE_DIR%\outputs\busi_ukan\log.csv" "%TARGET_DIR%\outputs\busi_ukan\" >nul
copy "%SOURCE_DIR%\outputs\busi_ukan\config.yml" "%TARGET_DIR%\outputs\busi_ukan\" >nul
copy "%SOURCE_DIR%\outputs\busi_ukan_cbam\log.csv" "%TARGET_DIR%\outputs\busi_ukan_cbam\" >nul
copy "%SOURCE_DIR%\outputs\busi_ukan_cbam\config.yml" "%TARGET_DIR%\outputs\busi_ukan_cbam\" >nul

echo [7/8] 复制根目录文件...
copy "%SOURCE_DIR%\README.md" "%TARGET_DIR%\" >nul
copy "%SOURCE_DIR%\.gitignore" "%TARGET_DIR%\" >nul

echo [8/8] 初始化 Git 仓库...
cd /d "%TARGET_DIR%"
git init
git add .
git commit -m "Initial commit: U-KAN reproduction and CBAM improvement"

echo.
echo ========================================
echo 轻量化仓库创建完成！
echo 位置: %TARGET_DIR%
echo ========================================
echo.
echo 下一步操作:
echo 1. cd "%TARGET_DIR%"
echo 2. git remote add origin https://github.com/你的用户名/新仓库名.git
echo 3. git push -u origin master
echo.
pause
