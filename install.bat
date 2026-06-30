@echo off
chcp 65001 >nul
echo ============================================
echo BatchBidCollusionDetector 系统安装
echo ============================================
echo.

echo [1/3] 检查Python版本...
python --version
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python 3.7或更高版本
    pause
    exit /b 1
)
echo.

echo [2/3] 安装依赖包...
echo 正在安装，请稍候...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo 错误: 依赖包安装失败
    echo 尝试使用国内镜像源...
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
)
echo.

echo [3/3] 验证安装...
python -c "import pdfplumber; print('[OK] pdfplumber')"
python -c "import jieba; print('[OK] jieba')"
python -c "import sklearn; print('[OK] scikit-learn')"
python -c "import numpy; print('[OK] numpy')"
python -c "import PIL; print('[OK] Pillow')"
python -c "import imagehash; print('[OK] imagehash')"
python -c "import networkx; print('[OK] networkx')"
echo.

echo ============================================
echo 安装完成！
echo ============================================
echo.
echo 使用示例：
echo   python main.py --input test_data\input --output test_data\output
echo.
echo 查看帮助：
echo   python main.py --help
echo.

pause
