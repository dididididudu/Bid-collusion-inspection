#!/bin/bash
# BatchBidCollusionDetector 系统安装脚本

echo "============================================"
echo "BatchBidCollusionDetector 系统安装"
echo "============================================"
echo ""

echo "[1/3] 检查Python版本..."
python3 --version
if [ $? -ne 0 ]; then
    echo "错误: 未找到Python，请先安装Python 3.7或更高版本"
    exit 1
fi
echo ""

echo "[2/3] 安装依赖包..."
echo "正在安装，请稍候..."
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt
echo ""

echo "[3/3] 验证安装..."
python3 -c "import pdfplumber; print('[OK] pdfplumber')"
python3 -c "import jieba; print('[OK] jieba')"
python3 -c "import sklearn; print('[OK] scikit-learn')"
python3 -c "import numpy; print('[OK] numpy')"
python3 -c "import PIL; print('[OK] Pillow')"
python3 -c "import imagehash; print('[OK] imagehash')"
python3 -c "import networkx; print('[OK] networkx')"
echo ""

echo "============================================"
echo "安装完成！"
echo "============================================"
echo ""
echo "使用示例："
echo "  python3 main.py --input test_data/input --output test_data/output"
echo ""
echo "查看帮助："
echo "  python3 main.py --help"
echo ""
