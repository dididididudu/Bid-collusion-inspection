#!/bin/bash
# ============================================================
# 投标文件串标围标检测系统 — GPU 服务器一键部署脚本
# ============================================================
# 用法:
#   chmod +x setup.sh
#   ./setup.sh
# ============================================================
set -e

echo "=============================================="
echo "  Bid Collusion Detection — Server Setup"
echo "=============================================="

# 1. 检测 Python
PYTHON=$(which python3 || which python)
echo "[1/5] Python: $PYTHON ($($PYTHON --version))"

# 2. 创建虚拟环境
if [ ! -d "venv" ]; then
    echo "[2/5] 创建虚拟环境..."
    $PYTHON -m venv venv
else
    echo "[2/5] 虚拟环境已存在"
fi
source venv/bin/activate

# 3. 安装依赖
echo "[3/5] 安装 Python 依赖 (GPU 版)..."
pip install --upgrade pip -q
pip install -r requirements-gpu.txt -q

# 4. 检测 GPU
echo "[4/5] 检测 GPU..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'  CUDA: 可用 ({torch.cuda.device_count()} GPU)')
    print(f'  GPU:  {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
else:
    print('  CUDA: 不可用，将使用 CPU')
"

# 5. 预下载模型（避免首次运行等待）
echo "[5/5] 预下载 SBERT 模型..."
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(
    'paraphrase-multilingual-MiniLM-L12-v2',
    device='cuda' if __import__('torch').cuda.is_available() else 'cpu',
    cache_folder='./models',
)
print('  模型下载完成')
"

echo ""
echo "=============================================="
echo "  部署完成！"
echo "  运行检测: ./run.sh <PDF目录> <输出目录>"
echo "=============================================="
