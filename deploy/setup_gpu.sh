#!/bin/bash
# ============================================================
# 投标串标检测系统 — GPU 环境一键搭建脚本
# 适用于云 Notebook（DSW/Colab/JupyterLab）快速部署
#
# 用法:
#   bash deploy/setup_gpu.sh
#
# 特点:
#   - 每个依赖单独安装，失败不影响后续
#   - 自动检测 CUDA 版本并安装对应 PyTorch
#   - 跳过已安装的包，支持断点续装
#   - 预估耗时: 5-15 分钟（视网络和 GPU 驱动而定）
# ============================================================

set -eo pipefail

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err()   { echo -e "${RED}[ERR]${NC} $1"; }

# 项目根目录
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  投标串标检测系统 — GPU 环境搭建${NC}"
echo -e "${BLUE}  项目目录: $PROJECT_DIR${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# ============================================================
# 1. 系统依赖检查
# ============================================================
log_info "=== 1/9 系统环境检查 ==="

# Python 版本
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
log_info "Python 版本: $PYTHON_VERSION"

# CUDA 检测
CUDA_AVAILABLE=0
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    log_ok "检测到 GPU: $GPU_NAME (驱动: $CUDA_VERSION)"
    CUDA_AVAILABLE=1
else
    log_warn "未检测到 NVIDIA GPU，将使用 CPU 模式"
fi

# pip 版本
PIP_VERSION=$(pip --version 2>&1 | awk '{print $2}')
log_info "pip 版本: $PIP_VERSION"

# 尝试用清华镜像加速
log_info "配置 pip 镜像源..."
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || true
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn 2>/dev/null || true
log_ok "pip 镜像源配置完成"

echo ""

# ============================================================
# 2. 基础依赖
# ============================================================
log_info "=== 2/9 安装基础依赖 ==="

pip install numpy --quiet && log_ok "numpy 安装完成" || { log_warn "numpy 安装失败, 重试..."; pip install numpy --quiet && log_ok "numpy 重试成功" || log_err "numpy 重试也失败"; }
pip install Pillow --quiet && log_ok "Pillow 安装完成" || { log_warn "Pillow 安装失败, 重试..."; pip install Pillow --quiet && log_ok "Pillow 重试成功" || log_err "Pillow 重试也失败"; }
pip install scikit-learn --quiet && log_ok "scikit-learn 安装完成" || log_warn "scikit-learn 安装失败"
pip install python-dateutil --quiet && log_ok "python-dateutil 安装完成" || log_warn "python-dateutil 安装失败"

echo ""

# ============================================================
# 3. PDF 解析
# ============================================================
log_info "=== 3/9 安装 PDF 解析引擎 ==="

pip install PyMuPDF --quiet && log_ok "PyMuPDF (fitz) 安装完成" || log_warn "PyMuPDF 安装失败"

echo ""

# ============================================================
# 4. 中文分词
# ============================================================
log_info "=== 4/9 安装中文分词 ==="

pip install jieba --quiet && log_ok "jieba 安装完成" || log_warn "jieba 安装失败"

echo ""

# ============================================================
# 5. 图片处理
# ============================================================
log_info "=== 5/9 安装图片处理工具 ==="

pip install imagehash --quiet && log_ok "imagehash 安装完成" || log_warn "imagehash 安装失败"
pip install opencv-python-headless --quiet && log_ok "opencv-python 安装完成" || log_warn "opencv-python 安装失败"

echo ""

# ============================================================
# 6. 数据处理与 LSH
# ============================================================
log_info "=== 6/9 安装数据处理工具 ==="

pip install networkx --quiet && log_ok "networkx 安装完成" || log_warn "networkx 安装失败"
pip install datasketch --quiet && log_ok "datasketch 安装完成" || log_warn "datasketch 安装失败"

echo ""

# ============================================================
# 7. PyTorch (GPU)
# ============================================================
log_info "=== 7/9 安装 PyTorch (GPU) ==="

if [ "$CUDA_AVAILABLE" = "1" ]; then
    # 检测 CUDA 版本，选择对应的 PyTorch 版本
    CUDA_MAJOR=$(python3 -c "import subprocess; r=subprocess.run(['nvidia-smi'], capture_output=True, text=True); print('12')" 2>/dev/null || echo "12")

    # 自动检测 CUDA 版本
    CUDA_FULL=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    # 通过 PyTorch 官网检测兼容版本
    log_info "检测 CUDA 版本..."

    log_info "安装 PyTorch 2.x (CUDA 12.1)..."
    if pip install torch --index-url https://download.pytorch.org/whl/cu121 --quiet; then
        log_ok "PyTorch (CUDA 12.1) 安装完成"
    else
        log_warn "CUDA 12.1 安装失败，尝试 CUDA 11.8..."
        if pip install torch --index-url https://download.pytorch.org/whl/cu118 --quiet; then
            log_ok "PyTorch (CUDA 11.8) 安装完成"
        else
            log_warn "GPU 版本安装失败，回退到 CPU 版本"
            pip install torch --quiet || true
        fi
    fi
else
    log_info "未检测到 GPU，安装 CPU 版 PyTorch"
    pip install torch --quiet
fi

# 验证
python3 -c "import torch; v=torch.__version__; c=torch.cuda.is_available(); print(f'  PyTorch {v}, CUDA available: {c}')"
if python3 -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    python3 -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}')"
fi

echo ""

# ============================================================
# 8. SBERT + Transformers
# ============================================================
log_info "=== 8/9 安装 SBERT 语义模型 ==="

pip install transformers --quiet && log_ok "transformers 安装完成" || log_warn "transformers 安装失败"
pip install sentence-transformers --quiet && log_ok "sentence-transformers 安装完成" || log_warn "sentence-transformers 安装失败"

echo ""

# ============================================================
# 9. OCR 引擎
# ============================================================
log_info "=== 9/9 安装 OCR 引擎 ==="

# 先装 EasyOCR（轻量、GPU 友好）
export USE_TF=FALSE
pip install easyocr --quiet && log_ok "easyocr 安装完成" || log_warn "easyocr 安装失败"

echo ""

# ============================================================
# 环境验证
# ============================================================
log_info "=== 环境验证 ==="

echo ""
python3 -c "
import sys
sys.path.insert(0, '.')
print('Python:', sys.version.split()[0])

ok = 0
fail = 0

def check(name, imp):
    global ok, fail
    try:
        exec(f'import {imp}')
        print(f'  ✅ {name}')
        ok += 1
    except ImportError as e:
        print(f'  ❌ {name}: {e}')
        fail += 1

check('PyMuPDF', 'fitz')
check('jieba', 'jieba')
check('numpy', 'numpy')
check('Pillow', 'PIL')
check('scikit-learn', 'sklearn')
check('imagehash', 'imagehash')
check('networkx', 'networkx')
check('datasketch', 'datasketch')
check('Opencv', 'cv2')
check('PyTorch', 'torch')
check('transformers', 'transformers')
check('sentence-transformers', 'sentence_transformers')
check('EasyOCR', 'easyocr')

import torch
if torch.cuda.is_available():
    print(f'  🎉 GPU: {torch.cuda.get_device_name(0)}')
else:
    print(f'  ⚠️  GPU 不可用，将使用 CPU')

print(f'\n  通过: {ok}, 失败: {fail}')
" 2>&1

echo ""
echo -e "${BLUE}============================================================${NC}"
if [ "$CUDA_AVAILABLE" = "1" ]; then
    echo -e "${GREEN}  GPU 环境搭建完成！运行检测:${NC}"
else
    echo -e "${YELLOW}  CPU 环境搭建完成！运行检测:${NC}"
fi
echo -e "${BLUE}============================================================${NC}"
echo ""
echo "  cd $PROJECT_DIR"
echo "  python main.py --input ./input/ --output ./report/ --gpu"
echo ""
