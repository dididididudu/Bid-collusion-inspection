#!/bin/bash
# ============================================================
# 投标文件串标围标检测系统 — Notebook 环境一键安装脚本
# ============================================================
#
# 适用环境:
#   ubuntu22.04-cuda12.1.0-py311-torch2.3.1-tf2.16.1
#
# 用法:
#   chmod +x install_notebook.sh
#   ./install_notebook.sh
#
# 国内镜像加速（可选）:
#   PIP_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple HF_MIRROR=https://hf-mirror.com ./install_notebook.sh
# ============================================================

set -e

# ---- 镜像源配置 ----
PIP_MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
HF_MIRROR="${HF_MIRROR:-https://hf-mirror.com}"

# ---- 颜色输出 ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()   { echo -e "${RED}[ERROR]${NC} $1"; }

# ---- 进度追踪 ----
STEP=0
TOTAL=6

next_step() {
    STEP=$((STEP + 1))
    echo ""
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}[${STEP}/${TOTAL}] $1${NC}"
    echo -e "${BLUE}============================================${NC}"
}

# ============================================================
echo -e "${GREEN}"
echo "=============================================="
echo "  Bid Collusion Detection — Notebook Setup"
echo "=============================================="
echo -e "${NC}"
echo "  环境: ubuntu22.04-cuda12.1.0-py311-torch2.3.1"
echo "  Pip 镜像: ${PIP_MIRROR}"
echo "  HF  镜像: ${HF_MIRROR}"
echo ""

# ---- [1/TOTAL] 环境检测 ----
next_step "环境检测"

# Python 版本
PYTHON=$(which python3 || which python)
if [ -z "$PYTHON" ]; then
    err "未找到 Python！"
    exit 1
fi
PY_VER=$($PYTHON --version 2>&1)
info "Python: ${PY_VER} (${PYTHON})"

# 检测 OS
OS_NAME=$(uname -s)
ARCH=$(uname -m)
info "系统: ${OS_NAME} ${ARCH}"

# 检测 CUDA
info "检测 CUDA..."
CUDA_AVAILABLE=0
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    info "NVIDIA 驱动: ${CUDA_VERSION}"
    info "GPU: ${GPU_NAME}"
    CUDA_AVAILABLE=1
else
    warn "未检测到 NVIDIA 驱动，将使用 CPU 模式"
fi

# 检测 PyTorch 和 CUDA 可用性
PYTORCH_CUDA=$($PYTHON -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
" 2>&1) || true
echo "$PYTORCH_CUDA"

if echo "$PYTORCH_CUDA" | grep -q "CUDA available: True"; then
    ok "PyTorch CUDA 可用！将启用 GPU 加速"
else
    warn "PyTorch CUDA 不可用，将使用 CPU"
fi

# ---- [2/TOTAL] 安装系统依赖 ----
next_step "安装系统依赖"

info "安装系统工具 (libGL, libgomp)..."
apt-get update -qq && apt-get install -y -qq \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    2>&1 | tail -5 || warn "部分系统包安装失败（非致命）"
ok "系统依赖安装完成"

# ---- [3/TOTAL] 安装 Python 依赖 ----
next_step "安装 Python 依赖"

PIP_ARGS=""
if [ "$PIP_MIRROR" != "https://pypi.tuna.tsinghua.edu.cn/simple" ]; then
    PIP_ARGS="-i ${PIP_MIRROR}"
fi

info "升级 pip..."
$PYTHON -m pip install --upgrade pip -q ${PIP_ARGS} 2>&1 | tail -3

info "安装核心依赖..."
$PYTHON -m pip install -r requirements-notebook.txt \
    ${PIP_ARGS} \
    --default-timeout=120 \
    --retries 3 \
    -q 2>&1 | tail -5

if [ $? -ne 0 ]; then
    err "依赖安装失败！请检查网络连接或手动执行:"
    err "  pip install -r requirements-notebook.txt -i ${PIP_MIRROR}"
    exit 1
fi
ok "核心依赖安装完成"

# ---- [4/TOTAL] 安装 OCR 引擎 ----
next_step "安装 OCR 引擎（支持中文文字提取）"

# 尝试安装 PaddlePaddle + PaddleOCR（CUDA 12.x 兼容版本）
info "尝试安装 PaddleOCR（GPU 版）..."
PADDLE_OK=0

# 检测 CUDA 版本以选择合适的 PaddlePaddle
CUDA_VER=$($PYTHON -c "
import torch
print('.'.join(torch.version.cuda.split('.')[:2]) if torch.cuda.is_available() and torch.version.cuda else '')
" 2>/dev/null)

if [ -n "$CUDA_VER" ]; then
    info "CUDA 版本: ${CUDA_VER}，安装匹配的 PaddlePaddle..."
    
    # PaddlePaddle 3.0.0rc0 支持 CUDA 12.x
    $PYTHON -m pip install paddlepaddle-gpu==3.0.0rc0 \
        ${PIP_ARGS} \
        --default-timeout=120 \
        --retries 3 \
        -q 2>&1 | tail -5
    
    if [ $? -eq 0 ]; then
        info "PaddlePaddle 安装成功，继续安装 PaddleOCR..."
        $PYTHON -m pip install paddleocr>=2.8.0 \
            ${PIP_ARGS} \
            --default-timeout=120 \
            --retries 3 \
            -q 2>&1 | tail -5
        
        if [ $? -eq 0 ]; then
            PADDLE_OK=1
            ok "PaddleOCR 安装成功！将使用 PaddleOCR GPU 引擎"
        fi
    fi
fi

# 如果 PaddleOCR 安装失败，降级到 EasyOCR
if [ $PADDLE_OK -eq 0 ]; then
    warn "PaddleOCR 安装失败，降级到 EasyOCR..."
    warn "（EasyOCR 基于 PyTorch，无需额外深度学习框架）"
    
    $PYTHON -m pip install easyocr>=1.7.0 \
        ${PIP_ARGS} \
        --default-timeout=120 \
        --retries 3 \
        -q 2>&1 | tail -5
    
    if [ $? -eq 0 ]; then
        ok "EasyOCR 安装成功！将使用 EasyOCR 引擎"
        
        # 更新配置文件使用 EasyOCR
        if [ -f "config.notebook.json" ]; then
            $PYTHON -c "
import json
with open('config.notebook.json', 'r') as f:
    cfg = json.load(f)
cfg['OCR_ENGINE'] = 'easyocr'
with open('config.notebook.json', 'w') as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)
print('config.notebook.json 已更新为 EasyOCR')
"
        fi
    else
        warn "EasyOCR 也安装失败，OCR 功能将不可用"
        warn "可稍后手动安装: pip install easyocr -i ${PIP_MIRROR}"
    fi
fi

# ---- [5/TOTAL] 预下载 SBERT 模型 ----
next_step "预下载 SBERT 模型（用于语义相似度匹配）"

info "设置 HF 镜像: ${HF_MIRROR}"
export HF_ENDPOINT="${HF_MIRROR}"
export HF_HUB_OFFLINE=0

SBERT_MODEL="paraphrase-multilingual-MiniLM-L12-v2"
$PYTHON -c "
import os
os.environ['HF_ENDPOINT'] = '${HF_MIRROR}'
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['HF_HUB_OFFLINE'] = '0'

from sentence_transformers import SentenceTransformer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'下载 SBERT 模型 (${SBERT_MODEL}) 到 ./models/，设备: {device}...')
model = SentenceTransformer(
    '${SBERT_MODEL}',
    device=device,
    cache_folder='./models',
)
print(f'模型加载完成，嵌入维度: {model.get_sentence_embedding_dimension()}')
" 2>&1

if [ $? -eq 0 ]; then
    ok "SBERT 模型下载完成"
else
    warn "SBERT 模型下载失败（首次运行时会自动下载）"
    warn "可手动执行:"
    warn "  HF_ENDPOINT=${HF_MIRROR} python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('${SBERT_MODEL}', cache_folder='./models')\""
fi

# ---- [6/TOTAL] 验证安装 ----
next_step "验证安装"

# 创建临时验证脚本
$PYTHON -c "
import sys
errors = []

# 核心模块
modules = [
    ('PyMuPDF', 'fitz'),
    ('jieba', 'jieba'),
    ('sklearn', 'sklearn'),
    ('numpy', 'numpy'),
    ('PIL', 'PIL'),
    ('imagehash', 'imagehash'),
    ('datasketch', 'datasketch'),
    ('networkx', 'networkx'),
    ('sentence_transformers', 'sentence_transformers'),
    ('transformers', 'transformers'),
]

for name, mod in modules:
    try:
        __import__(mod)
        print(f'  [OK] {name}')
    except ImportError:
        errors.append(name)
        print(f'  [FAIL] {name}')

# OCR 引擎
try:
    import paddleocr
    print(f'  [OK] PaddleOCR ({paddleocr.__version__})')
except ImportError:
    try:
        import easyocr
        print(f'  [OK] EasyOCR')
    except ImportError:
        print(f'  [WARN] OCR 引擎未安装（功能受限）')

# CUDA
import torch
if torch.cuda.is_available():
    print(f'  [OK] CUDA {torch.version.cuda} - {torch.cuda.get_device_name(0)}')
else:
    print(f'  [WARN] CUDA 不可用，将使用 CPU')

if errors:
    print(f'\n以下模块安装失败: {errors}')
    sys.exit(1)
else:
    print('\n✅ 所有核心模块安装成功！')
" 2>&1

echo ""
echo -e "${GREEN}==============================================${NC}"
echo -e "${GREEN}  安装完成！${NC}"
echo -e "${GREEN}==============================================${NC}"
echo ""
echo "运行检测:"
echo ""
echo "  # 方式一：使用 GPU 加速（推荐）"
echo "  python main.py --input ./bids/ --output ./report/ --gpu --config config.notebook.json"
echo ""
echo "  # 方式二：使用 CPU"
echo "  python main.py --input ./bids/ --output ./report/"
echo ""
echo "  # 仅运行诊断（验证环境）"
echo "  python main.py --diagnose --gpu --config config.notebook.json"
echo ""
echo "配置参考:"
echo "  config.notebook.json — Notebook GPU 优化配置"
echo ""