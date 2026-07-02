#!/bin/bash
# ============================================================
# 投标串标检测系统 — GPU 服务器一键部署 (CUDA 12.1)
# 用法: bash deploy/setup_server_gpu.sh
#
# 模型:
#   - PaddleOCR 3.x (PP-OCRv6) — 检测+识别, ~100MB
#   - SBERT multilingual-MiniLM-L12-v2 — 语义匹配, ~470MB
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "============================================"
echo "  投标串标检测系统 — GPU 服务器部署"
echo "  CUDA: 12.1"
echo "  OCR:  PaddleOCR 3.x (PP-OCRv6)"
echo "  SBERT: paraphrase-multilingual-MiniLM-L12-v2"
echo "  时间: $(date)"
echo "============================================"

# ---------- 1. 检查 CUDA ----------
echo ""
echo "[1/6] 检查 CUDA 环境..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "错误: 未检测到 NVIDIA GPU，请确认 CUDA 12.1 已安装"
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1
echo "  CUDA 可用"

# ---------- 2. 安装依赖 ----------
echo ""
echo "[2/6] 安装 GPU 依赖..."

PIP_MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"

# PyTorch: 已有CUDA版本则跳过
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
    echo "  PyTorch 已安装 ($TORCH_VER + CUDA), 跳过"
else
    echo "  安装 PyTorch (CUDA 12.1)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -2
fi

# PaddlePaddle GPU: 已有则跳过
if python3 -c "import paddle; assert paddle.is_compiled_with_cuda()" 2>/dev/null; then
    PADDLE_VER=$(python3 -c "import paddle; print(paddle.__version__)")
    echo "  PaddlePaddle 已安装 ($PADDLE_VER + CUDA), 跳过"
else
    echo "  安装 PaddlePaddle GPU (CUDA 12.1)..."
    pip install paddlepaddle-gpu==3.0.0.post120 \
        -f https://www.paddlepaddle.org.cn/whl/linux/cuda12.1/stable.html 2>&1 | tail -2
fi

# 其余依赖 (torch/paddle已存在则自动跳过)
pip install -r deploy/requirements.gpu.txt -i "$PIP_MIRROR" \
    --default-timeout=120 --retries 5 2>&1 | tail -3

echo "  依赖安装完成"

# ---------- 3. 验证 GPU 可用性 ----------
echo ""
echo "[3/6] 验证 GPU 加速..."
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA 不可用!'
print(f'  PyTorch CUDA: OK (GPU: {torch.cuda.get_device_name(0)})')

import paddle
print(f'  PaddlePaddle GPU: {\"OK\" if paddle.is_compiled_with_cuda() else \"FAIL\"}')
print(f'  CUDA 版本: {torch.version.cuda}')
"

# ---------- 4. 下载模型 ----------
echo ""
echo "[4/6] 下载模型文件..."

MODEL_DIR="${MODEL_DIR:-./models}"
mkdir -p "$MODEL_DIR/ocr" "$MODEL_DIR/sbert"

# PaddleOCR 3.x (PP-OCRv6)
echo "  下载 PaddleOCR 3.x 模型 (PP-OCRv6)..."
python3 -c "
import os
os.environ['PADDLEOCR_HOME'] = '$MODEL_DIR/ocr'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
from paddleocr import PaddleOCR
# PP-OCRv6: 完整检测+识别 pipeline
ocr = PaddleOCR(
    lang='ch',
    ocr_version='PP-OCRv6',
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
# 触发模型下载
import numpy as np
try:
    ocr.predict(np.ones((100,200,3), dtype=np.uint8)*255)
except:
    pass
print('  PaddleOCR PP-OCRv6 模型就绪')
" 2>&1 | grep -v "UserWarning\|warnings.warn\|Creating model\|Model files"

# SBERT
echo "  下载 SBERT 模型 (paraphrase-multilingual-MiniLM-L12-v2, ~470MB)..."
python3 -c "
import os
os.environ['HF_HOME'] = '$MODEL_DIR/sbert'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cuda')
print(f'  SBERT 模型就绪 (维度: {model.get_sentence_embedding_dimension()})')
" 2>&1 | tail -2

echo "  模型下载完成"

# ---------- 5. 健康检查 ----------
echo ""
echo "[5/6] 运行健康检查..."
python3 deploy/health_check.py --json 2>&1 | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for k, v in data.get('checks', {}).items():
        status = '+' if v.get('ok') else '!'
        print(f'  [{status}] {k}: {v.get(\"message\",\"\")}')
except:
    print('  无法解析结果')
"

# ---------- 6. 环境配置 ----------
echo ""
echo "[6/6] 生成环境配置..."
cat > deploy/env.gpu.sh << 'EOF'
#!/bin/bash
# GPU 环境变量 — 运行前 source 此文件
export OCR_MODEL_DIR=./models/ocr
export PADDLEOCR_HOME=./models/ocr
export HF_HOME=./models/sbert
export HF_ENDPOINT=https://hf-mirror.com
export OCR_OFFLINE=1
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
export CUDA_VISIBLE_DEVICES=0
EOF

echo ""
echo "============================================"
echo "  GPU 部署完成!"
echo "============================================"
echo ""
echo "模型大小参考:"
echo "  PaddleOCR PP-OCRv6: ~100MB (检测+识别)"
echo "  SBERT MiniLM-L12:   ~470MB (多语言语义模型)"
echo "  总计:               ~570MB"
echo ""
echo "运行:"
echo "  source deploy/env.gpu.sh"
echo "  python main.py --config deploy/config.gpu.json --gpu --input /data/bids --output /data/reports"
