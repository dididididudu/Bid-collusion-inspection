#!/bin/bash
# ============================================================
# 投标串标检测系统 — CPU 服务器一键部署
# 用法: bash deploy/setup_server.sh
#
# 版本: PaddleOCR 2.10 (det=False 仅识别) + PaddlePaddle 2.6
# 模型: PP-OCRv4 识别 (~5MB) + SBERT (~470MB)
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

echo "============================================"
echo "  投标串标检测系统 — CPU 服务器部署"
echo "  时间: $(date)"
echo "============================================"

# ---------- 1. 检查环境 ----------
echo ""
echo "[1/5] 检查 Python 环境..."

PYTHON=""
for cmd in python3.10 python3.11 python3.12 python3; do
    if command -v $cmd &> /dev/null; then
        VER=$($cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        MAJOR=$(echo $VER | cut -d. -f1)
        MINOR=$(echo $VER | cut -d. -f2)
        if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 10 ]; then
            PYTHON=$cmd
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo -e "${RED}错误: 需要 Python 3.10+，请先安装${NC}"
    echo "  Ubuntu: sudo apt install python3.10 python3.10-venv python3-pip"
    echo "  CentOS: sudo yum install python310 python310-pip"
    exit 1
fi
echo -e "  ${GREEN}Python: $($PYTHON --version)${NC}"

PIP="$PYTHON -m pip"
if ! $PYTHON -m pip --version &> /dev/null; then
    echo "  pip 未安装，尝试自动安装..."
    # 方法1: ensurepip 引导
    if $PYTHON -m ensurepip --upgrade 2>/dev/null; then
        echo "  pip 安装成功 (ensurepip)"
    # 方法2: 系统包管理器
    elif command -v apt &> /dev/null; then
        echo "  尝试 apt install python3-pip..."
        sudo apt update -qq && sudo apt install -y -qq python3-pip 2>&1 | tail -1
    elif command -v yum &> /dev/null; then
        echo "  尝试 yum install python3-pip..."
        sudo yum install -y python3-pip 2>&1 | tail -1
    # 方法3: get-pip.py
    else
        echo "  尝试 get-pip.py..."
        curl -sS https://bootstrap.pypa.io/get-pip.py | $PYTHON 2>&1 | tail -1
    fi
    # 再次验证
    if ! $PYTHON -m pip --version &> /dev/null; then
        echo -e "${RED}错误: pip 安装失败，请手动安装${NC}"
        echo "  Ubuntu: sudo apt install python3-pip"
        echo "  通用:  curl https://bootstrap.pypa.io/get-pip.py | sudo python3"
        exit 1
    fi
fi
echo -e "  ${GREEN}pip: $($PYTHON -m pip --version | head -1)${NC}"

# ---------- 2. 安装依赖 ----------
echo ""
echo "[2/5] 安装 Python 依赖..."

MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"

echo "  安装 PaddlePaddle 2.6 (CPU)..."
$PIP install paddlepaddle==2.6.2 \
    -i "$MIRROR" --default-timeout=120 --retries 5 2>&1 | tail -1

echo "  安装其余依赖..."
$PIP install -r deploy/requirements.production.txt \
    -i "$MIRROR" --default-timeout=120 --retries 5 2>&1 | tail -3

echo -e "  ${GREEN}依赖安装完成${NC}"

# ---------- 3. 验证关键模块 ----------
echo ""
echo "[3/5] 验证关键模块..."
$PYTHON -c "
import fitz; print(f'  PyMuPDF: {fitz.version[0]}')
import jieba; print(f'  jieba: OK')
from paddleocr import PaddleOCR; print(f'  PaddleOCR: OK')
from sentence_transformers import SentenceTransformer; print(f'  SBERT: OK')
" 2>&1 | grep -v "UserWarning\|warnings.warn\|pkg_resources"

# ---------- 4. 下载模型 ----------
echo ""
echo "[4/5] 下载模型..."

MODEL_DIR="${MODEL_DIR:-./models}"
mkdir -p "$MODEL_DIR/ocr" "$MODEL_DIR/sbert"

echo "  下载 PaddleOCR 识别模型 (PP-OCRv4_rec, ~5MB)..."
$PYTHON -c "
import os
os.environ['PADDLEOCR_HOME'] = '$MODEL_DIR/ocr'
from paddleocr import PaddleOCR
ocr = PaddleOCR(lang='ch', use_angle_cls=False, show_log=False,
                use_gpu=False, det=False, rec=True)
print('  PaddleOCR 模型就绪')
" 2>&1 | grep "就绪"

echo "  下载 SBERT 语义模型 (~470MB)..."
$PYTHON -c "
import os
os.environ['HF_HOME'] = '$MODEL_DIR/sbert'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')
print(f'  SBERT 模型就绪 (维度: {model.get_sentence_embedding_dimension()})')
" 2>&1 | grep "就绪"

echo -e "  ${GREEN}模型下载完成${NC}"

# ---------- 5. 生成运行脚本 ----------
echo ""
echo "[5/5] 生成运行配置..."

cat > run.sh << 'RUNEOF'
#!/bin/bash
# 投标检测运行脚本
cd "$(dirname "$0")"
export OCR_MODEL_DIR=./models/ocr
export PADDLEOCR_HOME=./models/ocr
export HF_HOME=./models/sbert
export HF_ENDPOINT=https://hf-mirror.com
export OCR_OFFLINE=1

python3 main.py \
    --config deploy/config.production.json \
    --input "${1:-./input}" \
    --output "${2:-./output}" \
    --offline
RUNEOF
chmod +x run.sh

echo ""
echo "============================================"
echo -e "  ${GREEN}部署完成!${NC}"
echo "============================================"
echo ""
echo "运行方式:"
echo "  ./run.sh /data/bids /data/reports"
echo ""
echo "模型空间:"
echo "  OCR:   ~5MB  (PP-OCRv4 识别)"
echo "  SBERT: ~470MB (多语言语义模型)"
echo "  总计:  ~475MB"
