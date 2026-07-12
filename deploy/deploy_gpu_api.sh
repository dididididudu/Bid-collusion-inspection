#!/usr/bin/env bash
# ============================================================
# 围标串标检测 API — GPU 服务器部署脚本
#
# 用法:
#   bash deploy/deploy_gpu_api.sh [--port 8001] [--cuda cu121|cu118]
# ============================================================
set -euo pipefail

PORT="8001"
CUDA_TAG="cu121"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="${2:-8001}"
      shift 2
      ;;
    --cuda)
      CUDA_TAG="${2:-cu121}"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "============================================"
echo " 围标串标检测 API — GPU 部署"
echo " 目录: $PROJECT_DIR"
echo " 端口: $PORT"
echo " CUDA: $CUDA_TAG"
echo "============================================"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] 未检测到 nvidia-smi，请先安装 NVIDIA 驱动。"
  exit 1
fi
nvidia-smi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! "$PYTHON_BIN" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"; then
  echo "[ERROR] 需要 Python 3.10+，当前: $($PYTHON_BIN --version)"
  exit 1
fi

echo "[1/5] 创建虚拟环境"
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

echo "[2/5] 安装 PyTorch GPU"
if [[ "$CUDA_TAG" == "cu118" ]]; then
  pip install torch --index-url https://download.pytorch.org/whl/cu118
else
  pip install torch --index-url https://download.pytorch.org/whl/cu121
fi

echo "[3/5] 安装 PaddlePaddle GPU"
if [[ "$CUDA_TAG" == "cu118" ]]; then
  pip install paddlepaddle-gpu -f https://www.paddlepaddle.org.cn/whl/linux/cuda11.8/stable.html
else
  pip install paddlepaddle-gpu -f https://www.paddlepaddle.org.cn/whl/linux/cuda12.1/stable.html
fi

echo "[4/5] 安装项目依赖"
pip install -r deploy/requirements.gpu.txt
pip install -r deploy/requirements.api.txt

echo "[5/5] 验证 GPU 环境"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

from sentence_transformers import SentenceTransformer
m = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2",
    device="cuda" if torch.cuda.is_available() else "cpu",
    cache_folder="./models",
    trust_remote_code=True,
    local_files_only=False,
)
print("sbert dim:", m.get_sentence_embedding_dimension())
PY

cat > .env.gpu <<EOF
USE_GPU=true
SBERT_DEVICE=cuda
SBERT_BATCH_SIZE=256
PHASE1_WORKERS=4
PHASE3_WORKERS=4
PHASE3_USE_PROCESS_POOL=1
OCR_WORKERS=1
COLLUSIVE_ENABLE_OCR=1
COLLUSIVE_ENABLE_IMAGE_ANALYSIS=1
COLLUSIVE_STABLE_WORKDIR=1
COLLUSIVE_KEEP_WORKDIR=1
COLLUSIVE_HOST=0.0.0.0
COLLUSIVE_PORT=$PORT
EOF

echo ""
echo "GPU 环境准备完成。启动方式:"
echo "  source .venv/bin/activate"
echo "  set -a; source .env.gpu; set +a"
echo "  python collusive_check_api.py"
echo ""
echo "重量测试:"
echo "  python scripts/perf_test_collusive_api.py --pdf-dir batch_downloads/75689 --api http://127.0.0.1:$PORT --heavy"
