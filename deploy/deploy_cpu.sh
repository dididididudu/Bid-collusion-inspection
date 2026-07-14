#!/usr/bin/env bash
# ============================================================
# 围标串标检测服务 — CPU 服务器一键部署脚本
# 用法: sudo bash deploy/deploy_cpu.sh [--port 8001]
# ============================================================
set -euo pipefail

PORT="${2:-8001}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "============================================"
echo " 围标串标检测 API — CPU 服务器部署"
echo " 目录: $PROJECT_DIR"
echo " 端口: $PORT"
echo "============================================"

# ── 1. 系统依赖 ──
echo ""
echo "[1/6] 安装系统依赖..."
if command -v apt-get &>/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3.10 python3.10-venv libgl1 libglib2.0-0 gcc
elif command -v yum &>/dev/null; then
    sudo yum install -y -q python3.10 python3.10-devel gcc mesa-libGL glib2
else
    echo "[WARN] 未知包管理器，请手动安装 Python 3.10+ 和 libGL"
fi

# ── 2. 虚拟环境 ──
echo ""
echo "[2/6] 创建虚拟环境..."
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade -q pip setuptools wheel

# ── 3. 安装依赖 ──
echo ""
echo "[3/6] 安装 Python 依赖..."
pip install -r requirements-cpu.txt -q
pip install -r deploy/requirements.api.txt -q
pip install gunicorn -q
echo "  ✅ 依赖安装完成"

# ── 4. 模型验证 ──
echo ""
echo "[4/6] 验证模型..."
echo "  ⏳ 检查 SBERT 模型..."
python -c "
import os
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(
    'paraphrase-multilingual-MiniLM-L12-v2',
    device='cpu',
    cache_folder='./models',
    trust_remote_code=True,
    local_files_only=True,
)
print(f'  ✅ SBERT 就绪 (维度: {model.get_sentence_embedding_dimension()})')
" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "  ⚠️  本地 SBERT 模型不可用，首次运行将自动下载"
fi

echo "  ⏳ 检查 RapidOCR 模型..."
python -c "
from rapidocr_onnxruntime import RapidOCR
RapidOCR()
print('  ✅ RapidOCR 就绪')
" 2>/dev/null || echo "  ⚠️  RapidOCR 首次运行会自动下载"

# ── 5. 健康检查 ──
echo ""
echo "[5/6] 健康检查..."
python deploy/health_check.py 2>&1 | tail -5
echo "  ✅ 健康检查完成"

# ── 6. systemd 服务 ──
echo ""
echo "[6/6] 注册系统服务..."
SERVICE_NAME="bid-collusion-api"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

sudo tee "$SERVICE_FILE" > /dev/null << EOF
[Unit]
Description=围标串标检测 API 服务
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$PROJECT_DIR
Environment="PATH=$PROJECT_DIR/.venv/bin"
Environment="CUDA_VISIBLE_DEVICES="
Environment="BID_HOST=0.0.0.0"
Environment="BID_PORT=$PORT"
Environment="COLLUSIVE_HOST=0.0.0.0"
Environment="COLLUSIVE_PORT=$PORT"
ExecStart=$PROJECT_DIR/.venv/bin/python collusive_check_api.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"
echo "  ✅ 服务已启动"

echo ""
echo "============================================"
echo " 部署完成！"
echo " API: http://0.0.0.0:$PORT"
echo " 文档: http://0.0.0.0:$PORT/docs"
echo " 检查: curl http://localhost:$PORT/api/v1/collusive-check/health"
echo " 日志: sudo journalctl -u $SERVICE_NAME -f"
echo "============================================"
