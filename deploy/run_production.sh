#!/bin/bash
# ============================================================
# 生产运行脚本
# 用法: bash deploy/run_production.sh [输入目录] [输出目录]
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# 默认路径（可通过环境变量覆盖）
INPUT_DIR="${1:-${BID_INPUT_DIR:-/data/bids}}"
OUTPUT_DIR="${2:-${BID_OUTPUT_DIR:-/data/reports}}"

echo "============================================"
echo "  投标串标检测系统 — 生产运行"
echo "  时间: $(date)"
echo "  输入: $INPUT_DIR"
echo "  输出: $OUTPUT_DIR"
echo "============================================"

# 检查输入目录
if [ ! -d "$INPUT_DIR" ]; then
    echo "错误: 输入目录不存在: $INPUT_DIR"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 加载环境变量
export OCR_MODEL_DIR="${OCR_MODEL_DIR:-./models/ocr}"
export PADDLEOCR_HOME="${PADDLEOCR_HOME:-./models/ocr}"
export HF_HOME="${HF_HOME:-./models/sbert}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export OCR_OFFLINE="${OCR_OFFLINE:-1}"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# 运行检测
python3 main.py \
    --config deploy/config.production.json \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --offline \
    "$@"

echo ""
echo "检测完成，报告输出到: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"/
