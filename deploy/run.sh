#!/bin/bash
# ============================================================
# 投标文件串标围标检测系统 — 快速运行脚本
# ============================================================
# 用法:
#   ./run.sh <PDF目录> <输出目录>
#   ./run.sh ./bids ./reports
#   ./run.sh ./bids ./reports --gpu
# ============================================================
set -e

# 参数
INPUT_DIR="${1:-./test_data/input}"
OUTPUT_DIR="${2:-./test_data/output}"
shift 2 2>/dev/null || true
EXTRA_ARGS="$@"

# 激活虚拟环境
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 检测 GPU
GPU_FLAG=""
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    GPU_FLAG="--gpu"
fi

echo "=============================================="
echo "  投标文件串标围标检测"
echo "  输入: $INPUT_DIR"
echo "  输出: $OUTPUT_DIR"
echo "  GPU:  $([ -n "$GPU_FLAG" ] && echo '启用' || echo '未检测到')"
echo "=============================================="

# 运行
python main.py \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    $GPU_FLAG \
    --log-level INFO \
    $EXTRA_ARGS

echo ""
echo "=============================================="
echo "  检测完成"
echo "  报告: $OUTPUT_DIR/summary.txt"
echo "  网页: $OUTPUT_DIR/detection_report.html"
echo "=============================================="
