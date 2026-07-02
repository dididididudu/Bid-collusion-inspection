# ============================================================
# 投标串标检测系统 — Docker 部署
#
# 构建: docker build -t bid-detector -f deploy/Dockerfile .
# 运行:
#   CPU: docker run -v /data/bids:/input -v /data/output:/output bid-detector
#   GPU: docker run --gpus all -v /data/bids:/input -v /data/output:/output bid-detector --gpu
# ============================================================

# ---- Stage 1: 模型下载 ----
FROM python:3.10-slim AS model-builder

WORKDIR /app
COPY deploy/requirements.production.txt /app/
RUN pip install --no-cache-dir -r requirements.production.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 下载 SBERT 模型到 /app/models/sbert
RUN python -c "
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from sentence_transformers import SentenceTransformer
SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2',
    cache_folder='/app/models/sbert', device='cpu')
"

# 下载 PaddleOCR 识别模型到 /app/models/ocr
RUN python -c "
import os
os.environ['PADDLEOCR_HOME'] = '/app/models/ocr'
from paddleocr import PaddleOCR
PaddleOCR(lang='ch', use_angle_cls=False, show_log=False,
          use_gpu=False, det=False, rec=True)
"

# ---- Stage 2: 运行环境 ----
FROM python:3.10-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 复制模型（从 Stage 1）
COPY --from=model-builder /app/models /app/models

# 安装 Python 依赖（不含模型重下载）
COPY deploy/requirements.production.txt /app/
RUN pip install --no-cache-dir -r requirements.production.txt \
    && rm requirements.production.txt

# 复制应用代码
COPY . /app/

# 环境变量
ENV OCR_MODEL_DIR=/app/models/ocr
ENV PADDLEOCR_HOME=/app/models/ocr
ENV HF_HOME=/app/models/sbert
ENV HF_ENDPOINT=https://hf-mirror.com
ENV OCR_OFFLINE=1
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# 入口
ENTRYPOINT ["python", "main.py"]
CMD ["--input", "/input", "--output", "/output", "--offline", "--log-level", "INFO"]
