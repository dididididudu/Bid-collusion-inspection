# 投标文件串标围标检测系统 — Docker GPU 部署
# 构建: docker build -t bid-detector -f deploy/Dockerfile .
# 运行: docker run --gpus all -v /data/bids:/input -v /data/output:/output bid-detector

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements-gpu.txt .
RUN pip install --no-cache-dir -r requirements-gpu.txt

# 预下载模型（构建时缓存）
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', \
    cache_folder='/app/models', device='cpu')"

# 复制代码
COPY . .

# 入口
ENTRYPOINT ["python", "main.py"]
CMD ["--input", "/input", "--output", "/output", "--gpu", "--log-level", "INFO"]
