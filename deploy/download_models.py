#!/usr/bin/env python3
"""
统一模型下载脚本 — PaddleOCR 2.10 + SBERT

用法:
    # 下载全部模型
    python deploy/download_models.py --all

    # 仅下载 OCR 模型
    python deploy/download_models.py --ocr

    # 仅下载 SBERT 模型
    python deploy/download_models.py --sbert

    # 指定输出目录
    python deploy/download_models.py --all --output ./models

服务器部署:
    1. 运行此脚本下载模型到 ./models/
    2. 将 models/ 目录打包上传到服务器
    3. 设置环境变量指向模型目录
"""

import os
import sys
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def download_paddleocr_models(output_dir: str):
    """下载 PaddleOCR 2.10 识别模型 (PP-OCRv4)

    模型大小: ~5MB (识别模型)
    存储路径: {output_dir}/ocr/
    """
    os.environ["PADDLEOCR_HOME"] = os.path.join(output_dir, "ocr")
    os.makedirs(os.environ["PADDLEOCR_HOME"], exist_ok=True)

    logger.info("下载 PaddleOCR 2.10 识别模型 (PP-OCRv4_rec)...")
    try:
        from paddleocr import PaddleOCR
        # 初始化触发模型下载 (det=False 仅识别)
        ocr = PaddleOCR(
            lang="ch",
            use_angle_cls=False,
            show_log=False,
            use_gpu=False,
            det=False,   # 仅识别，不检测
            rec=True,
        )
        logger.info("PaddleOCR 识别模型就绪")
        return True
    except ImportError:
        logger.error("PaddleOCR 未安装，请先运行: pip install paddleocr==2.10.0")
        return False
    except Exception as e:
        logger.warning(f"PaddleOCR 模型下载失败: {e}")
        return False


def download_sbert_models(output_dir: str):
    """下载 SBERT 语义模型

    模型: paraphrase-multilingual-MiniLM-L12-v2
    大小: ~120MB
    存储路径: {output_dir}/sbert/
    """
    os.environ["HF_HOME"] = os.path.join(output_dir, "sbert")
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)

    model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    logger.info(f"下载 SBERT 模型 ({model_name}, ~120MB)...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name, device="cpu")
        logger.info(f"SBERT 模型就绪 (维度: {model.get_sentence_embedding_dimension()})")
        return True
    except ImportError:
        logger.error("sentence-transformers 未安装")
        return False
    except Exception as e:
        logger.warning(f"SBERT 模型下载失败: {e}")
        return False


def list_models(output_dir: str):
    """列出已下载的模型"""
    logger.info("=" * 50)
    logger.info("已下载的模型")

    for name, subdir in [("PaddleOCR", "ocr"), ("SBERT", "sbert")]:
        path = os.path.join(output_dir, subdir)
        if os.path.exists(path):
            total_size = 0
            for root, dirs, files in os.walk(path):
                for f in files:
                    try:
                        total_size += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
            logger.info(f"  {name}: {path} ({total_size / 1024 / 1024:.1f} MB)")
        else:
            logger.info(f"  {name}: (未下载)")

    logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="统一模型下载")
    parser.add_argument("--all", action="store_true", help="下载全部模型")
    parser.add_argument("--ocr", action="store_true", help="仅下载 PaddleOCR 模型")
    parser.add_argument("--sbert", action="store_true", help="仅下载 SBERT 模型")
    parser.add_argument("--output", default="./models", help="模型输出目录")
    parser.add_argument("--list", action="store_true", help="列出已下载的模型")

    args = parser.parse_args()

    if args.list:
        list_models(args.output)
        return

    if not (args.all or args.ocr or args.sbert):
        parser.print_help()
        return

    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    success = True

    if args.all or args.ocr:
        if not download_paddleocr_models(output_dir):
            success = False

    if args.all or args.sbert:
        if not download_sbert_models(output_dir):
            success = False

    if success:
        logger.info("=" * 50)
        logger.info("模型下载完成!")
        logger.info(f"模型目录: {output_dir}")
        logger.info("")
        logger.info("部署步骤:")
        logger.info(f"  1. 打包: tar -czf models.tar.gz {output_dir}")
        logger.info("  2. 上传到服务器并解压")
        logger.info(f"  3. 设置环境变量:")
        logger.info(f"     export OCR_MODEL_DIR={output_dir}/ocr")
        logger.info(f"     export PADDLEOCR_HOME={output_dir}/ocr")
        logger.info(f"     export HF_HOME={output_dir}/sbert")
        logger.info("=" * 50)
    else:
        logger.error("部分模型下载失败，请检查网络或重试")
        sys.exit(1)


if __name__ == "__main__":
    main()
