#!/usr/bin/env python3
"""
PaddleOCR 模型预下载工具

在部署前将 OCR 模型下载到指定目录，支持离线部署。
支持 PaddleOCR 2.x 和 3.x。

用法:
    # 下载 PaddleOCR 2.x 模型
    python deploy/download_ocr_models.py --output ./models/ocr --version 2

    # 下载 PaddleOCR 3.x 模型
    python deploy/download_ocr_models.py --output ./models/ocr --version 3

    # 列出已下载的模型
    python deploy/download_ocr_models.py --list

服务器部署步骤:
    1. 在有网络的机器上运行此脚本下载模型
    2. 将模型目录打包: tar -czf ocr_models.tar.gz ./models/ocr
    3. 上传到服务器并解压
    4. 设置环境变量: export OCR_MODEL_DIR=/path/to/models/ocr
    5. 使用离线模式运行: python main.py --offline --input ... --output ...
"""

import os
import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_paddleocr_v2(output_dir: str):
    """下载 PaddleOCR 2.x 模型（识别 + 检测）

    PaddleOCR 2.x 模型结构:
        ~/.paddleocr/whl/rec/ch/ch_PP-OCRv4_rec_infer/
        ~/.paddleocr/whl/det/ch/ch_PP-OCRv4_det_infer/
    """
    logger.info("下载 PaddleOCR 2.x 模型...")

    try:
        from paddleocr import PaddleOCR
    except ImportError:
        logger.error(
            "PaddleOCR 未安装。请先安装:\n"
            "  pip install paddleocr==2.10.0 paddlepaddle"
        )
        return False

    # 初始化 OCR 引擎触发模型下载
    logger.info("正在下载识别模型 (PP-OCRv4_rec)...")
    try:
        ocr = PaddleOCR(
            lang='ch',
            use_angle_cls=False,
            show_log=True,
            use_gpu=False,
            det=False,   # 只下载识别模型
            rec=True,
        )
        logger.info("识别模型下载成功")
    except Exception as e:
        logger.warning(f"识别模型下载失败: {e}")

    # 如果也需要检测模型
    logger.info("正在下载检测模型 (PP-OCRv4_det)...")
    try:
        ocr_det = PaddleOCR(
            lang='ch',
            use_angle_cls=False,
            show_log=True,
            use_gpu=False,
            det=True,
            rec=True,
        )
        logger.info("检测模型下载成功")
    except Exception as e:
        logger.warning(f"检测模型下载失败: {e}")

    # 复制到目标目录
    src_dir = os.path.expanduser('~/.paddleocr')
    if os.path.exists(src_dir):
        _copy_models(src_dir, output_dir)
        logger.info(f"模型已复制到: {output_dir}")
    else:
        logger.warning(f"源模型目录不存在: {src_dir}")

    return True


def download_paddleocr_v3(output_dir: str):
    """下载 PaddleOCR 3.x 模型（使用 paddlex pipelines）

    PaddleOCR 3.x 模型存储:
        ~/.paddlex/official_models/PP-OCRv5_mobile_rec/
        ~/.paddlex/official_models/PP-OCRv5_server_det/
    """
    logger.info("下载 PaddleOCR 3.x 模型...")

    try:
        from paddleocr import PaddleOCR
    except ImportError:
        logger.error(
            "PaddleOCR 3.x 未安装。请先安装:\n"
            "  pip install paddleocr>=3.0 paddlepaddle"
        )
        return False

    logger.info("正在下载 PaddleOCR v3 模型（检测 + 识别）...")
    try:
        ocr = PaddleOCR(
            lang='ch',
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        # 触发模型加载和下载
        import numpy as np
        test_img = np.ones((100, 200, 3), dtype=np.uint8) * 255
        try:
            ocr.predict(test_img)
        except Exception:
            pass  # 测试图片无文字，忽略
        logger.info("PaddleOCR v3 模型下载成功")
    except Exception as e:
        logger.error(f"PaddleOCR v3 模型下载失败: {e}")
        return False

    # 复制到目标目录
    src_dir = os.path.expanduser('~/.paddlex')
    if os.path.exists(src_dir):
        _copy_models(src_dir, output_dir)
        logger.info(f"模型已复制到: {output_dir}")
    else:
        logger.warning(f"源模型目录不存在: {src_dir}")

    return True


def list_downloaded_models():
    """列出已下载的 PaddleOCR 模型"""
    logger.info("=" * 50)
    logger.info("已下载的 PaddleOCR 模型")
    logger.info("=" * 50)

    for name, path in [
        ('PaddleOCR 2.x (~/.paddleocr)', os.path.expanduser('~/.paddleocr')),
        ('PaddleOCR 3.x (~/.paddlex)', os.path.expanduser('~/.paddlex')),
    ]:
        if os.path.exists(path):
            total_size = 0
            logger.info(f"\n{name}:")
            for item in sorted(os.listdir(path)):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path) and not item.startswith('.'):
                    size = sum(
                        os.path.getsize(os.path.join(root, f))
                        for root, _, files in os.walk(item_path)
                        for f in files
                    )
                    total_size += size
                    logger.info(f"  {item}: {size / 1024 / 1024:.1f} MB")
            logger.info(f"  总计: {total_size / 1024 / 1024:.1f} MB")
        else:
            logger.info(f"\n{name}: (不存在)")


def _copy_models(src_dir: str, dst_dir: str):
    """复制模型文件到目标目录"""
    import shutil

    os.makedirs(dst_dir, exist_ok=True)

    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dst_path = os.path.join(dst_dir, item)

        if item.startswith('.'):
            continue

        if os.path.isdir(src_path):
            if os.path.exists(dst_path):
                logger.info(f"  跳过已存在: {item}")
                continue
            shutil.copytree(src_path, dst_path)
            logger.info(f"  复制目录: {item}")


def main():
    parser = argparse.ArgumentParser(
        description='PaddleOCR 模型预下载工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 下载 v2 模型
  python deploy/download_ocr_models.py --output ./models/ocr

  # 指定版本
  python deploy/download_ocr_models.py --output ./models/ocr --version 3

  # 列出已下载的模型
  python deploy/download_ocr_models.py --list
        """
    )

    parser.add_argument(
        '--output', type=str, default='./models/ocr',
        help='模型输出目录 (默认: ./models/ocr)'
    )
    parser.add_argument(
        '--version', type=int, default=2, choices=[2, 3],
        help='PaddleOCR 版本 (默认: 2)'
    )
    parser.add_argument(
        '--list', action='store_true', default=False,
        help='列出已下载的模型'
    )

    args = parser.parse_args()

    if args.list:
        list_downloaded_models()
        return

    logger.info(f"目标目录: {args.output}")
    logger.info(f"PaddleOCR 版本: {args.version}")

    if args.version == 2:
        success = download_paddleocr_v2(args.output)
    else:
        success = download_paddleocr_v3(args.output)

    if success:
        logger.info("\n下载完成！部署说明:")
        logger.info(f"  1. 设置环境变量: export OCR_MODEL_DIR={os.path.abspath(args.output)}")
        logger.info("  2. 运行检测: python main.py --offline --input <dir> --output <dir>")
    else:
        logger.error("下载失败，请检查网络连接和 PaddleOCR 安装。")
        sys.exit(1)


if __name__ == '__main__':
    main()
