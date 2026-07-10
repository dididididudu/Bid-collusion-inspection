"""
BatchBidCollusionDetector - 投标文件串标围标检测系统
主程序入口
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = './models'
os.environ.setdefault('USE_TF', 'FALSE')  # 阻止旧版TF与新numpy冲突
# 注意: 不在此处设置 TRANSFORMERS_OFFLINE=1
# 首次运行时需要在线下载/验证模型，仅在 --offline 模式下启用离线限制

import multiprocessing
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # 已设置过

import sys
import argparse
import logging
from datetime import datetime

from config import load_config, DetectionConfig
from scoring import RiskScoringEngine
from report import ReportGenerator
from data_structures import GlobalReport
from pipeline.orchestrator import BidDetectionOrchestrator


def setup_logging(log_level: str = "INFO") -> None:
    from logging.handlers import RotatingFileHandler
    
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(getattr(logging, log_level.upper()))
    
    file_handler = RotatingFileHandler(
        'detection.log',
        mode='a',
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
        encoding='utf-8',
        delay=False
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(getattr(logging, log_level.upper()))
    
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)
    
    # 静默第三方库的 INFO/DEBUG 日志
    for _lib in ['pdfminer', 'transformers', 'sentence_transformers', 'sklearn',
                 'PIL', 'jieba', 'paddleocr', 'paddle', 'paddlex',
                 'easyocr', 'torch', 'cv2', 'matplotlib', 'urllib3',
                 'PIL.PngImagePlugin', 'PIL.TiffImagePlugin']:
        logging.getLogger(_lib).setLevel(logging.WARNING)
    # 管道内部模块日志较细，默认只显示 WARNING 以上
    for _mod in ['pipeline.checkpoint', 'pipeline.streaming_context',
                 'pipeline.parallel_workers', 'pipeline.ocr_helpers',
                 'extraction.pdf_extractor', 'extraction.feature_cache',
                 'extraction.text_processor', 'matching.paragraph_matcher',
                 'matching.semantic_matcher', 'matching.lsh_index',
                 'matching.selector', 'embedding.embedding_engine',
                 'image_analysis.image_hasher', 'image_analysis.image_matcher',
                 'image_analysis.image_ocr']:
        logging.getLogger(_mod).setLevel(logging.WARNING)
    # orchestrator 和 scoring 保留 INFO（阶段进度和风险结果）
    
    logging.info(f"日志系统已初始化: 级别={log_level}, 最大文件大小=10MB, 保留备份=3个")


def _print_startup_diagnostics(config: DetectionConfig) -> None:
    """打印启动诊断信息"""
    logger = logging.getLogger(__name__)
    import platform

    logger.info("=" * 60)
    logger.info("启动诊断")
    logger.info("=" * 60)
    logger.info(f"Python 版本: {platform.python_version()}")
    logger.info(f"平台: {platform.platform()}")

    # GPU 检测
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        logger.info(f"PyTorch: {torch.__version__}, CUDA: {cuda_ok}")
        if cuda_ok:
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        logger.info("PyTorch: 未安装")

    # 配置摘要
    logger.info(f"OCR 引擎: {config.OCR_ENGINE}")
    logger.info(f"OCR 离线模式: {config.OCR_OFFLINE_MODE}")
    if config.OCR_MODEL_DIR:
        logger.info(f"OCR 模型目录: {config.OCR_MODEL_DIR}")
    if config.PADDLEOCR_HOME:
        logger.info(f"PaddleOCR 缓存: {config.PADDLEOCR_HOME}")
    logger.info(f"SBERT 设备: {config.SBERT_DEVICE}")
    logger.info(f"并行: Phase1={config.PHASE1_WORKERS}, Phase3={config.PHASE3_WORKERS}")
    logger.info(f"缓存禁用: {config.DISABLE_CACHE}")
    logger.info("=" * 60)


def main(input_dir: str, output_dir: str, config_path: str = None,
         log_level: str = "INFO", use_gpu: bool = False,
         no_checkpoint: bool = False, offline: bool = False,
         dimensions: str = None) -> None:
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("BatchBidCollusionDetector - 投标文件串标围标检测")
    logger.info("=" * 60)

    try:
        logger.info("加载配置...")
        config = load_config(config_path)

        if use_gpu:
            config.USE_GPU = True
            config._auto_detect_device()
            logger.info(f"已启用 GPU 加速 (SBERT 设备: {config.SBERT_DEVICE})")
        if no_checkpoint:
            config.ENABLE_CHECKPOINT = False
            logger.info("已禁用断点续传")
        if offline:
            config.OCR_OFFLINE_MODE = True
            os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_HUB_OFFLINE'] = '1'
            logger.info("已启用离线模式")
        if dimensions:
            enabled_list = [d.strip() for d in dimensions.split(',') if d.strip()]
            for dim in config.ENABLED_DIMENSIONS:
                config.ENABLED_DIMENSIONS[dim] = dim in enabled_list
            logger.info(f"已限制检测维度: {enabled_list}")
        # 打印启动诊断
        _print_startup_diagnostics(config)

        logger.info(f"配置加载完成: Phase1={config.PHASE1_WORKERS}, Phase3={config.PHASE3_WORKERS}, OCR={config.ENABLE_OCR}")

        # 预加载 SBERT 模型（避免 Phase 1.5/3 中重复加载）
        logger.info("预加载 SBERT 模型...")
        sbert_model = None
        try:
            from sentence_transformers import SentenceTransformer
            sbert_model = SentenceTransformer(
                'paraphrase-multilingual-MiniLM-L12-v2',
                device=config.SBERT_DEVICE if config.SBERT_DEVICE != "auto" else "cpu",
                cache_folder='./models',
                trust_remote_code=True,
                local_files_only=False,
            )
            logger.info(f"SBERT 模型预加载完成 (设备: {sbert_model.device})")
        except Exception as e:
            logger.warning(f"SBERT 模型预加载失败（后续将按需加载）: {e}")

        # 自动清除旧缓存文件（每次启动干净运行）
        if config.DISABLE_CACHE:
            import glob as _glob
            for _f in _glob.glob(os.path.join(config.CACHE_DIR, "features.db*")):
                try:
                    os.remove(_f)
                except OSError:
                    pass
            for _f in _glob.glob(os.path.join(config.CHECKPOINT_DIR, "*")):
                try:
                    os.remove(_f)
                except OSError:
                    pass
            logger.info("旧缓存文件已清除")

        service = BidDetectionOrchestrator(config)

        if sbert_model is not None:
            service.embedding_engine.set_model(sbert_model)
            logger.info("预加载 SBERT 模型已注入 EmbeddingEngine")

        process_start = datetime.now()
        service.detect(input_dir, output_dir)
        process_time = (datetime.now() - process_start).total_seconds()

        logger.info(f"总耗时: {process_time:.2f}秒")

    except Exception as e:
        logger.error(f"程序执行出错: {e}", exc_info=True)
        raise


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='BatchBidCollusionDetector - 投标文件串标围标检测系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py --input ./bids/ --output ./report/
  python main.py --input ./bids/ --output ./report/ --config config.json
  python main.py --input ./bids/ --output ./report/ --log-level DEBUG
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=False,
        help='输入PDF文件目录路径 (--diagnose 模式不需要)'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=False,
        help='输出报告目录路径 (--diagnose 模式不需要)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='配置文件路径（JSON格式，可选）'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='日志级别（默认：INFO）'
    )

    parser.add_argument(
        '--gpu',
        action='store_true',
        default=False,
        help='启用 GPU 加速（需要 CUDA/MPS）'
    )

    parser.add_argument(
        '--no-checkpoint',
        action='store_true',
        default=False,
        help='禁用断点续传'
    )

    parser.add_argument(
        '--offline',
        action='store_true',
        default=False,
        help='离线模式：禁止所有模型在线下载'
    )

    parser.add_argument(
        '--dimensions',
        type=str,
        default=None,
        help='启用的检测维度，逗号分隔，如: file_id,text,contact (默认全部启用)'
    )

    parser.add_argument(
        '--diagnose',
        action='store_true',
        default=False,
        help='仅运行环境诊断，不执行检测'
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # --diagnose 模式：仅打印诊断信息后退出
    if args.diagnose:
        setup_logging(args.log_level)
        logger = logging.getLogger(__name__)
        config = load_config(args.config)
        if args.gpu:
            config.USE_GPU = True
        if args.offline:
            config.OCR_OFFLINE_MODE = True
        _print_startup_diagnostics(config)

        # 打印 OCR 详细诊断
        from image_analysis.image_ocr import ImageOCREngine
        print(ImageOCREngine.diagnose())
        sys.exit(0)

    if not os.path.exists(args.input):
        print(f"错误: 输入目录不存在: {args.input}")
        sys.exit(1)

    if not os.path.isdir(args.input):
        print(f"错误: 输入路径不是目录: {args.input}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    main(
        input_dir=args.input,
        output_dir=args.output,
        config_path=args.config,
        log_level=args.log_level,
        use_gpu=args.gpu,
        no_checkpoint=args.no_checkpoint,
        offline=args.offline,
        dimensions=args.dimensions,
    )