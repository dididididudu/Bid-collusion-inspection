"""
BatchBidCollusionDetector - 投标文件串标围标检测系统
主程序入口
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = './models'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import sys
import argparse
import logging
import glob
from datetime import datetime
from typing import List, Optional

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
    
    logging.getLogger('pdfminer').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('sklearn').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    logging.info(f"日志系统已初始化: 级别={log_level}, 最大文件大小=10MB, 保留备份=3个")


def main(input_dir: str, output_dir: str, config_path: str = None,
         log_level: str = "INFO", use_gpu: bool = False,
         no_checkpoint: bool = False) -> None:
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("BatchBidCollusionDetector - 投标文件串标围标检测系统")
    logger.info("=" * 60)

    try:
        logger.info("加载配置...")
        config = load_config(config_path)

        if use_gpu:
            config.USE_GPU = True
            config.SBERT_DEVICE = "auto"
            logger.info("已启用 GPU 加速")
        if no_checkpoint:
            config.ENABLE_CHECKPOINT = False
            logger.info("已禁用断点续传")

        logger.info(f"配置加载完成: MAX_WORKERS={config.MAX_WORKERS}, CHECKPOINT_INTERVAL={config.CHECKPOINT_INTERVAL}")

        service = BidDetectionOrchestrator(config)

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
        required=True,
        help='输入PDF文件目录路径'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='输出报告目录路径'
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

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

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
    )