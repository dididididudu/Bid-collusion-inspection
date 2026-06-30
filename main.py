"""
BatchBidCollusionDetector - 投标文件串标围标检测系统
主程序入口
"""
import os
# 设置Hugging Face国内镜像源，加速SBERT模型下载
# 必须在导入任何transformers相关库之前设置
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = './models'
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # 优先使用本地缓存，避免网络请求

import sys
import argparse
import logging
import glob
from datetime import datetime
from typing import List

from config import load_config, DetectionConfig
from extractor import DocumentFeatureExtractor
from selector import CandidatePairSelector
from analyzer import PairwiseAnalyzer
from scoring import RiskScoringEngine
from report import ReportGenerator
from data_structures import BidFeature


def setup_logging(log_level: str = "INFO") -> None:
    """配置日志系统 - 优化版：日志轮转+第三方库过滤"""
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


def scan_pdf_files(input_dir: str) -> List[str]:
    """扫描输入目录，收集所有PDF文件路径"""
    pattern = os.path.join(input_dir, "*.pdf")
    pdf_files = glob.glob(pattern)

    if not pdf_files:
        logging.warning(f"未在目录 {input_dir} 中找到PDF文件")

    return pdf_files


def main(input_dir: str, output_dir: str, config_path: str = None, log_level: str = "INFO") -> None:
    """
    主流程

    Args:
        input_dir: 输入PDF文件目录
        output_dir: 输出报告目录
        config_path: 配置文件路径（可选）
        log_level: 日志级别
    """
    # 1. 配置日志
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("BatchBidCollusionDetector - 投标文件串标围标检测系统")
    logger.info("=" * 60)

    start_time = datetime.now()

    try:
        # 2. 加载配置
        logger.info("加载配置...")
        config = load_config(config_path)
        logger.info(f"配置加载完成: MAX_WORKERS={config.MAX_WORKERS}")

        # 3. 扫描输入目录
        logger.info(f"扫描输入目录: {input_dir}")
        file_paths = scan_pdf_files(input_dir)

        if not file_paths:
            logger.error("未找到PDF文件，程序退出")
            return

        logger.info(f"找到 {len(file_paths)} 个PDF文件")

        # 4. 模块 A：批量特征提取（并行）
        logger.info("=" * 60)
        logger.info("模块 A：特征提取")
        logger.info("=" * 60)
        extractor = DocumentFeatureExtractor(config)
        features = extractor.batch_extract(file_paths)

        if not features:
            logger.error("未能提取任何有效特征，程序退出")
            return

        logger.info(f"成功提取 {len(features)} 个文档特征")

        # 5. 模块 B：快速初筛
        logger.info("=" * 60)
        logger.info("模块 B：快速初筛")
        logger.info("=" * 60)
        selector = CandidatePairSelector(config)
        candidate_pairs = selector.select(features)

        if not candidate_pairs:
            logger.info("未发现候选对，所有文档间无明显相似性")
            # 仍然生成报告
            scoring_engine = RiskScoringEngine(config)
            global_report = scoring_engine.generate_report([], features)

            report_generator = ReportGenerator(config)
            report_generator.generate(global_report, output_dir)

            logger.info("检测完成，未发现可疑文档对")
            return

        logger.info(f"初筛完成，候选对数: {len(candidate_pairs)}")

        # 6. 模块 C：精细比对（并行处理候选对）
        logger.info("=" * 60)
        logger.info("模块 C：精细相似度分析")
        logger.info("=" * 60)
        analyzer = PairwiseAnalyzer(config)

        # 构建文档ID到特征的映射
        feature_map = {f.doc_id: f for f in features}

        pairwise_results = []
        total_pairs = len(candidate_pairs)

        for i, (id_a, id_b) in enumerate(candidate_pairs, 1):
            if i % 10 == 0 or i == total_pairs:
                logger.info(f"分析进度: {i}/{total_pairs}")

            feature_a = feature_map.get(id_a)
            feature_b = feature_map.get(id_b)

            if feature_a and feature_b:
                result = analyzer.analyze(feature_a, feature_b)
                pairwise_results.append(result)

        logger.info(f"精细分析完成，共分析 {len(pairwise_results)} 对")

        # 7. 模块 D：风险评级与聚类
        logger.info("=" * 60)
        logger.info("模块 D：风险评级与聚类")
        logger.info("=" * 60)
        scoring_engine = RiskScoringEngine(config)
        global_report = scoring_engine.generate_report(pairwise_results, features)

        logger.info(f"风险评级完成:")
        logger.info(f"  - 可疑对数: {global_report.suspicious_pairs}")
        logger.info(f"  - 高风险对数: {global_report.high_risk_pairs}")
        logger.info(f"  - 风险聚类数: {len(global_report.risk_clusters)}")

        # 8. 模块 E：报告生成
        logger.info("=" * 60)
        logger.info("模块 E：报告生成")
        logger.info("=" * 60)
        report_generator = ReportGenerator(config)
        report_generator.generate(global_report, output_dir)

        # 9. 完成
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()

        logger.info("=" * 60)
        logger.info("检测完成")
        logger.info("=" * 60)
        logger.info(f"总耗时: {elapsed_time:.2f} 秒")
        logger.info(f"检测文件数: {global_report.total_files}")
        logger.info(f"可疑对数: {global_report.suspicious_pairs}")
        logger.info(f"高风险对数: {global_report.high_risk_pairs}")
        logger.info(f"报告输出目录: {output_dir}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"程序执行出错: {e}", exc_info=True)
        raise


def parse_arguments():
    """解析命令行参数"""
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

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # 检查输入目录
    if not os.path.exists(args.input):
        print(f"错误: 输入目录不存在: {args.input}")
        sys.exit(1)

    if not os.path.isdir(args.input):
        print(f"错误: 输入路径不是目录: {args.input}")
        sys.exit(1)

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 执行主程序
    main(
        input_dir=args.input,
        output_dir=args.output,
        config_path=args.config,
        log_level=args.log_level
    )
