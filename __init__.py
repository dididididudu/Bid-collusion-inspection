"""
BatchBidCollusionDetector
投标文件串标围标检测系统
Version 1.0.0
"""

__version__ = "1.0.0"
__author__ = "BatchBidCollusionDetector Development Team"
__description__ = "投标文件串标围标检测系统 - 自动化检测PDF投标文件中的串标、围标行为"

# 模块导出
from .config import DetectionConfig, load_config
from .data_structures import (
    BidFeature,
    MetadataFeature,
    QuoteSignature,
    PairwiseResult,
    EvidenceChain,
    GlobalReport,
    Cluster,
    FileProfile
)
from .extractor import DocumentFeatureExtractor
from .selector import CandidatePairSelector
from .analyzer import PairwiseAnalyzer
from .scoring import RiskScoringEngine
from .report import ReportGenerator

__all__ = [
    'DetectionConfig',
    'load_config',
    'BidFeature',
    'MetadataFeature',
    'QuoteSignature',
    'PairwiseResult',
    'EvidenceChain',
    'GlobalReport',
    'Cluster',
    'FileProfile',
    'DocumentFeatureExtractor',
    'CandidatePairSelector',
    'PairwiseAnalyzer',
    'RiskScoringEngine',
    'ReportGenerator',
]
