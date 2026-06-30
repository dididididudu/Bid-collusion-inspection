"""
配置参数系统
"""
from typing import Optional
from dataclasses import dataclass, field
import json


@dataclass
class DetectionConfig:
    """检测配置参数"""

    # 文本相似度（SBERT专用阈值 - 针对投标文件优化）
    TEXT_GLOBAL_THRESHOLD: float = 0.85  # SBERT语义相似度阈值，提高以减少误报
    TEXT_LOCAL_THRESHOLD: float = 0.92   # 段落级相似度阈值，只保留真正高度相似的段落
    TYPO_MIN_LENGTH: int = 4
    
    # 混合评分权重
    SCORE_WEIGHT_MAX: float = 0.4  # 最大相似度权重
    SCORE_WEIGHT_TOP_K: float = 0.3  # Top-K相似度权重
    SCORE_WEIGHT_MEAN: float = 0.2  # 平均相似度权重
    SCORE_WEIGHT_COVERAGE: float = 0.1  # 覆盖率权重
    SCORE_TOP_K: int = 5  # Top-K取前K个
    
    # SBERT动态阈值参数
    SBERT_BASE_THRESHOLD: float = 0.75  # 基础阈值
    SBERT_SHORT_PARAGRAPH_THRESHOLD: float = 0.85  # 短段落阈值（<100字符）
    SBERT_SHORT_PARAGRAPH_LEN: int = 100  # 短段落长度阈值
    
    # 连续克隆块参数
    CLONE_BLOCK_MIN_LENGTH: int = 3  # 最小连续段落数
    CLONE_BLOCK_MAX_GAP: int = 1  # 允许的最大间隔

    # 元数据
    METADATA_MATCH_THRESHOLD: int = 3
    TIME_BUCKET_FORMAT: str = "%Y-%m-%dT%H"  # 按小时分桶

    # 报价
    QUOTE_COMMON_THRESHOLD: int = 2
    QUOTE_TAIL_SIM_THRESHOLD: float = 0.80
    MIN_QUOTES_FOR_PATTERN: int = 3

    # 图片
    IMAGE_COMMON_THRESHOLD: int = 1

    # 风险评分
    RISK_HIGH_THRESHOLD: int = 70
    RISK_MEDIUM_THRESHOLD: int = 40
    RISK_LOW_THRESHOLD: int = 15

    # 性能
    MAX_WORKERS: int = 8  # 并行进程数
    SIMHASH_BANDS: int = 4  # LSH band 数
    SIMHASH_ROWS: int = 16  # 每 band 行数
    MAX_CANDIDATE_PAIRS: int = 5000  # 候选对上限，超限报警

    # 停用词表路径（支持外部文件加载）
    STOPWORDS_PATH: Optional[str] = None

    # 文本长度限制
    MAX_TEXT_LENGTH: int = 100000  # 防止超大文件内存溢出

    @classmethod
    def from_json(cls, config_path: str) -> 'DetectionConfig':
        """从JSON配置文件加载配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            return cls(**config_dict)
        except Exception as e:
            raise ValueError(f"加载配置文件失败: {e}")

    def to_json(self, config_path: str) -> None:
        """保存配置到JSON文件"""
        config_dict = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)


def load_config(config_path: Optional[str] = None) -> DetectionConfig:
    """加载配置"""
    if config_path:
        return DetectionConfig.from_json(config_path)
    return DetectionConfig()
