"""
配置参数系统
"""
from typing import Optional
from dataclasses import dataclass, field
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class DetectionConfig:
    """检测配置参数"""

    TEXT_GLOBAL_THRESHOLD: float = 0.80
    TEXT_LOCAL_THRESHOLD: float = 0.85
    TYPO_MIN_LENGTH: int = 4

    SCORE_WEIGHT_MAX: float = 0.35
    SCORE_WEIGHT_TOP_K: float = 0.30
    SCORE_WEIGHT_MEAN: float = 0.25
    SCORE_TOP_K: int = 8

    SBERT_BASE_THRESHOLD: float = 0.60
    SBERT_SHORT_PARAGRAPH_THRESHOLD: float = 0.80
    SBERT_SHORT_PARAGRAPH_LEN: int = 100

    CLONE_BLOCK_MIN_LENGTH: int = 2
    CLONE_BLOCK_MAX_GAP: int = 2

    METADATA_MATCH_THRESHOLD: int = 3
    TIME_BUCKET_FORMAT: str = "%Y-%m-%dT%H"

    QUOTE_COMMON_THRESHOLD: int = 2
    QUOTE_TAIL_SIM_THRESHOLD: float = 0.80
    MIN_QUOTES_FOR_PATTERN: int = 3

    IMAGE_COMMON_THRESHOLD: int = 1

    RISK_HIGH_THRESHOLD: int = 70
    RISK_MEDIUM_THRESHOLD: int = 40
    RISK_LOW_THRESHOLD: int = 15

    MAX_WORKERS: int = 8
    MINHASH_BANDS: int = 8
    MINHASH_ROWS: int = 4
    MINHASH_JACCARD_THRESHOLD: float = 0.2
    MAX_CANDIDATE_PAIRS: int = 5000

    STOPWORDS_PATH: Optional[str] = None

    CHUNK_PAGE_SIZE: int = 50
    MAX_CHUNKS_IN_MEMORY: int = 5

    ENABLE_CHECKPOINT: bool = True
    CHECKPOINT_DIR: str = "./checkpoints"
    CACHE_DIR: str = "./cache"
    CHECKPOINT_INTERVAL: int = 50  # 增大间隔以减少检查点I/O频率

    USE_GPU: bool = False
    SBERT_DEVICE: str = "cpu"
    USE_ONNX: bool = False
    ONNX_MODEL_PATH: Optional[str] = None
    SBERT_BATCH_SIZE: int = 64

    MINHASH_NUM_HASHES: int = 128
    MINHASH_LSH_THRESHOLD: float = 0.3
    PARAGRAPH_LSH_THRESHOLD: float = 0.3
    MINHASH_NUM_HASHES_PARAGRAPH: int = 32

    PARAGRAPH_MATCH_STAGE1_TOP_K: int = 5000
    PARAGRAPH_MATCH_STAGE2_TOP_K: int = 10000
    PARAGRAPH_MIN_JACCARD: float = 0.05

    REPORT_MAX_MATCHES_PER_PAIR: int = 10000
    REPORT_INCLUDE_ALL_MATCHES: bool = True
    REPORT_DETAIL_LEVEL: str = "full"

    PDF_EXTRACTOR_BACKEND: str = "pymupdf"

    MAX_MEMORY_MB: int = 2048

    # 嵌入缓存
    ENABLE_EMBEDDING_CACHE: bool = True  # 全局 SBERT 嵌入缓存（Phase 1.5）
    EMBEDDING_DIM: int = 384  # SBERT 嵌入维度
    EMBED_WORKERS: int = 1  # Phase 1.5 编码工作进程数（模型大，少 worker）

    # 文档级向量预筛
    DOC_VECTOR_FILTER_ENABLED: bool = True
    DOC_VECTOR_THRESHOLD: float = 0.3  # 文档余弦相似度阈值
    METADATA_FILTER_ENABLED: bool = True  # 元数据指纹候选筛选

    DISABLE_CACHE: bool = True

    def __post_init__(self):
        if self.CHUNK_PAGE_SIZE < 10:
            raise ValueError("CHUNK_PAGE_SIZE 必须 >= 10")

        if self.PDF_EXTRACTOR_BACKEND not in ("pymupdf", "pdfplumber"):
            raise ValueError("PDF_EXTRACTOR_BACKEND 必须是 'pymupdf' 或 'pdfplumber'")

        if self.SBERT_DEVICE not in ("cpu", "cuda", "mps", "auto"):
            raise ValueError("SBERT_DEVICE 必须是 'cpu', 'cuda', 'mps' 或 'auto'")

        if self.SBERT_DEVICE == "auto" or self.USE_GPU:
            self._auto_detect_device()

    def _auto_detect_device(self):
        try:
            import torch
            if torch.cuda.is_available():
                self.SBERT_DEVICE = "cuda"
                logger.info("自动检测到 CUDA GPU，将使用 GPU 加速")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.SBERT_DEVICE = "mps"
                logger.info("自动检测到 Apple MPS，将使用 MPS 加速")
            else:
                self.SBERT_DEVICE = "cpu"
                logger.info("未检测到 GPU，将使用 CPU")
        except ImportError:
            self.SBERT_DEVICE = "cpu"
            logger.warning("torch 未安装，将使用 CPU")

    @classmethod
    def from_json(cls, config_path: str) -> 'DetectionConfig':
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)

            valid_fields = {k: v for k, v in config_dict.items()
                           if k in cls.__dataclass_fields__}
            if len(valid_fields) < len(config_dict):
                ignored = set(config_dict.keys()) - set(valid_fields.keys())
                logger.debug(f"忽略未知配置字段: {ignored}")

            return cls(**valid_fields)
        except Exception as e:
            raise ValueError(f"加载配置文件失败: {e}")

    def to_json(self, config_path: str) -> None:
        config_dict = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)


def load_config(config_path: Optional[str] = None) -> DetectionConfig:
    if config_path:
        return DetectionConfig.from_json(config_path)
    return DetectionConfig()