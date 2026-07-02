"""
配置参数系统 — 支持环境变量覆盖，便于服务器部署
"""
import os
from typing import Optional
from dataclasses import dataclass, field
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class DetectionConfig:
    """检测配置参数

    所有字段均可通过 JSON 配置文件设置。
    部分字段支持环境变量覆盖（优先级: 代码显式设置 > JSON 配置 > 环境变量 > 默认值）。
    """

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
    EMBED_WORKERS: int = 2  # Phase 1.5 编码工作进程数

    # 并行处理
    PHASE1_WORKERS: int = 8       # Phase 1 提取并行度 (ProcessPoolExecutor)
    PHASE3_WORKERS: int = 8       # Phase 3 分析并行度 (ThreadPoolExecutor)
    DB_BUSY_TIMEOUT: int = 30000  # SQLite 写锁等待超时 (ms)

    # 文档级向量预筛
    DOC_VECTOR_FILTER_ENABLED: bool = True
    DOC_VECTOR_THRESHOLD: float = 0.3  # 文档余弦相似度阈值
    # 图片 OCR
    ENABLE_OCR: bool = True              # 自动对页面图片运行 OCR
    OCR_ENGINE: str = "paddleocr"        # OCR 引擎: "paddleocr" / "easyocr"
    OCR_LANGUAGES: list = None           # None = 默认 ['ch_sim', 'en']
    OCR_SAMPLE_STEP: int = 1             # 每隔 N 页运行一次 OCR（1 = 每页）
    OCR_MIN_CONFIDENCE: float = 0.3      # OCR 最低置信度阈值

    # OCR 部署配置
    OCR_MODEL_DIR: Optional[str] = None  # PaddleOCR 自定义模型目录（离线部署）
    OCR_OFFLINE_MODE: bool = False       # 强制离线，禁止模型下载
    PADDLEOCR_HOME: Optional[str] = None # PaddleOCR 缓存根目录（环境变量也可设置）
    OCR_RETRY_COUNT: int = 3             # OCR 单张图片失败重试次数
    ENGINE_INIT_TIMEOUT: int = 120       # 引擎初始化超时（秒）

    # 图片提取
    IMAGE_MIN_SIZE: int = 50             # 嵌入图片最小尺寸（像素），小于此值的过滤

    # 标书模板语过滤（减少因招标文件原文导致的误检）
    BID_BOILERPLATE_FILTER: bool = True  # 启用标书模板语过滤
    BID_BOILERPLATE_WEIGHT: float = 0.3  # 模板语匹配的权重衰减系数（0-1, 越小越严格）

    METADATA_FILTER_ENABLED: bool = True  # 元数据指纹候选筛选

    DISABLE_CACHE: bool = False

    def __post_init__(self):
        if self.CHUNK_PAGE_SIZE < 10:
            raise ValueError("CHUNK_PAGE_SIZE 必须 >= 10")

        if self.PDF_EXTRACTOR_BACKEND not in ("pymupdf", "pdfplumber"):
            raise ValueError("PDF_EXTRACTOR_BACKEND 必须是 'pymupdf' 或 'pdfplumber'")

        if self.SBERT_DEVICE not in ("cpu", "cuda", "mps", "auto"):
            raise ValueError("SBERT_DEVICE 必须是 'cpu', 'cuda', 'mps' 或 'auto'")

        if self.SBERT_DEVICE == "auto" or self.USE_GPU:
            self._auto_detect_device()

        # 从环境变量读取 OCR 部署配置（优先级低于 JSON 配置文件中显式设置的值）
        self._apply_env_overrides()

    def _apply_env_overrides(self):
        """从环境变量读取部署配置，仅覆盖尚未显式设置的字段"""
        # PADDLEOCR_HOME: 如果 JSON 中未设置，从环境变量读取
        if self.PADDLEOCR_HOME is None:
            env_home = os.environ.get('PADDLEOCR_HOME', '')
            if env_home:
                self.PADDLEOCR_HOME = env_home
                logger.info(f"从环境变量读取 PADDLEOCR_HOME={env_home}")

        # OCR_MODEL_DIR: 如果 JSON 中未设置，从环境变量读取
        if self.OCR_MODEL_DIR is None:
            env_model = os.environ.get('OCR_MODEL_DIR', '')
            if env_model:
                self.OCR_MODEL_DIR = env_model
                logger.info(f"从环境变量读取 OCR_MODEL_DIR={env_model}")

        # OCR_OFFLINE_MODE: 支持环境变量 OCR_OFFLINE=1 或 TRANSFORMERS_OFFLINE=1
        if not self.OCR_OFFLINE_MODE:
            if os.environ.get('OCR_OFFLINE', '') == '1':
                self.OCR_OFFLINE_MODE = True
                logger.info("从环境变量启用 OCR 离线模式 (OCR_OFFLINE=1)")

        # 应用离线模式：设置 PaddleOCR 相关环境变量
        if self.OCR_OFFLINE_MODE:
            os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
            os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
            logger.info("OCR 离线模式: 已禁用模型在线下载")

        # PADDLEOCR_HOME: 设置自定义缓存目录
        if self.PADDLEOCR_HOME:
            os.environ['PADDLEOCR_HOME'] = self.PADDLEOCR_HOME
            os.makedirs(self.PADDLEOCR_HOME, exist_ok=True)

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