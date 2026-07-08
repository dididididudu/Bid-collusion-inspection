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
    """检测配置参数"""

    # ── 文本匹配 ──
    TEXT_LOCAL_THRESHOLD: float = 0.85
    SBERT_BASE_THRESHOLD: float = 0.60
    SBERT_SHORT_PARAGRAPH_THRESHOLD: float = 0.80
    SBERT_SHORT_PARAGRAPH_LEN: int = 100

    # ── 评分 ──
    SCORE_WEIGHT_MAX: float = 0.35
    SCORE_WEIGHT_TOP_K: float = 0.30
    SCORE_WEIGHT_MEAN: float = 0.25
    SCORE_TOP_K: int = 8

    # ── 克隆块 ──
    CLONE_BLOCK_MIN_LENGTH: int = 2
    CLONE_BLOCK_MAX_GAP: int = 2
    CLONE_BLOCK_MERGE_MODE: str = "strict"  # strict | loose（方案四）

    # ── 匹配过滤 ──
    MATCH_MIN_COVERAGE: float = 0.15  # 段落匹配最低公共文本覆盖率（方案一）

    # ── 报告展示裁剪（方案二）──
    REPORT_TRIM_THRESHOLD: float = 0.60
    REPORT_TRIM_CONTEXT: int = 30
    REPORT_CLONE_SUMMARY: bool = True     # 克隆块展示紧凑摘要

    # ── 风险等级 ──
    RISK_HIGH_THRESHOLD: int = 70
    RISK_MEDIUM_THRESHOLD: int = 40
    RISK_LOW_THRESHOLD: int = 15

    # ── MinHash / LSH ──
    MINHASH_JACCARD_THRESHOLD: float = 0.3
    MINHASH_NUM_HASHES: int = 128
    MINHASH_NUM_HASHES_PARAGRAPH: int = 32
    PARAGRAPH_LSH_THRESHOLD: float = 0.3
    PARAGRAPH_MATCH_STAGE1_TOP_K: int = 5000
    PARAGRAPH_MATCH_STAGE2_TOP_K: int = 10000
    PARAGRAPH_MIN_JACCARD: float = 0.10
    MAX_CANDIDATE_PAIRS: int = 5000

    # ── 流式 / 存储 ──
    CHUNK_PAGE_SIZE: int = 50
    MAX_CHUNKS_IN_MEMORY: int = 5
    PDF_EXTRACTOR_BACKEND: str = "pymupdf"
    MAX_MEMORY_MB: int = 2048
    STOPWORDS_PATH: Optional[str] = None

    # ── 断点 / 缓存 ──
    ENABLE_CHECKPOINT: bool = True
    CHECKPOINT_DIR: str = "./checkpoints"
    CHECKPOINT_INTERVAL: int = 50
    CACHE_DIR: str = "./cache"
    DISABLE_CACHE: bool = False

    # ── 并行 ──
    PHASE1_WORKERS: int = 4
    PHASE3_WORKERS: int = 4
    DB_BUSY_TIMEOUT: int = 120000

    # ── GPU / SBERT ──
    USE_GPU: bool = False
    SBERT_DEVICE: str = "cpu"
    SBERT_BATCH_SIZE: int = 64
    USE_ONNX: bool = False
    ONNX_MODEL_PATH: Optional[str] = None
    ENABLE_EMBEDDING_CACHE: bool = True
    EMBEDDING_DIM: int = 384
    EMBED_WORKERS: int = 2

    # ── 文档预筛 ──
    DOC_VECTOR_FILTER_ENABLED: bool = True
    DOC_VECTOR_THRESHOLD: float = 0.3
    METADATA_FILTER_ENABLED: bool = True
    TIME_BUCKET_FORMAT: str = "%Y-%m-%dT%H"

    # ── OCR ──
    ENABLE_OCR: bool = True
    OCR_ENGINE: str = "easyocr"
    OCR_LANGUAGES: list = None
    OCR_SAMPLE_STEP: int = 1
    OCR_MIN_CONFIDENCE: float = 0.3
    OCR_WORKERS: int = 4
    OCR_MODEL_DIR: Optional[str] = None
    OCR_OFFLINE_MODE: bool = False
    PADDLEOCR_HOME: Optional[str] = None
    OCR_RETRY_COUNT: int = 3
    ENGINE_INIT_TIMEOUT: int = 120

    # ── 图片 ──
    IMAGE_MIN_SIZE: int = 50
    IMAGE_MAX_SIZE: int = 2000           # 超过此宽/高的图片等比缩放后再算哈希（方案6）
    IMAGE_BOILERPLATE_HASHES: list = field(default_factory=list)  # 已知模板哈希黑名单（方案6）
    BID_BOILERPLATE_FILTER: bool = True
    BID_BOILERPLATE_WEIGHT: float = 0.3

    # ── 报告 ──
    REPORT_MAX_MATCHES_PER_PAIR: int = 10000
    REPORT_INCLUDE_ALL_MATCHES: bool = True
    REPORT_DETAIL_LEVEL: str = "full"
    REPORT_HIGHLIGHT_MAX: int = 500        # difflib 高亮计算的最大匹配对数（减少 Phase 4 耗时）

    # ── 检测维度开关（前端可勾选）──
    # content_similarity 控制文本+图片（重操作），其余轻量维度始终运行不消耗时间
    ENABLED_DIMENSIONS: dict = field(default_factory=lambda: {
        'content_similarity': True,  # 内容相似度（文本+图片）
        'file_id': True,             # 文件码雷同
        'author': True,              # 文档作者雷同
        'editor': True,              # 编辑经办人雷同
        'contact': True,             # 联系人雷同
        'company_name': True,        # 公司名称异常
        'credit_code': True,         # 信用代码雷同
        'member_id': False,          # 会员号雷同（默认关闭）
    })

    def __post_init__(self):
        if self.CHUNK_PAGE_SIZE < 10:
            raise ValueError("CHUNK_PAGE_SIZE >= 10")
        if self.PDF_EXTRACTOR_BACKEND not in ("pymupdf", "pdfplumber"):
            raise ValueError("PDF_EXTRACTOR_BACKEND must be pymupdf or pdfplumber")
        if self.SBERT_DEVICE not in ("cpu", "cuda", "mps", "auto"):
            raise ValueError("SBERT_DEVICE must be cpu, cuda, mps, or auto")
        if self.SBERT_DEVICE == "auto" or self.USE_GPU:
            self._auto_detect_device()
        self._apply_env_overrides()

        # Worker 数合理性：不超过 CPU 核心数
        import os as _os
        _cpu = _os.cpu_count() or 4
        if self.PHASE1_WORKERS > _cpu:
            self.PHASE1_WORKERS = max(1, _cpu // 2)
            logger.info(f"PHASE1_WORKERS 修正为 {self.PHASE1_WORKERS} (CPU={_cpu})")
        if self.PHASE3_WORKERS > _cpu:
            self.PHASE3_WORKERS = max(1, _cpu // 2)
            logger.info(f"PHASE3_WORKERS 修正为 {self.PHASE3_WORKERS} (CPU={_cpu})")
        if self.OCR_WORKERS < 1 or self.OCR_WORKERS > _cpu:
            self.OCR_WORKERS = max(1, min(_cpu, 4))
            logger.info(f"OCR_WORKERS 修正为 {self.OCR_WORKERS}")

    def _auto_detect_device(self):
        try:
            import torch
            if torch.cuda.is_available():
                self.SBERT_DEVICE = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.SBERT_DEVICE = "mps"
        except ImportError:
            pass

    def _apply_env_overrides(self):
        if self.PADDLEOCR_HOME is None:
            env_home = os.environ.get('PADDLEOCR_HOME', '')
            if env_home:
                self.PADDLEOCR_HOME = env_home
        if self.OCR_MODEL_DIR is None:
            env_model = os.environ.get('OCR_MODEL_DIR', '')
            if env_model:
                self.OCR_MODEL_DIR = env_model
        if not self.OCR_OFFLINE_MODE and os.environ.get('OCR_OFFLINE', '') == '1':
            self.OCR_OFFLINE_MODE = True
        if self.OCR_OFFLINE_MODE:
            os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
            os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
        if self.PADDLEOCR_HOME:
            os.environ['PADDLEOCR_HOME'] = self.PADDLEOCR_HOME
            os.makedirs(self.PADDLEOCR_HOME, exist_ok=True)

    @classmethod
    def from_json(cls, config_path: str) -> 'DetectionConfig':
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            valid_fields = {k: v for k, v in config_dict.items()
                           if k in cls.__dataclass_fields__}
            return cls(**valid_fields)
        except Exception as e:
            raise ValueError(f"failed to load config: {e}")

    def to_json(self, config_path: str) -> None:
        config_dict = {k: v for k, v in self.__dict__.items()
                       if not k.startswith('_')}
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)


def load_config(config_path: Optional[str] = None) -> DetectionConfig:
    if config_path:
        return DetectionConfig.from_json(config_path)
    return DetectionConfig()
