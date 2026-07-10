"""
配置参数系统 — 支持环境变量覆盖，便于服务器部署
"""
import os
from typing import Optional
from dataclasses import dataclass, field
import json
import logging

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    from pathlib import Path
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"已加载配置文件: {env_path}")
except ImportError:
    logger.debug("python-dotenv 未安装，跳过 .env 加载")


@dataclass
class DetectionConfig:
    """检测配置参数"""

    # ── 文本匹配 ──
    TEXT_LOCAL_THRESHOLD: float = 0.85
    SBERT_BASE_THRESHOLD: float = 0.75
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
    MINHASH_LSH_THRESHOLD: float = 0.3          # datasketch MinHashLSH 阈值
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
    PHASE1_WORKERS: int = 2         # CPU 场景不宜过高，文本提取 IO 密集 2 即可
    PHASE3_WORKERS: int = 2         # CPU 场景不超过核心数/2
    DB_BUSY_TIMEOUT: int = 120000

    # ── GPU / SBERT ──
    USE_GPU: bool = False            # CPU 服务器不使用 GPU
    SBERT_DEVICE: str = "cpu"        # 强制 CPU，不自动检测 CUDA
    SBERT_BATCH_SIZE: int = 64       # CPU 场景批处理（64 充分利用 SIMD）
    USE_ONNX: bool = False
    ONNX_MODEL_PATH: Optional[str] = None
    ENABLE_EMBEDDING_CACHE: bool = True
    EMBEDDING_DIM: int = 384
    EMBED_WORKERS: int = 1           # CPU 上单进程即可
    GPU_MANAGER_ENABLED: bool = False  # GPU Manager 开关（CPU服务器禁用）
    OCR_BATCH_SIZE: int = 4            # OCR 批处理大小（CPU场景减小）
    OCR_BATCH_TIMEOUT: float = 1.0     # 批处理聚合超时（CPU场景缩短）

    # ── 文档预筛 ──
    DOC_VECTOR_FILTER_ENABLED: bool = True
    DOC_VECTOR_THRESHOLD: float = 0.15
    METADATA_FILTER_ENABLED: bool = True
    TIME_BUCKET_FORMAT: str = "%Y-%m-%dT%H"

    # ── OCR ──
    ENABLE_OCR: bool = False
    OCR_ENGINE: str = "paddleocr"      # PaddleOCR 中文识别快，CPU 友好
    OCR_LANGUAGES: list = None
    OCR_SAMPLE_STEP: int = 2          # CPU 场景隔页采样，耗时减半
    OCR_MIN_CONFIDENCE: float = 0.3
    OCR_WORKERS: int = 1              # CPU 场景单进程 OCR，多进程争 CPU 反而慢
    OCR_MODEL_DIR: Optional[str] = None
    OCR_OFFLINE_MODE: bool = False
    PADDLEOCR_HOME: Optional[str] = None
    OCR_RETRY_COUNT: int = 3
    ENGINE_INIT_TIMEOUT: int = 120

    # ── 技术标/商务标维度过滤（管线仅处理指定维度）──
    # "all" = 全部页面, "technical" = 仅技术标页, "commercial" = 仅商务标页
    ANALYSIS_DIMENSION: str = "all"

    # ── 图片 ──
    IMAGE_MIN_SIZE: int = 50
    IMAGE_MAX_SIZE: int = 2000           # 超过此宽/高的图片等比缩放后再算哈希（方案6）
    IMAGE_BOILERPLATE_HASHES: list = field(default_factory=list)  # 已知模板哈希黑名单（方案6）
    BID_BOILERPLATE_FILTER: bool = True
    BID_BOILERPLATE_WEIGHT: float = 0.3

    # ── 目录排除（方案六：目录结构雷同不参与查重）──
    TOC_FILTER_ENABLED: bool = True      # 启用目录段落过滤
    TOC_PAGE_RATIO: float = 0.2          # 目录检测页面范围（前 20% 页面）

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
        env_ocr_engine = os.environ.get('OCR_ENGINE', '').lower()
        if env_ocr_engine in ('rapidocr', 'paddleocr', 'easyocr'):
            self.OCR_ENGINE = env_ocr_engine
            logger.info(f"OCR_ENGINE 从环境变量加载: {self.OCR_ENGINE}")
        
        env_use_gpu = os.environ.get('USE_GPU', '').lower()
        if env_use_gpu in ('true', '1', 'yes'):
            self.USE_GPU = True
            logger.info("USE_GPU 从环境变量加载: True")
        elif env_use_gpu in ('false', '0', 'no'):
            self.USE_GPU = False
            logger.info("USE_GPU 从环境变量加载: False")
        
        env_sbert_device = os.environ.get('SBERT_DEVICE', '').lower()
        if env_sbert_device in ('cpu', 'cuda', 'mps'):
            self.SBERT_DEVICE = env_sbert_device
            logger.info(f"SBERT_DEVICE 从环境变量加载: {self.SBERT_DEVICE}")
        
        env_ocr_workers = os.environ.get('OCR_WORKERS', '')
        if env_ocr_workers.isdigit():
            self.OCR_WORKERS = int(env_ocr_workers)
            logger.info(f"OCR_WORKERS 从环境变量加载: {self.OCR_WORKERS}")
        
        env_ocr_batch_size = os.environ.get('OCR_BATCH_SIZE', '')
        if env_ocr_batch_size.isdigit():
            self.OCR_BATCH_SIZE = int(env_ocr_batch_size)
            logger.info(f"OCR_BATCH_SIZE 从环境变量加载: {self.OCR_BATCH_SIZE}")
        
        env_phase1 = os.environ.get('PHASE1_WORKERS', '')
        if env_phase1.isdigit():
            self.PHASE1_WORKERS = int(env_phase1)
            logger.info(f"PHASE1_WORKERS 从环境变量加载: {self.PHASE1_WORKERS}")
        
        env_phase3 = os.environ.get('PHASE3_WORKERS', '')
        if env_phase3.isdigit():
            self.PHASE3_WORKERS = int(env_phase3)
            logger.info(f"PHASE3_WORKERS 从环境变量加载: {self.PHASE3_WORKERS}")
        
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
