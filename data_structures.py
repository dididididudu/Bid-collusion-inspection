"""
核心数据结构定义
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from datetime import datetime


# ============================================================
# 基础特征数据结构
# ============================================================

@dataclass
class MetadataFeature:
    """元数据特征"""
    author: str = ""
    creator: str = ""
    producer: str = ""
    created_time: str = ""  # ISO 8601 格式
    modified_time: str = ""
    software_fingerprint: str = ""  # creator + producer 拼接归一化
    time_bucket: str = ""  # 创建时间按小时取整


@dataclass
class QuoteSignature:
    """报价统计特征"""
    count: int = 0
    values: List[float] = field(default_factory=list)
    tail_distribution: Dict[str, int] = field(default_factory=dict)
    integer_ratio: float = 0.0
    mean: float = 0.0
    std: float = 0.0


# ============================================================
# 文档特征 (轻量化版本)
# ============================================================

@dataclass
class BidFeature:
    """单文档特征（轻量化描述符）

    流式模式下：
      - text_content, paragraphs, paragraph_hashes 存储在 SQLite 中，此处为空
      - 新增 page_count, doc_minhash, chunk_count 用于快速筛选

    非流式模式下（向后兼容）：
      - 所有字段行为与之前一致
    """
    doc_id: str
    filename: str
    file_size: int
    file_path: str = ""  # 原始 PDF 路径（供图片导出）

    # === 文本内容 (旧版：内存存储；流式：为空，从 SQLite 惰性加载) ===
    text_content: str = ""
    text_length: int = 0
    text_simhash: str = ""  # 64位十六进制字符串
    paragraphs: List[str] = field(default_factory=list)
    paragraph_hashes: List[str] = field(default_factory=list)

    # === 元数据与提取信息 ===
    metadata: MetadataFeature = field(default_factory=MetadataFeature)
    quotes: List[float] = field(default_factory=list)
    quote_signature: QuoteSignature = field(default_factory=QuoteSignature)
    image_hashes: List[str] = field(default_factory=list)
    extracted_at: str = field(default_factory=lambda: datetime.now().isoformat())
    is_scanned: bool = False  # 是否为扫描版

    # === 新增：流式模式字段 ===
    page_count: int = 0  # PDF 总页数
    doc_minhash: Optional[List[int]] = None  # 聚合所有段落的 MinHash 签名（128 维）
    chunk_count: int = 0  # 文本块数量


# ============================================================
# 流式处理专用数据结构
# ============================================================

@dataclass
class ChunkMetadata:
    """文本块轻量描述符（仅元数据，不含文本内容）

    文本内容存储在 SQLite 中，按需通过 storage_key 加载。
    """
    doc_id: str
    chunk_index: int  # 块序号（从 0 开始）
    start_page: int  # 起始页码（0-based）
    end_page: int  # 结束页码（0-based，含）
    text_length: int  # 字符数
    simhash: str = ""  # 该块的 SimHash
    paragraph_count: int = 0  # 块内段落数


@dataclass
class ChunkResult:
    """单个文本块处理的结果"""
    doc_id: str
    chunk_index: int
    start_page: int
    end_page: int
    text: str  # 块内完整文本
    paragraphs: List[str] = field(default_factory=list)
    paragraph_hashes: List[str] = field(default_factory=list)
    simhash: str = ""
    quotes: List[float] = field(default_factory=list)
    image_hashes: List[str] = field(default_factory=list)


@dataclass
class CheckpointState:
    """可序列化的管道进度状态

    用于断点续传，每个阶段结束时写入检查点文件。
    Phase 3（分析阶段）每 N 对增量写入。
    """
    phase: int = 0  # 当前完成的阶段编号 (0-5)
    completed_pairs: int = 0  # 已完成分析的对数
    total_pairs: int = 0  # 候选总对数
    processed_files: Set[str] = field(default_factory=set)  # Phase 1 已处理的文件
    completed_pair_ids: Set[str] = field(default_factory=set)  # Phase 3 已完成的对
    start_time: str = ""  # 管道启动时间
    config_hash: str = ""  # 配置哈希（检测配置漂移）
    input_hash: str = ""  # 输入文件夹内容哈希（检测文件变化）
    version: int = 4  # 检查点格式版本（v4: 新增 Phase 1.5）


# ============================================================
# 段落匹配与证据
# ============================================================

@dataclass
class ParagraphMatch:
    """段落匹配详情"""
    similarity: float = 0.0
    paragraph_a: str = ""
    paragraph_b: str = ""
    paragraph_a_index: int = 0
    paragraph_b_index: int = 0
    detection_method: str = ""  # SequenceMatcher / SBERT / MinHash-Jaccard
    is_continuous_clone: bool = False  # 是否属于连续克隆块
    continuous_clone_group_id: str = ""  # 连续克隆块组ID
    highlighted_text_a: str = ""  # 标记重复部分后的文本A
    highlighted_text_b: str = ""  # 标记重复部分后的文本B


@dataclass
class TextEvidence:
    """文本雷同证据"""
    local_similarity: float = 0.0
    common_paragraphs: List[str] = field(default_factory=list)
    paragraph_matches: List[Dict] = field(default_factory=list)
    continuous_clone_blocks: List[Dict] = field(default_factory=list)  # 连续克隆块列表
    detection_summary: Dict = field(default_factory=dict)  # 检测方法汇总


@dataclass
class MetadataEvidence:
    """元数据关联证据"""
    matched_fields: List[str] = field(default_factory=list)
    matched_values: Dict[str, str] = field(default_factory=dict)
    same_time_bucket: bool = False


@dataclass
class ImageEvidence:
    """图片重复证据（增强版 — 四层检测）"""
    # 哈希层
    common_image_count: int = 0
    common_image_hashes: List[str] = field(default_factory=list)
    exact_image_count: int = 0           # L1: 完全相同的图片对
    near_identical_count: int = 0        # L1: 几乎相同的图片对
    similar_image_count: int = 0         # L1: 相似的图片对

    # OCR 层
    ps_suspicious: bool = False          # L2: PS 嫌疑
    ps_suspicious_count: int = 0
    shared_typos: List[str] = field(default_factory=list)  # L3: 相同错别字
    shared_typo_count: int = 0
    text_identical_count: int = 0        # L4: 图片文字完全相同
    text_similar_count: int = 0          # L4: 图片文字高度相似

    # OCR 原始结果
    ocr_results_a: List[Dict] = field(default_factory=list)
    ocr_results_b: List[Dict] = field(default_factory=list)

    # 匹配图片路径（供 HTML 报告嵌入展示，JSON 中仅存储路径）
    matched_image_paths: Dict = field(default_factory=dict)

    # 综合
    image_risk_score: int = 0            # 图片维度风险分 (0-30)
    image_risk_factors: List[str] = field(default_factory=list)


@dataclass
class EvidenceChain:
    """证据链"""
    text_evidence: TextEvidence = field(default_factory=TextEvidence)
    metadata_evidence: MetadataEvidence = field(default_factory=MetadataEvidence)
    image_evidence: ImageEvidence = field(default_factory=ImageEvidence)


# ============================================================
# 检测结果
# ============================================================

@dataclass
class PairwiseResult:
    """文档对相似度结果"""
    pair_id: str
    doc_a_id: str
    doc_b_id: str
    similarity_scores: Dict[str, float] = field(default_factory=dict)
    risk_level: str = "NONE"  # "NONE" | "LOW" | "MEDIUM" | "HIGH"
    risk_score: int = 0  # 0-100 整数
    risk_factors: List[str] = field(default_factory=list)
    evidence: EvidenceChain = field(default_factory=EvidenceChain)


@dataclass
class Cluster:
    """风险聚类"""
    cluster_id: str
    doc_ids: List[str]
    cluster_type: str  # "TEXT_CLONE" | "META_GROUP" | "QUOTE_RING"
    confidence: float = 0.0


@dataclass
class FileProfile:
    """单文档风险画像"""
    doc_id: str
    filename: str
    related_suspicious_count: int = 0
    max_risk_level: str = "NONE"
    related_clusters: List[str] = field(default_factory=list)


@dataclass
class GlobalReport:
    """全局检测报告"""
    report_id: str
    generated_at: str
    total_files: int
    total_pairs: int
    candidate_pairs: int
    suspicious_pairs: int
    high_risk_pairs: int
    risk_clusters: List[Cluster] = field(default_factory=list)
    pairwise_results: List[PairwiseResult] = field(default_factory=list)
    file_profiles: Dict[str, FileProfile] = field(default_factory=dict)
    error_log: List[str] = field(default_factory=list)
    single_doc_risks: Dict[str, str] = field(default_factory=dict)  # 单文档风险等级
