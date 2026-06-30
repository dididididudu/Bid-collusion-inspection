"""
核心数据结构定义
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime


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


@dataclass
class BidFeature:
    """单文档特征"""
    doc_id: str
    filename: str
    file_size: int
    text_content: str
    text_length: int
    text_simhash: str  # 64位字符串
    paragraph_hashes: List[str] = field(default_factory=list)
    metadata: MetadataFeature = field(default_factory=MetadataFeature)
    quotes: List[float] = field(default_factory=list)
    quote_signature: QuoteSignature = field(default_factory=QuoteSignature)
    image_hashes: List[str] = field(default_factory=list)
    extracted_at: str = field(default_factory=lambda: datetime.now().isoformat())
    is_scanned: bool = False  # 是否为扫描版


@dataclass
class ParagraphMatch:
    """段落匹配详情"""
    similarity: float = 0.0
    paragraph_a: str = ""
    paragraph_b: str = ""
    paragraph_a_index: int = 0
    paragraph_b_index: int = 0
    detection_method: str = ""  # SequenceMatcher / SBERT
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
    """图片重复证据"""
    common_image_count: int = 0
    common_image_hashes: List[str] = field(default_factory=list)


@dataclass
class EvidenceChain:
    """证据链"""
    text_evidence: TextEvidence = field(default_factory=TextEvidence)
    metadata_evidence: MetadataEvidence = field(default_factory=MetadataEvidence)
    image_evidence: ImageEvidence = field(default_factory=ImageEvidence)


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
