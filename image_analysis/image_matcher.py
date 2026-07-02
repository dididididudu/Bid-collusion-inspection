"""
图片对比引擎 — 四层检测

L1: 完全相同/近似图片（pHash/dHash 汉明距离）
L2: PS 嫌疑（OCR 文字相同但图片哈希不同）
L3: 相同错别字（OCR 文本中共有异常低频词）
L4: 图片文字完全相同（不同图片中 OCR 文字高度一致）
"""

import logging
from typing import List
from dataclasses import dataclass, field

import numpy as np

from image_analysis.image_hasher import ImageHasher, HashMatchResult
from image_analysis.image_ocr import OCRResult

logger = logging.getLogger(__name__)


@dataclass
class ImageMatchResult:
    """单对文档的图片对比综合结果"""
    # L1: 哈希匹配
    exact_image_count: int = 0          # 完全相同的图片对
    near_identical_count: int = 0       # 几乎相同的图片对
    similar_image_count: int = 0        # 相似的图片对
    hash_matches: List[HashMatchResult] = field(default_factory=list)

    # L2: PS 嫌疑
    ps_suspicious: bool = False
    ps_suspicious_count: int = 0        # PS 嫌疑的图片对数量

    # L3: 相同错别字
    shared_typos: List[str] = field(default_factory=list)
    shared_typo_count: int = 0

    # L4: 图片文字完全相同
    text_identical_count: int = 0       # OCR 文字完全相同的图片对
    text_similar_count: int = 0         # OCR 文字高度相似的图片对

    # 综合
    image_risk_score: int = 0           # 0-30 图片维度风险分
    image_risk_factors: List[str] = field(default_factory=list)


class ImageMatcher:
    """图片对比引擎 — 四层检测"""

    # L2 阈值
    OCR_SIMILARITY_THRESHOLD = 0.85     # OCR 文字相似度 ≥ 此值视为"文字相同"
    HASH_DISSIMILARITY_THRESHOLD = 10   # 汉明距离 ≥ 此值视为"图片不同"

    # L3 阈值
    TYPO_MIN_COUNT = 2                  # 最少共同异常词数
    TYPO_MIN_LENGTH = 2                 # 异常词最小长度（过滤单字）

    # L4 阈值
    TEXT_IDENTICAL_THRESHOLD = 0.95     # 文字完全相同
    TEXT_SIMILAR_THRESHOLD = 0.85       # 文字高度相似
    TEXT_MIN_LENGTH = 10                # 最小文字长度

    def __init__(self):
        self.hasher = ImageHasher()

    def analyze(
        self,
        hashes_a: List[str],
        hashes_b: List[str],
        ocr_results_a: List[OCRResult] = None,
        ocr_results_b: List[OCRResult] = None,
    ) -> ImageMatchResult:
        """执行四层图片对比分析

        Args:
            hashes_a: 文档A的图片哈希列表
            hashes_b: 文档B的图片哈希列表
            ocr_results_a: 文档A的OCR结果列表（可选）
            ocr_results_b: 文档B的OCR结果列表（可选）

        Returns:
            ImageMatchResult 综合匹配结果
        """
        result = ImageMatchResult()

        # === L1: 图片哈希层 ===
        self._analyze_hash_layer(hashes_a, hashes_b, result)

        # === L2-L4: OCR 文字层（需 OCR 数据） ===
        if ocr_results_a and ocr_results_b:
            self._analyze_ocr_layers(
                ocr_results_a, ocr_results_b, result
            )

        # === 综合评分 ===
        self._compute_image_risk_score(result)

        return result

    # ================================================================
    # L1: 图片哈希层
    # ================================================================

    def _analyze_hash_layer(
        self,
        hashes_a: List[str],
        hashes_b: List[str],
        result: ImageMatchResult,
    ):
        """L1: 基于 pHash/dHash 汉明距离找相同/相似图片"""
        matches = self.hasher.match_hashes(hashes_a, hashes_b)
        result.hash_matches = matches

        for m in matches:
            if m.is_exact:
                result.exact_image_count += 1
            elif m.is_similar:
                result.near_identical_count += 1
            else:
                result.similar_image_count += 1

        if result.exact_image_count > 0:
            result.image_risk_factors.append(
                f"发现 {result.exact_image_count} 对完全相同图片"
            )
        if result.near_identical_count > 0:
            result.image_risk_factors.append(
                f"发现 {result.near_identical_count} 对高度相似图片"
            )

    # ================================================================
    # L2-L4: OCR 文字层
    # ================================================================

    def _analyze_ocr_layers(
        self,
        ocr_a: List[OCRResult],
        ocr_b: List[OCRResult],
        result: ImageMatchResult,
    ):
        """L2-L4: 基于 OCR 文字的检测"""
        # 收集所有 OCR 的词
        all_words_a = []
        all_words_b = []
        for r in ocr_a:
            all_words_a.extend(r.words)
        for r in ocr_b:
            all_words_b.extend(r.words)

        # --- L4: 图片文字完全相同 ---
        self._detect_identical_text(ocr_a, ocr_b, result)

        # --- L2: PS 嫌疑（文字相同但图片不同） ---
        self._detect_ps_suspicious(ocr_a, ocr_b, result)

        # --- L3: 相同错别字 ---
        self._detect_shared_typos(all_words_a, all_words_b, result)

    def _detect_identical_text(
        self,
        ocr_a: List[OCRResult],
        ocr_b: List[OCRResult],
        result: ImageMatchResult,
    ):
        """L4: 检测不同图片中包含完全相同的文字"""
        for ra in ocr_a:
            if len(ra.text) < self.TEXT_MIN_LENGTH:
                continue
            for rb in ocr_b:
                if len(rb.text) < self.TEXT_MIN_LENGTH:
                    continue
                sim = self._text_similarity(ra.text, rb.text)

                if sim >= self.TEXT_IDENTICAL_THRESHOLD:
                    result.text_identical_count += 1
                elif sim >= self.TEXT_SIMILAR_THRESHOLD:
                    result.text_similar_count += 1

        if result.text_identical_count > 0:
            result.image_risk_factors.append(
                f"发现 {result.text_identical_count} 对图片文字完全相同"
            )
        if result.text_similar_count > 0:
            result.image_risk_factors.append(
                f"发现 {result.text_similar_count} 对图片文字高度相似"
            )

    def _detect_ps_suspicious(
        self,
        ocr_a: List[OCRResult],
        ocr_b: List[OCRResult],
        result: ImageMatchResult,
    ):
        """L2: PS 嫌疑检测（改进版：真实图片哈希对比）

        逻辑：OCR 文字高度相似，但图片哈希差异大 → PS 嫌疑。
        即：文字内容没变，但图片被修改过（如改背景、换logo）。

        优先使用图片哈希汉明距离判定；无 image_hash 时
        回退到 OCR 置信度差异作为辅助信号。
        """
        for ra in ocr_a:
            if len(ra.text) < self.TEXT_MIN_LENGTH:
                continue
            for rb in ocr_b:
                if len(rb.text) < self.TEXT_MIN_LENGTH:
                    continue

                ocr_sim = self._text_similarity(ra.text, rb.text)
                if ocr_sim < self.OCR_SIMILARITY_THRESHOLD:
                    continue

                # OCR 文字高度相似 → 检查图片哈希差异
                if ra.image_hash and rb.image_hash:
                    # 主路径：真实图片哈希比对
                    hash_dist = ImageHasher.hamming_distance(
                        ra.image_hash, rb.image_hash
                    )
                    if hash_dist >= self.HASH_DISSIMILARITY_THRESHOLD:
                        result.ps_suspicious_count += 1
                else:
                    # 回退路径：置信度差异（弱信号）
                    conf_diff = abs(ra.confidence - rb.confidence)
                    if conf_diff > 0.15:
                        result.ps_suspicious_count += 1

        if result.ps_suspicious_count > 0:
            result.ps_suspicious = True
            result.image_risk_factors.append(
                f"⚠ PS嫌疑: {result.ps_suspicious_count} 对图片文字相同但图片特征不同"
            )

    # 常见中文 OCR 易混淆字符对（形近字）
    _OCR_CONFUSION_PAIRS = {
        ('日', '曰'), ('己', '已'), ('末', '未'), ('土', '士'),
        ('干', '千'), ('人', '入'), ('天', '夭'), ('王', '玉'),
        ('戍', '戌'), ('概', '慨'), ('拨', '拔'), ('析', '折'),
        ('准', '淮'), ('贷', '货'), ('辨', '辩'), ('概', '慨'),
        ('燥', '躁'), ('侯', '候'), ('拦', '栏'), ('历', '厉'),
    }

    def _detect_shared_typos(
        self,
        words_a: List[str],
        words_b: List[str],
        result: ImageMatchResult,
    ):
        """L3: 相同错别字检测

        核心思路：OCR 错误产生的"假词"（非合法中文词）不会偶然出现在
        两个独立文档中。如果两份文档的 OCR 结果包含相同的非合法词，
        说明它们来自同一源图片（或同一 OCR 引擎的错误模式）。

        检测逻辑:
        1. 找出两文档中的"非合法词"（不在词库中的词）
        2. 两文档共有的非合法词 = 相同 OCR 错字 → 高嫌疑
        3. 同时检测 OCR 易混淆字符对产生的错字
        """
        if not words_a or not words_b:
            return

        # 加载常用中文词库（jieba 内置词典 + 扩展）
        common_words = self._get_common_word_set()

        # 找出各文档的非合法词（可能是 OCR 错误产生的）
        invalid_a = {w for w in words_a
                     if len(w) >= self.TYPO_MIN_LENGTH
                     and w not in common_words
                     and not w.isascii()}  # 排除纯英文
        invalid_b = {w for w in words_b
                     if len(w) >= self.TYPO_MIN_LENGTH
                     and w not in common_words
                     and not w.isascii()}

        # 共同非合法词 = 相同错别字（强证据）
        shared_invalid = invalid_a & invalid_b

        # 额外检测：OCR 易混淆字符产生的错字
        # 如果两文档各有一个词，仅差一个易混淆字符，也视为相同错字
        confusion_typos = set()
        for wa in invalid_a:
            for wb in invalid_b:
                if len(wa) == len(wb) and wa != wb:
                    # 检查是否为单字符替换
                    diffs = [(i, wa[i], wb[i]) for i in range(len(wa)) if wa[i] != wb[i]]
                    if len(diffs) == 1:
                        i, ca, cb = diffs[0]
                        if (ca, cb) in self._OCR_CONFUSION_PAIRS or \
                           (cb, ca) in self._OCR_CONFUSION_PAIRS:
                            confusion_typos.add(f"{wa}↔{wb}")

        # 合并结果
        all_typos = sorted(shared_invalid)[:15]
        if confusion_typos:
            all_typos.extend(sorted(confusion_typos)[:5])

        result.shared_typos = all_typos
        result.shared_typo_count = len(shared_invalid) + len(confusion_typos)

        if result.shared_typo_count >= self.TYPO_MIN_COUNT:
            detail = ', '.join(all_typos[:5])
            if len(all_typos) > 5:
                detail += '...'
            result.image_risk_factors.append(
                f"⚠ 相同错别字: {result.shared_typo_count} 个 ({detail})"
            )

    @staticmethod
    def _get_common_word_set() -> set:
        """获取常用中文词集合（缓存）"""
        if not hasattr(ImageMatcher, '_COMMON_WORDS_CACHE'):
            import jieba
            # jieba 内置词典中的词都是合法中文词
            common = set()
            try:
                # 读取 jieba 词典
                dict_path = jieba.get_dict_file()
                with open(dict_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            word = parts[0]
                            if len(word) >= 2:
                                common.add(word)
            except Exception:
                pass
            # 常见技术术语补充
            common.update({
                '运维', '管控', '模型', '训练', '系统', '平台', '数据',
                '算法', '架构', '部署', '监控', '预警', '检测', '分析',
                '接口', '模块', '配置', '参数', '日志', '缓存', '队列',
                '集群', '容器', '编排', '调度', '负载', '均衡', '代理',
            })
            ImageMatcher._COMMON_WORDS_CACHE = frozenset(common)
        return ImageMatcher._COMMON_WORDS_CACHE

    # ================================================================
    # 文本相似度（复用 jieba Jaccard，不依赖 SBERT）
    # ================================================================

    @staticmethod
    def _text_similarity(text_a: str, text_b: str) -> float:
        """计算两段 OCR 文本的相似度（词级 Jaccard）"""
        import jieba

        if not text_a or not text_b:
            return 0.0

        words_a = {w for w in jieba.cut(text_a) if len(w) > 1}
        words_b = {w for w in jieba.cut(text_b) if len(w) > 1}

        if not words_a or not words_b:
            return 0.0

        intersection = words_a & words_b
        union = words_a | words_b

        return len(intersection) / len(union) if union else 0.0

    # ================================================================
    # 综合评分
    # ================================================================

    def _compute_image_risk_score(self, result: ImageMatchResult):
        """根据四层检测结果计算图片风险分（0-30）"""
        score = 0

        # L1: 相同图片（最强证据）
        score += min(15, result.exact_image_count * 5)
        score += min(8, result.near_identical_count * 2)

        # L2: PS 嫌疑
        if result.ps_suspicious:
            score += 10

        # L3: 相同错别字
        if result.shared_typo_count >= 2:
            score += min(8, result.shared_typo_count * 2)

        # L4: 图片文字完全相同
        score += min(5, result.text_identical_count * 2)

        result.image_risk_score = min(30, score)
