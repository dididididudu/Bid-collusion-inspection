"""
图片对比引擎 — 四层检测（精度优先版）

L1: 多哈希共识图片匹配（pHash + dHash + 长宽比联合判决）
L2: SBERT 语义文本比对（优先） / jieba Jaccard（回退）
L3: 相同错别字 + 相同稀有词（共享特有词汇）
L4: PS 嫌疑（多哈希非文字区域比对 + 尺寸 + 颜色分布）
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field

import numpy as np

from image_analysis.image_hasher import (
    ImageHasher, HashMatchResult, ImageSignature, ImageMatchVerdict,
)
from image_analysis.image_ocr import OCRResult

logger = logging.getLogger(__name__)


@dataclass
class ImageMatchResult:
    """单对文档的图片对比综合结果"""
    # L1: 多哈希共识匹配
    exact_image_count: int = 0          # 完全相同的图片对
    near_identical_count: int = 0       # 几乎相同的图片对
    similar_image_count: int = 0        # 相似的图片对
    hash_matches: List[HashMatchResult] = field(default_factory=list)
    image_verdicts: List[ImageMatchVerdict] = field(default_factory=list)

    # L2: 图片文字比对
    text_identical_count: int = 0       # OCR 文字完全相同的图片对
    text_similar_count: int = 0         # OCR 文字高度相似的图片对
    text_matches: List[dict] = field(default_factory=list)  # 详细匹配记录

    # L3: 相同错别字 / 共有特有词
    shared_typos: List[str] = field(default_factory=list)
    shared_typo_count: int = 0
    shared_rare_words: List[str] = field(default_factory=list)  # 共有稀有词

    # L4: PS 嫌疑（文字不同但非文字区域相同）
    ps_suspicious: bool = False
    ps_suspicious_count: int = 0        # PS 嫌疑的图片对数量
    ps_details: List[dict] = field(default_factory=list)

    # 综合
    image_risk_score: int = 0           # 0-30 图片维度风险分
    image_risk_factors: List[str] = field(default_factory=list)


class ImageMatcher:
    """图片对比引擎 — 四层检测（精度优先版）"""

    # L1 阈值
    HASH_EXACT_DIST = 0
    HASH_NEAR_IDENTICAL_DIST = 5
    HASH_SIMILAR_DIST = 10

    # L2 阈值
    TEXT_IDENTICAL_THRESHOLD = 0.92     # SBERT 余弦 ≥ 此值 → 完全相同
    TEXT_SIMILAR_THRESHOLD = 0.80       # SBERT 余弦 ≥ 此值 → 高度相似
    TEXT_MIN_LENGTH = 10                # 最小文字长度

    # L3 阈值
    TYPO_MIN_COUNT = 2                  # 最少共同异常词数
    TYPO_MIN_LENGTH = 2                 # 异常词最小长度

    # L4 阈值
    NON_TEXT_HASH_SIMILAR = 10          # non_text 汉明距离 ≤ 此值视为背景相似
    PS_TEXT_SIMILARITY_MAX = 0.55       # 文字相似度 ≤ 此值才触发 PS 检测
    PS_DIMENSION_TOLERANCE = 0.20       # 尺寸差异 ≤ 20%

    def __init__(self, semantic_matcher=None):
        self.hasher = ImageHasher()
        self.semantic_matcher = semantic_matcher  # 可选 SBERT 语义匹配器

    def analyze(
        self,
        hashes_a: List[str],
        hashes_b: List[str],
        ocr_results_a: List[OCRResult] = None,
        ocr_results_b: List[OCRResult] = None,
        sigs_a: List[ImageSignature] = None,   # 预解析的签名列表
        sigs_b: List[ImageSignature] = None,
        max_matches: int = 0,
        boilerplate_hashes: set = None,
    ) -> ImageMatchResult:
        """执行四层图片对比分析

        Args:
            hashes_a/b: 原始哈希字符串列表（用于旧版兼容）
            ocr_results_a/b: OCR 结果列表
            sigs_a/b: 预解析的图片签名（优先使用）
            max_matches: >0 时找到此数量匹配后提前终止（方案2b）
            boilerplate_hashes: 已知模板哈希黑名单，跳过这些图片（方案6）
        """
        result = ImageMatchResult()

        # === L1: 图片哈希层（多哈希共识） ===
        # 优先使用预签名，否则从哈希字符串解析
        if sigs_a and sigs_b:
            sigs_a_used = sigs_a
            sigs_b_used = sigs_b
        else:
            sigs_a_used = ImageHasher.parse_hashes_into_signatures(hashes_a)
            sigs_b_used = ImageHasher.parse_hashes_into_signatures(hashes_b)

        # 合并 OCR 结果的尺寸+缩略图信息到签名
        if ocr_results_a:
            wm_a, hm_a = ImageHasher.filter_ocr_hashes(ocr_results_a)
            for sig in sigs_a_used:
                if sig.source_id in wm_a:
                    sig.width = wm_a[sig.source_id]
                if sig.source_id in hm_a:
                    sig.height = hm_a[sig.source_id]
            # 从 OCR 结果映射缩略图
            for r in ocr_results_a:
                if hasattr(r, 'image_hash'):
                    h = r.image_hash
                    thumb = r.thumbnail
                else:
                    h = r.get('image_hash', '')
                    thumb = r.get('thumbnail', b'')
                if h and thumb:
                    _, _, htype = ImageHasher.parse_hash_string(h)
                    for sig in sigs_a_used:
                        if (htype == 'p' and sig.phash and sig.phash in h) or \
                           (htype == 'd' and sig.dhash and sig.dhash in h):
                            sig.thumbnail = thumb
                            break
        if ocr_results_b:
            wm_b, hm_b = ImageHasher.filter_ocr_hashes(ocr_results_b)
            for sig in sigs_b_used:
                if sig.source_id in wm_b:
                    sig.width = wm_b[sig.source_id]
                if sig.source_id in hm_b:
                    sig.height = hm_b[sig.source_id]
            for r in ocr_results_b:
                if hasattr(r, 'image_hash'):
                    h = r.image_hash
                    thumb = r.thumbnail
                else:
                    h = r.get('image_hash', '')
                    thumb = r.get('thumbnail', b'')
                if h and thumb:
                    _, _, htype = ImageHasher.parse_hash_string(h)
                    for sig in sigs_b_used:
                        if (htype == 'p' and sig.phash and sig.phash in h) or \
                           (htype == 'd' and sig.dhash and sig.dhash in h):
                            sig.thumbnail = thumb
                            break

        self._analyze_hash_layer_enhanced(
            sigs_a_used, sigs_b_used, result,
            max_matches=max_matches,
            boilerplate_hashes=boilerplate_hashes,
        )

        if not ocr_results_a or not ocr_results_b:
            result.image_risk_score = 0
            return result

        # === L2: 图片文字比对（SBERT 优先） ===
        self._detect_text_similarity(ocr_results_a, ocr_results_b, result)

        # === L3: 相同错别字 + 相同稀有词 ===
        all_words_a = []
        all_words_b = []
        for r in ocr_results_a:
            all_words_a.extend(r.words)
        for r in ocr_results_b:
            all_words_b.extend(r.words)
        self._detect_shared_typos(all_words_a, all_words_b, result)
        self._detect_shared_rare_words(all_words_a, all_words_b, result)

        # === L4: PS 嫌疑层（多证据联合） ===
        self._detect_ps_suspicious_enhanced(
            ocr_results_a, ocr_results_b, result
        )

        return result

    # ================================================================
    # L1: 多哈希共识图片匹配
    # ================================================================

    def _analyze_hash_layer_enhanced(
        self,
        sigs_a: List[ImageSignature],
        sigs_b: List[ImageSignature],
        result: ImageMatchResult,
        max_matches: int = 0,
        boilerplate_hashes: set = None,
    ):
        """L1: 多哈希共识匹配 — 要求多个哈希类型同时达标"""
        verdicts = self.hasher.match_images(
            sigs_a, sigs_b,
            max_matches=max_matches,
            boilerplate_hashes=boilerplate_hashes,
        )
        result.image_verdicts = verdicts

        for v in verdicts:
            if v.phash_dist == 0 and v.dhash_dist <= 0:
                result.exact_image_count += 1
            elif v.phash_dist <= self.HASH_NEAR_IDENTICAL_DIST:
                result.near_identical_count += 1
            else:
                result.similar_image_count += 1

        # 也运行旧版匹配保留向后兼容
        raw_hashes_a = []
        for sig in sigs_a:
            raw_hashes_a.extend(sig.raw_hashes)
        raw_hashes_b = []
        for sig in sigs_b:
            raw_hashes_b.extend(sig.raw_hashes)
        if raw_hashes_a and raw_hashes_b:
            result.hash_matches = self.hasher.match_hashes(raw_hashes_a, raw_hashes_b)

        if result.exact_image_count > 0:
            result.image_risk_factors.append(
                f"发现 {result.exact_image_count} 对完全相同图片"
            )
        if result.near_identical_count > 0:
            result.image_risk_factors.append(
                f"发现 {result.near_identical_count} 对高度相似图片"
            )
        if result.similar_image_count > 0:
            result.image_risk_factors.append(
                f"发现 {result.similar_image_count} 对相似图片"
            )

    # ================================================================
    # L2: SBERT 语义文本比对 / jieba Jaccard 回退
    # ================================================================

    def _detect_text_similarity(
        self,
        ocr_a: List[OCRResult],
        ocr_b: List[OCRResult],
        result: ImageMatchResult,
    ):
        """L2: OCR 文本相似度比对

        优先使用 SBERT 语义相似度（更准确），
        回退到 jieba Jaccard（旧逻辑兼容）。
        """
        if self.semantic_matcher and self.semantic_matcher.is_available:
            self._detect_text_similarity_sbert(ocr_a, ocr_b, result)
        else:
            self._detect_text_similarity_jaccard(ocr_a, ocr_b, result)

    def _detect_text_similarity_sbert(
        self,
        ocr_a: List[OCRResult],
        ocr_b: List[OCRResult],
        result: ImageMatchResult,
    ):
        """L2-SBERT：用 SBERT 语义模型计算文本相似度

        SBERT 比 Jaccard 更能识别"表达相同含义但用词不同"的情况。
        对于架构图标题、证书结构化文本都非常有效。
        """
        candidates = []
        text_map_a = {}
        text_map_b = {}

        for i, ra in enumerate(ocr_a):
            if len(ra.text) >= self.TEXT_MIN_LENGTH:
                text_map_a[i] = ra.text
        for j, rb in enumerate(ocr_b):
            if len(rb.text) >= self.TEXT_MIN_LENGTH:
                text_map_b[j] = rb.text

        if not text_map_a or not text_map_b:
            return

        # 将所有 OCR 文本送去 SBERT 评分
        for i_a, text_a in text_map_a.items():
            for i_b, text_b in text_map_b.items():
                candidates.append((i_a, i_b, 0.0))

        # 使用 SBERT 批量评分
        sbert_results = self.semantic_matcher.score_pairs(
            candidates, text_map_a, text_map_b,
        )

        matched_texts = []
        for sr in sbert_results:
            sim = sr['similarity']
            if sim >= self.TEXT_IDENTICAL_THRESHOLD:
                result.text_identical_count += 1
                matched_texts.append({
                    'text_a': sr['paragraph_a'][:100],
                    'text_b': sr['paragraph_b'][:100],
                    'similarity': sim,
                    'method': 'SBERT',
                })
            elif sim >= self.TEXT_SIMILAR_THRESHOLD:
                result.text_similar_count += 1

        result.text_matches = matched_texts[:20]  # 仅保留前20条详情

        if result.text_identical_count > 0:
            result.image_risk_factors.append(
                f"发现 {result.text_identical_count} 对图片SBERT语义完全相同"
            )
        if result.text_similar_count > 0:
            result.image_risk_factors.append(
                f"发现 {result.text_similar_count} 对图片文字语义高度相似"
            )

    def _detect_text_similarity_jaccard(
        self,
        ocr_a: List[OCRResult],
        ocr_b: List[OCRResult],
        result: ImageMatchResult,
    ):
        """L2-Jaccard：jieba 词级 Jaccard 回退方案"""
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

    # ================================================================
    # L3: 相同错别字 + 相同稀有词
    # ================================================================

    # 常见中文 OCR 易混淆字符对（形近字）
    _OCR_CONFUSION_PAIRS = {
        ('日', '曰'), ('己', '已'), ('末', '未'), ('土', '士'),
        ('干', '千'), ('人', '入'), ('天', '夭'), ('王', '玉'),
        ('戍', '戌'), ('概', '慨'), ('拨', '拔'), ('析', '折'),
        ('准', '淮'), ('贷', '货'), ('辨', '辩'),
        ('燥', '躁'), ('侯', '候'), ('拦', '栏'), ('历', '厉'),
        ('睛', '晴'), ('徒', '徙'), ('茶', '荼'),
    }

    def _detect_shared_typos(
        self,
        words_a: List[str],
        words_b: List[str],
        result: ImageMatchResult,
    ):
        """L3: 相同错别字检测"""
        if not words_a or not words_b:
            return

        common_words = self._get_common_word_set()

        invalid_a = {w for w in words_a
                     if len(w) >= self.TYPO_MIN_LENGTH
                     and w not in common_words
                     and not w.isascii()}
        invalid_b = {w for w in words_b
                     if len(w) >= self.TYPO_MIN_LENGTH
                     and w not in common_words
                     and not w.isascii()}

        shared_invalid = invalid_a & invalid_b

        confusion_typos = set()
        for wa in invalid_a:
            for wb in invalid_b:
                if len(wa) == len(wb) and wa != wb:
                    diffs = [(i, wa[i], wb[i]) for i in range(len(wa)) if wa[i] != wb[i]]
                    if len(diffs) == 1:
                        i, ca, cb = diffs[0]
                        if (ca, cb) in self._OCR_CONFUSION_PAIRS or \
                           (cb, ca) in self._OCR_CONFUSION_PAIRS:
                            confusion_typos.add(f"{wa}↔{wb}")

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

    def _detect_shared_rare_words(
        self,
        words_a: List[str],
        words_b: List[str],
        result: ImageMatchResult,
    ):
        """L3 补充: 检测共有稀有词（不常见但合法的词）

        两份文档的图片中出现相同的稀有技术术语 → 来源相关性信号。
        """
        if not words_a or not words_b:
            return

        common_words = self._get_common_word_set()
        rare_a = {w for w in words_a if len(w) >= 4
                  and w in common_words and not w.isascii()
                  and w not in self._get_very_common_words()}
        rare_b = {w for w in words_b if len(w) >= 4
                  and w in common_words and not w.isascii()
                  and w not in self._get_very_common_words()}

        shared_rare = rare_a & rare_b
        if shared_rare and len(shared_rare) >= 2:
            result.shared_rare_words = sorted(shared_rare)[:10]
            detail = ', '.join(result.shared_rare_words[:5])
            if len(result.shared_rare_words) > 5:
                detail += '...'
            result.image_risk_factors.append(
                f"ℹ 共有稀有词 {len(shared_rare)} 个 ({detail})"
            )

    _VERY_COMMON_CACHE = None

    @classmethod
    def _get_very_common_words(cls) -> set:
        """超常用词（过滤掉，不作为稀有词信号）"""
        if cls._VERY_COMMON_CACHE is None:
            cls._VERY_COMMON_CACHE = {
                '系统', '平台', '数据', '管理', '服务', '支持', '信息',
                '分析', '处理', '控制', '配置', '设计', '开发', '测试',
                '用户', '项目', '方案', '技术', '标准', '内容', '功能',
                '方法', '方式', '情况', '规定', '要求', '提供', '进行',
                '公司', '地址', '电话', '邮编', '日期', '编号', '名称',
            }
        return cls._VERY_COMMON_CACHE

    # ================================================================
    # L4: PS 嫌疑检测（多证据联合）
    # ================================================================

    def _detect_ps_suspicious_enhanced(
        self,
        ocr_a: List[OCRResult],
        ocr_b: List[OCRResult],
        result: ImageMatchResult,
    ):
        """L4: PS 嫌疑检测 — 双向检测

        场景A（旧L2逻辑）: 文字相同/高度相似 + 图片本身不同
          → 同一段文字被用于不同背景的图片（如套模板改样式）
        场景B（新L4逻辑）: 文字不同 + non_text_hash 接近
          → 同一张背景图上改了文字

        满足任一场景 → PS 嫌疑确认
        """
        ps_pairs = []

        for ra in ocr_a:
            if len(ra.text) < self.TEXT_MIN_LENGTH:
                continue
            for rb in ocr_b:
                if len(rb.text) < self.TEXT_MIN_LENGTH:
                    continue

                text_sim = self._text_similarity(ra.text, rb.text)

                # === 场景B: 文字不同 + non_text_hash 接近 ===
                # PS 嫌疑定义：图片（背景）相同但文字被修改过
                if text_sim < self.PS_TEXT_SIMILARITY_MAX:
                    non_text_evidence = False
                    non_text_dist = 999

                    if ra.non_text_hash and rb.non_text_hash:
                        non_text_dist = ImageHasher.hamming_distance(
                            ra.non_text_hash, rb.non_text_hash
                        )
                        if non_text_dist <= self.NON_TEXT_HASH_SIMILAR:
                            non_text_evidence = True
                    elif ra.image_hash and rb.image_hash:
                        full_dist = ImageHasher.hamming_distance(
                            ra.image_hash, rb.image_hash
                        )
                        if full_dist <= self.NON_TEXT_HASH_SIMILAR:
                            non_text_dist = full_dist
                            non_text_evidence = True

                    if not non_text_evidence:
                        continue

                    # 辅助证据
                    dimension_evidence = False
                    if ra.image_width > 0 and rb.image_width > 0:
                        w_ratio = max(ra.image_width, rb.image_width) / max(1, min(ra.image_width, rb.image_width))
                        h_ratio = max(ra.image_height, rb.image_height) / max(1, min(ra.image_height, rb.image_height))
                        if w_ratio <= 1.0 + self.PS_DIMENSION_TOLERANCE and \
                           h_ratio <= 1.0 + self.PS_DIMENSION_TOLERANCE:
                            dimension_evidence = True

                    len_evidence = False
                    text_len_a = len(ra.text)
                    text_len_b = len(rb.text)
                    if text_len_a > 0 and text_len_b > 0:
                        len_ratio = max(text_len_a, text_len_b) / max(1, min(text_len_a, text_len_b))
                        if len_ratio <= 1.5:
                            len_evidence = True

                    evidence_count = 1 + (1 if dimension_evidence else 0) + (1 if len_evidence else 0)
                    if evidence_count >= 2:
                        result.ps_suspicious_count += 1
                        ps_pairs.append({
                            'text_a': ra.text[:80],
                            'text_b': rb.text[:80],
                            'text_sim': round(text_sim, 3),
                            'non_text_dist': non_text_dist,
                            'scenario': 'B_背景相同_文字不同',
                            'evidence_count': evidence_count,
                        })

        result.ps_details = ps_pairs

        if result.ps_suspicious_count > 0:
            result.ps_suspicious = True
            result.image_risk_factors.append(
                f"⚠ PS嫌疑: {result.ps_suspicious_count} 对图片文字不同但背景相同"
            )
            logger.debug(
                f"PS 嫌疑详情: {result.ps_suspicious_count} 对, "
                f"non_text 汉明距离 avg="
                f"{sum(p['non_text_dist'] for p in ps_pairs) / max(1, len(ps_pairs)):.1f}"
            )

    # ================================================================
    # 辅助方法
    # ================================================================

    @staticmethod
    def _get_common_word_set() -> set:
        """获取常用中文词集合（缓存）"""
        if not hasattr(ImageMatcher, '_COMMON_WORDS_CACHE'):
            import jieba
            common = set()
            try:
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
            common.update({
                '运维', '管控', '模型', '训练', '系统', '平台', '数据',
                '算法', '架构', '部署', '监控', '预警', '检测', '分析',
                '接口', '模块', '配置', '参数', '日志', '缓存', '队列',
                '集群', '容器', '编排', '调度', '负载', '均衡', '代理',
            })
            ImageMatcher._COMMON_WORDS_CACHE = frozenset(common)
        return ImageMatcher._COMMON_WORDS_CACHE

    @staticmethod
    def _text_similarity(text_a: str, text_b: str) -> float:
        """jieba Jaccard 文本相似度（回退用）"""
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
    # 图片证据归纳（已不再计算风险分，保留方法签名避免调用报错）
    # ================================================================

    def _compute_image_risk_score(
        self,
        result: ImageMatchResult,
        sigs_a: List[ImageSignature] = None,
        sigs_b: List[ImageSignature] = None,
    ):
        """简化版：不再计算图片风险分"""
        result.image_risk_score = 0
