"""
Three-stage paragraph matching engine

Stage 1:  Vectorized MinHash Jaccard fast screening
Stage 2a: Exact word Jaccard matching (finds identical text)
Stage 2b: SBERT semantic verification (finds reworded/synonymous content)
"""

import logging
import jieba
from typing import List, Dict, Tuple, Optional

import numpy as np

from data_structures import BidFeature
from config import DetectionConfig
from extraction.feature_cache import DocumentCache
from matching.lsh_index import ParagraphLevelLSH
from matching.semantic_matcher import SemanticMatcher

logger = logging.getLogger(__name__)


class ParagraphMatcher:
    """Three-stage paragraph matching engine"""

    def __init__(self, config: DetectionConfig):
        self.config = config
        self.semantic_matcher: Optional[SemanticMatcher] = None

    def _ensure_semantic_matcher(self):
        if self.semantic_matcher is None:
            self.semantic_matcher = SemanticMatcher(self.config)

    def match(
        self,
        doc_a: BidFeature,
        doc_b: BidFeature,
        cache: DocumentCache,
    ) -> List[Dict]:
        """Execute three-stage paragraph matching

        Returns matches sorted by similarity descending.
        """
        # Load all paragraph MinHash signatures from SQLite
        minhashes_a = cache.load_all_paragraph_minhashes(doc_a.doc_id)
        minhashes_b = cache.load_all_paragraph_minhashes(doc_b.doc_id)

        if not minhashes_a or not minhashes_b:
            logger.warning(
                f"Missing paragraph data: {doc_a.filename}({len(minhashes_a)}) vs "
                f"{doc_b.filename}({len(minhashes_b)})"
            )
            return []

        logger.info(
            f"Three-stage matching: {doc_a.filename}({len(minhashes_a)} units) vs "
            f"{doc_b.filename}({len(minhashes_b)} units)"
        )

        # === Stage 1: Vectorized MinHash Jaccard ===
        stage1_candidates = self._stage1_vectorized_minhash(
            minhashes_a, minhashes_b
        )
        logger.info(
            f"Stage 1 (MinHash): {len(minhashes_a)}x{len(minhashes_b)} -> "
            f"{len(stage1_candidates)} candidates"
        )

        if not stage1_candidates:
            stage1_candidates = self._stage1_lsh_fallback(minhashes_a, minhashes_b)
            logger.info(f"LSH fallback: {len(stage1_candidates)} candidates")

        if not stage1_candidates:
            return []

        # 过滤跨类型候选对（OCR 段落只和 OCR 段落匹配，文本只和文本匹配）
        source_a = cache.get_paragraph_source_map(doc_a.doc_id)
        source_b = cache.get_paragraph_source_map(doc_b.doc_id)
        before = len(stage1_candidates)
        stage1_candidates = [
            (i, j, sim) for i, j, sim in stage1_candidates
            if source_a.get(i, 'text') == source_b.get(j, 'text')
        ]
        if before - len(stage1_candidates) > 0:
            logger.debug(
                f"跨类型过滤: 去除 {before - len(stage1_candidates)} 对 "
                f"(OCR↔文本不匹配)"
            )

        if not stage1_candidates:
            return []

        # Truncate to Top-K
        top_k = min(len(stage1_candidates), self.config.PARAGRAPH_MATCH_STAGE1_TOP_K)
        stage1_candidates = stage1_candidates[:top_k]

        # === Stage 2: Text verification ===
        self._ensure_semantic_matcher()

        if self.semantic_matcher.is_available:
            # Load paragraph texts (only for candidates)
            para_texts_a = {}
            para_texts_b = {}

            for i, j, _ in stage1_candidates:
                if i not in para_texts_a:
                    text = cache.load_paragraph_text(doc_a.doc_id, i)
                    if text:
                        para_texts_a[i] = text
                if j not in para_texts_b:
                    text = cache.load_paragraph_text(doc_b.doc_id, j)
                    if text:
                        para_texts_b[j] = text

            # Filter candidates with valid text
            valid_candidates = [
                (i, j, sim) for i, j, sim in stage1_candidates
                if i in para_texts_a and j in para_texts_b
            ]

            if valid_candidates:
                # === Stage 2a: Exact word Jaccard (fast, finds identical text) ===
                exact_matches = []
                semantic_candidates = []

                # 预计算所有唯一段落文本的 jieba 分词集合（避免循环内重复分词）
                cached_words_a = {}
                cached_words_b = {}

                for i, j, minhash_sim in valid_candidates:
                    text_a = para_texts_a[i]
                    text_b = para_texts_b[j]
                    if not text_a or not text_b:
                        continue

                    # 使用缓存避免对同一段落重复分词
                    if i not in cached_words_a:
                        words = set(jieba.cut(text_a))
                        cached_words_a[i] = {w for w in words if len(w) > 1}
                    if j not in cached_words_b:
                        words = set(jieba.cut(text_b))
                        cached_words_b[j] = {w for w in words if len(w) > 1}

                    words_a = cached_words_a[i]
                    words_b = cached_words_b[j]

                    if not words_a or not words_b:
                        continue

                    intersection = words_a & words_b
                    union = words_a | words_b
                    word_jaccard = len(intersection) / len(union) if union else 0

                    if word_jaccard >= 0.75:
                        # High text overlap -> confirm directly (no SBERT needed)
                        exact_matches.append({
                            'similarity': word_jaccard,
                            'paragraph_a_index': i,
                            'paragraph_b_index': j,
                            'detection_method': 'Exact-Jaccard',
                            'paragraph_a': text_a,
                            'paragraph_b': text_b,
                            'is_continuous_clone': False,
                            'continuous_clone_group_id': '',
                            'highlighted_text_a': '',
                            'highlighted_text_b': '',
                            'common_parts': [],
                        })
                    elif word_jaccard >= 0.15:
                        # Medium similarity -> send to SBERT
                        semantic_candidates.append((i, j, word_jaccard))

                logger.info(
                    f"Stage 2a (Jaccard): {len(exact_matches)} exact matches, "
                    f"{len(semantic_candidates)} to SBERT"
                )

                # === Stage 2b: SBERT semantic verification ===
                sbert_results = []
                if semantic_candidates:
                    # 检查是否有预计算嵌入缓存（Phase 1.5 已运行）
                    if (self.config.ENABLE_EMBEDDING_CACHE
                            and hasattr(self.semantic_matcher, 'score_pairs_from_cache')):
                        sbert_results = self.semantic_matcher.score_pairs_from_cache(
                            semantic_candidates, para_texts_a, para_texts_b, cache,
                            doc_a_id=doc_a.doc_id, doc_b_id=doc_b.doc_id,
                        )
                    else:
                        # 回退：实时 SBERT 编码
                        sbert_results = self.semantic_matcher.score_pairs(
                            semantic_candidates, para_texts_a, para_texts_b
                        )

                # Merge results: exact matches first
                stage2_results = exact_matches + sbert_results
                stage2_results.sort(key=lambda x: x['similarity'], reverse=True)

            else:
                stage2_results = []
        else:
            # SBERT unavailable, use MinHash Jaccard as final score
            logger.warning("SBERT unavailable, using MinHash Jaccard")
            stage2_results = [
                {
                    'similarity': sim,
                    'paragraph_a_index': i,
                    'paragraph_b_index': j,
                    'detection_method': 'MinHash-Jaccard',
                    'paragraph_a': '',
                    'paragraph_b': '',
                }
                for i, j, sim in stage1_candidates
            ]

        # Fill in paragraph text for report generation
        for result in stage2_results:
            i = result['paragraph_a_index']
            j = result['paragraph_b_index']
            if not result.get('paragraph_a'):
                result['paragraph_a'] = cache.load_paragraph_text(doc_a.doc_id, i) or ''
            if not result.get('paragraph_b'):
                result['paragraph_b'] = cache.load_paragraph_text(doc_b.doc_id, j) or ''

        # === 标书模板语过滤：降低招标文件原文/通用模板语的权重 ===
        if getattr(self.config, 'BID_BOILERPLATE_FILTER', False):
            boilerplate_count = 0
            for result in stage2_results:
                bp_ratio = _compute_boilerplate_ratio(
                    result.get('paragraph_a', ''),
                    result.get('paragraph_b', ''),
                )
                result['boilerplate_ratio'] = bp_ratio
                # 模板语比例越高，相似度衰减越多
                weight = getattr(self.config, 'BID_BOILERPLATE_WEIGHT', 0.3)
                decay = 1.0 - bp_ratio * weight
                original_sim = result['similarity']
                result['similarity'] = round(original_sim * decay, 4)
                result['original_similarity'] = original_sim  # 保留原始分数供参考
                if bp_ratio > 0.5:
                    boilerplate_count += 1
            if boilerplate_count > 0:
                logger.info(
                    f"Boilerplate filter: {boilerplate_count}/{len(stage2_results)} "
                    f"matches have high template language ratio"
                )

        # Sort by similarity descending
        stage2_results.sort(key=lambda x: x['similarity'], reverse=True)

        # Limit final match count
        if len(stage2_results) > self.config.PARAGRAPH_MATCH_STAGE2_TOP_K:
            stage2_results = stage2_results[:self.config.PARAGRAPH_MATCH_STAGE2_TOP_K]

        vc_count = len(valid_candidates) if 'valid_candidates' in dir() else 0
        logger.info(
            f"Stage 2 results: {vc_count} candidates -> {len(stage2_results)} matches"
        )

        return stage2_results

    def _stage1_vectorized_minhash(
        self,
        minhashes_a: Dict[int, str],
        minhashes_b: Dict[int, str],
    ) -> List[Tuple[int, int, float]]:
        """Vectorized MinHash Jaccard computation

        Uses numpy broadcasting for ~100x speedup vs nested loops.
        """
        indices_a = list(minhashes_a.keys())
        indices_b = list(minhashes_b.keys())

        if not indices_a or not indices_b:
            return []

        first_hash = minhashes_a[indices_a[0]]
        if not first_hash:
            return []
        dim = len(first_hash.split(','))

        def build_matrix(indices, minhash_dict):
            mat = np.zeros((len(indices), dim), dtype=np.int64)
            for idx, para_idx in enumerate(indices):
                h_str = minhash_dict[para_idx]
                if h_str:
                    try:
                        values = [int(v) for v in h_str.split(',')]
                        if len(values) == dim:
                            mat[idx] = [v % (2**63 - 1) for v in values]
                    except (ValueError, OverflowError) as e:
                        logger.debug(f"MinHash conversion failed para={para_idx}: {e}")
                        continue
            return mat

        mat_a = build_matrix(indices_a, minhashes_a)
        mat_b = build_matrix(indices_b, minhashes_b)

        matches = (mat_a[:, None, :] == mat_b[None, :, :])
        jaccard = matches.mean(axis=2)

        threshold = self.config.PARAGRAPH_MIN_JACCARD
        candidate_mask = jaccard >= threshold

        if not candidate_mask.any():
            return []

        candidate_coords = np.argwhere(candidate_mask)
        results = []
        for a_idx, b_idx in candidate_coords:
            para_a = indices_a[a_idx]
            para_b = indices_b[b_idx]
            sim = float(jaccard[a_idx, b_idx])
            results.append((para_a, para_b, sim))

        results.sort(key=lambda x: x[2], reverse=True)
        return results

    def _stage1_lsh_fallback(
        self,
        minhashes_a: Dict[int, str],
        minhashes_b: Dict[int, str],
    ) -> List[Tuple[int, int, float]]:
        """LSH fallback when vectorized method returns empty"""
        try:
            paras_a = [
                {'para_index': i, 'minhash': h}
                for i, h in minhashes_a.items()
            ]
            paras_b = [
                {'para_index': i, 'minhash': h}
                for i, h in minhashes_b.items()
            ]
            lsh = ParagraphLevelLSH(self.config)
            return lsh.build_and_query(paras_a, paras_b)
        except ImportError:
            return []


# ================================================================
# 标书模板语检测 — 降低招标文件原文导致的误检
# ================================================================

# 中国标书中常见模板语/招标文件原文关键词
_BID_BOILERPLATE_PATTERNS = [
    # 测试验收类
    '测试过程', '操作步骤', '期望结果', '评估准则', '测试结果与预期',
    '验收标准', '验收条件', '测试方法', '测试用例', '检查测试结果',
    # 服务承诺类
    '投标人应具备', '投标人承诺', '服务承诺', '售后服务',
    '项目合同期内', '验收后', '维护期内', '质量保证期',
    '稳定性和安全性', '提供技术服务', '现场或远程', '技术支持服务',
    'BUG', '修复', '运行异常处理', '响应',
    # 招标响应类
    '符合招标文件', '满足招标', '无偏离', '完全响应', '正偏离',
    '招标文件要求', '技术规格', '商务条款',
    # 格式模板类
    '序号', '项目名称', '规格型号', '技术参数', '备注',
    '投标人名称', '法定代表人', '授权代表',
    # 通用管理类
    '质量保证体系', '安全管理', '环境保护', '文明施工',
    '项目管理', '进度计划', '资源配置', '人员配备',
]


def _compute_boilerplate_ratio(text_a: str, text_b: str) -> float:
    """计算两个段落中模板语的占比（0.0=全部原创, 1.0=全是模板语）

    模板语定义：在中国标书中反复出现的招标文件原文、通用格式语言。
    这类内容来自招标文件而非投标人原创，不应作为串标证据。

    算法：统计两个文本中模板语关键词的命中率。
    """
    if not text_a or not text_b:
        return 0.0

    a_hits = 0
    b_hits = 0
    a_len = max(len(text_a), 1)
    b_len = max(len(text_b), 1)

    for pattern in _BID_BOILERPLATE_PATTERNS:
        if pattern in text_a:
            a_hits += len(pattern)
        if pattern in text_b:
            b_hits += len(pattern)

    ratio_a = min(1.0, a_hits / a_len)
    ratio_b = min(1.0, b_hits / b_len)

    # 取两段中模板语比例的最大值（只要有一段是模板语就降低权重）
    return max(ratio_a, ratio_b)
