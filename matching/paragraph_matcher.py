"""
Three-stage paragraph matching engine

Stage 1:  Vectorized MinHash Jaccard fast screening
Stage 2a: Exact word Jaccard matching (finds identical text)
Stage 2b: SBERT semantic verification (finds reworded/synonymous content)
"""

import logging
import hashlib
import jieba
from difflib import SequenceMatcher
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
        para_full_a: Dict[int, dict] = None,
        para_full_b: Dict[int, dict] = None,
        para_embeddings_a: Dict[int, np.ndarray] = None,
        para_embeddings_b: Dict[int, np.ndarray] = None,
    ) -> List[Dict]:
        """Execute three-stage paragraph matching

        Args:
            para_full_a/b: 可选的预加载段落数据。传入后可跳过 cache 查询，
                           避免同一文档在多个配对中重复从 SQLite 加载。

        Returns matches sorted by similarity descending.
        """
        # 使用调用方传入的预加载段落数据，避免重复 SQLite 查询
        if para_full_a is None:
            para_full_a = cache.load_all_paragraphs_full(doc_a.doc_id)
        if para_full_b is None:
            para_full_b = cache.load_all_paragraphs_full(doc_b.doc_id)

        minhashes_a = {
            k: (v.get('minhash_array') if v.get('minhash_array') is not None else v.get('minhash', ''))
            for k, v in para_full_a.items()
            if v.get('minhash_array') is not None or v.get('minhash')
        }
        minhashes_b = {
            k: (v.get('minhash_array') if v.get('minhash_array') is not None else v.get('minhash', ''))
            for k, v in para_full_b.items()
            if v.get('minhash_array') is not None or v.get('minhash')
        }

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

        # 过滤跨类型候选对（从 para_full 读取 source，无需额外查询）
        before = len(stage1_candidates)
        stage1_candidates = [
            (i, j, sim) for i, j, sim in stage1_candidates
            if para_full_a.get(i, {}).get('source', 'text') == para_full_b.get(j, {}).get('source', 'text')
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

        # 初始化 valid_candidates，确保所有分支中该变量都有定义
        valid_candidates = []

        cache_sbert_available = (
            self.config.ENABLE_EMBEDDING_CACHE
            and hasattr(self.semantic_matcher, 'score_pairs_from_cache')
        )
        realtime_sbert_available = False if cache_sbert_available else self.semantic_matcher.is_available

        if cache_sbert_available or realtime_sbert_available:
            # 从 para_full 直接读取文本和预分词（已在 match 开头一次查询加载）
            para_texts_a = {}
            para_texts_b = {}
            word_sets_a = {}
            word_sets_b = {}

            for i, j, _ in stage1_candidates:
                if i not in para_texts_a and i in para_full_a:
                    pd = para_full_a[i]
                    para_texts_a[i] = pd['text']
                    tokens = pd['tokens']
                    word_sets_a[i] = {w for w in tokens if len(w) > 1} if tokens else {w for w in jieba.cut(pd['text']) if len(w) > 1}
                if j not in para_texts_b and j in para_full_b:
                    pd = para_full_b[j]
                    para_texts_b[j] = pd['text']
                    tokens = pd['tokens']
                    word_sets_b[j] = {w for w in tokens if len(w) > 1} if tokens else {w for w in jieba.cut(pd['text']) if len(w) > 1}

            valid_candidates = [
                (i, j, sim) for i, j, sim in stage1_candidates
                if i in para_texts_a and j in para_texts_b
            ]

            if valid_candidates:
                exact_matches = []
                semantic_candidates = []

                for i, j, minhash_sim in valid_candidates:
                    words_a = word_sets_a.get(i, set())
                    words_b = word_sets_b.get(j, set())
                    if not words_a or not words_b:
                        continue

                    intersection = words_a & words_b
                    union = words_a | words_b
                    word_jaccard = len(intersection) / len(union) if union else 0

                    text_a = para_texts_a.get(i, '')
                    text_b = para_texts_b.get(j, '')
                    seq_ratio = self._fast_sequence_ratio(
                        text_a, text_b, word_jaccard, minhash_sim
                    )

                    if seq_ratio > 0.85:
                        exact_matches.append({
                            'similarity': seq_ratio,
                            'paragraph_a_index': i,
                            'paragraph_b_index': j,
                            'page_num_a': para_full_a.get(i, {}).get('page_num', -1),
                            'page_num_b': para_full_b.get(j, {}).get('page_num', -1),
                            'detection_method': 'Exact-Jaccard',
                            'paragraph_a': text_a,
                            'paragraph_b': text_b,
                            'is_continuous_clone': False,
                            'continuous_clone_group_id': '',
                            'highlighted_text_a': '',
                            'highlighted_text_b': '',
                            'common_parts': [],
                        })
                    elif seq_ratio >= 0.4:
                        semantic_candidates.append((i, j, seq_ratio))
                    elif minhash_sim >= 0.25:
                        # 召回补充：词级重合较高但字符级低（同义改写），送入语义验证
                        semantic_candidates.append((i, j, seq_ratio))

                logger.info(
                    f"Stage 2a (Jaccard): {len(exact_matches)} exact matches, "
                    f"{len(semantic_candidates)} to SBERT"
                )

                # === Stage 2b: SBERT semantic verification ===
                sbert_results = []
                if semantic_candidates:
                    # 检查是否有预计算嵌入缓存（Phase 1.5 已运行）
                    if cache_sbert_available:
                        sbert_results = self.semantic_matcher.score_pairs_from_cache(
                            semantic_candidates, para_texts_a, para_texts_b, cache,
                            doc_a_id=doc_a.doc_id, doc_b_id=doc_b.doc_id,
                            preloaded_embeddings_a=para_embeddings_a,
                            preloaded_embeddings_b=para_embeddings_b,
                        )
                        logger.debug(f"score_pairs_from_cache: {len(semantic_candidates)} candidates -> {len(sbert_results)} results")
                    elif realtime_sbert_available:
                        # 回退：实时 SBERT 编码
                        sbert_results = self.semantic_matcher.score_pairs(
                            semantic_candidates, para_texts_a, para_texts_b
                        )
                        logger.debug(f"score_pairs: {len(semantic_candidates)} candidates -> {len(sbert_results)} results")

                # Merge results: exact matches first
                stage2_results = exact_matches + sbert_results

                # 内容词重叠后过滤：移除 SBERT 相似但内容词不重叠的模板匹配
                # (如"承诺方名称：A公司（盖章）" vs "投标人名称：B公司（公章）")
                def _content_tokens(text):
                    toks = set(w for w in jieba.cut(text) if len(w) >= 2)
                    return toks
                filtered = []
                post_filter_removed = 0
                for r in stage2_results:
                    if r.get('detection_method') == 'SBERT' and r['similarity'] < 0.90:
                        toks_a = _content_tokens(r.get('paragraph_a', ''))
                        toks_b = _content_tokens(r.get('paragraph_b', ''))
                        if toks_a and toks_b:
                            overlap = len(toks_a & toks_b) / len(toks_a | toks_b)
                            if overlap < 0.20:
                                post_filter_removed += 1
                                continue
                    filtered.append(r)
                if post_filter_removed > 0:
                    logger.info(f"Post-filter removed {post_filter_removed} SBERT matches (content word overlap < 0.20)")
                stage2_results = filtered
                stage2_results.sort(key=lambda x: x['similarity'], reverse=True)

                # 文本去重：相同文本对（如签名块在多页重复出现）只保留相似度最高的一条
                seen_text_pairs = set()
                deduped = []
                dedup_removed = 0
                for r in stage2_results:
                    text_key = (
                        r.get('paragraph_a', '')[:200],
                        r.get('paragraph_b', '')[:200],
                    )
                    if text_key in seen_text_pairs:
                        dedup_removed += 1
                        continue
                    seen_text_pairs.add(text_key)
                    deduped.append(r)
                if dedup_removed > 0:
                    logger.info(f"Dedup removed {dedup_removed} duplicate matches (same text pair)")
                stage2_results = deduped

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

        # Fill in paragraph text and page numbers for report generation
        for result in stage2_results:
            i = result['paragraph_a_index']
            j = result['paragraph_b_index']
            # 从已加载的 para_full 获取（无 SQLite 查询）
            if not result.get('paragraph_a'):
                result['paragraph_a'] = para_full_a.get(i, {}).get('text', '')
            if not result.get('paragraph_b'):
                result['paragraph_b'] = para_full_b.get(j, {}).get('text', '')
            # 填充页码
            if 'page_num_a' not in result:
                result['page_num_a'] = para_full_a.get(i, {}).get('page_num', -1)
            if 'page_num_b' not in result:
                result['page_num_b'] = para_full_b.get(j, {}).get('page_num', -1)

        # === 标书模板语过滤：降低招标文件原文/通用模板语的权重 ===
        if getattr(self.config, 'BID_BOILERPLATE_FILTER', False):
            boilerplate_count = 0
            hard_filtered_count = 0
            template_hashes = set(getattr(self.config, 'BID_TEMPLATE_TEXT_HASHES', []) or [])
            for result in stage2_results:
                text_a = result.get('paragraph_a', '')
                text_b = result.get('paragraph_b', '')
                bp_ratio = _compute_boilerplate_ratio(text_a, text_b)
                info_score = min(
                    _compute_informativeness_score(text_a, bp_ratio),
                    _compute_informativeness_score(text_b, bp_ratio),
                )
                hash_a = _template_text_hash(text_a)
                hash_b = _template_text_hash(text_b)
                batch_common = bool(
                    template_hashes
                    and (hash_a in template_hashes or hash_b in template_hashes)
                )
                result['boilerplate_ratio'] = bp_ratio
                result['informativeness_score'] = round(info_score, 4)
                result['batch_common_template'] = batch_common

                if _should_hard_filter_template_match(result, self.config):
                    result['filtered_reason'] = 'template_low_information'
                    hard_filtered_count += 1
                    continue

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
            if hard_filtered_count > 0:
                logger.info(
                    f"Template hard filter: removed {hard_filtered_count} "
                    f"low-information boilerplate matches"
                )

            stage2_results = [
                r for r in stage2_results
                if r.get('filtered_reason') != 'template_low_information'
            ]

            # 衰减后重新阈值过滤：模板语导致相似度衰减后低于阈值的匹配应移除
            # （如"投标人名称：A公司（公章）" 原始 sim=0.90 通过阈值，衰减后 0.84 应被过滤）
            post_decay_filtered = []
            post_decay_removed = 0
            for result in stage2_results:
                text_a = result.get('paragraph_a', '')
                text_b = result.get('paragraph_b', '')
                avg_len = (len(text_a) + len(text_b)) / 2
                if avg_len < self.config.SBERT_SHORT_PARAGRAPH_LEN:
                    threshold = self.config.SBERT_SHORT_PARAGRAPH_THRESHOLD
                else:
                    threshold = self.config.SBERT_BASE_THRESHOLD
                threshold = max(0.75, min(0.90, threshold))
                if result['similarity'] < threshold:
                    post_decay_removed += 1
                    continue
                post_decay_filtered.append(result)
            if post_decay_removed > 0:
                logger.info(
                    f"Post-decay threshold filter: removed {post_decay_removed} matches "
                    f"(similarity fell below threshold after boilerplate decay)"
                )
            stage2_results = post_decay_filtered

        # Sort by similarity descending
        stage2_results.sort(key=lambda x: x['similarity'], reverse=True)

        # Limit final match count
        if len(stage2_results) > self.config.PARAGRAPH_MATCH_STAGE2_TOP_K:
            stage2_results = stage2_results[:self.config.PARAGRAPH_MATCH_STAGE2_TOP_K]

        vc_count = len(valid_candidates)
        logger.info(
            f"Stage 2 results: {vc_count} candidates -> {len(stage2_results)} matches"
        )

        self._mark_continuous_clones(stage2_results)

        return stage2_results

    def _mark_continuous_clones(self, results: List[Dict]) -> None:
        """标记连续克隆块（超过3个连续相似段落）"""
        if len(results) < 3:
            return

        sorted_results = sorted(results, key=lambda x: x['paragraph_a_index'])

        clone_group_id = 0
        run_start = 0
        run_len = 1

        for i in range(1, len(sorted_results)):
            prev = sorted_results[i - 1]
            curr = sorted_results[i]
            if (curr['paragraph_a_index'] == prev['paragraph_a_index'] + 1 and
                    curr['paragraph_b_index'] == prev['paragraph_b_index'] + 1):
                run_len += 1
            else:
                if run_len >= 3:
                    clone_group_id += 1
                    for k in range(run_start, run_start + run_len):
                        sorted_results[k]['is_continuous_clone'] = True
                        sorted_results[k]['continuous_clone_group_id'] = f'clone_{clone_group_id}'
                run_start = i
                run_len = 1

        if run_len >= 3:
            clone_group_id += 1
            for k in range(run_start, run_start + run_len):
                sorted_results[k]['is_continuous_clone'] = True
                sorted_results[k]['continuous_clone_group_id'] = f'clone_{clone_group_id}'

        clone_count = sum(1 for r in results if r.get('is_continuous_clone'))
        if clone_count > 0:
            logger.info(f"连续克隆块检测: {clone_count} 个段落属于连续克隆块")

    def _fast_sequence_ratio(
        self, text_a: str, text_b: str, word_jaccard: float, minhash_sim: float
    ) -> float:
        """对明显不相似或过长候选短路，避免 difflib 吃满 CPU。"""
        if not text_a or not text_b:
            return 0.0
        len_a, len_b = len(text_a), len(text_b)
        short_len, long_len = min(len_a, len_b), max(len_a, len_b)
        if long_len == 0:
            return 0.0
        length_ratio = short_len / long_len
        min_ratio = getattr(self.config, 'SEQUENCE_MATCHER_LENGTH_RATIO', 0.55)
        if length_ratio < min_ratio and word_jaccard < 0.20 and minhash_sim < 0.25:
            return 0.0
        max_chars = getattr(self.config, 'SEQUENCE_MATCHER_MAX_CHARS', 1200)
        if long_len > max_chars and word_jaccard < 0.35 and minhash_sim < 0.35:
            return min(word_jaccard, minhash_sim)
        if long_len > max_chars:
            head = max_chars // 2
            tail = max_chars - head
            text_a = text_a[:head] + text_a[-tail:]
            text_b = text_b[:head] + text_b[-tail:]
        return SequenceMatcher(None, text_a, text_b).ratio()

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

        # 新库直接用 minhash_blob 反序列化出的 ndarray，旧库回退字符串解析。
        for k, v in list(minhashes_a.items()):
            if isinstance(v, np.ndarray):
                continue
            if not isinstance(v, str):
                minhashes_a[k] = str(v)
            elif not v:
                del minhashes_a[k]
        for k, v in list(minhashes_b.items()):
            if isinstance(v, np.ndarray):
                continue
            if not isinstance(v, str):
                minhashes_b[k] = str(v)
            elif not v:
                del minhashes_b[k]

        indices_a = list(minhashes_a.keys())
        indices_b = list(minhashes_b.keys())
        if not indices_a or not indices_b:
            return []

        first_hash = minhashes_a[indices_a[0]]
        if first_hash is None or (isinstance(first_hash, str) and not first_hash):
            return []
        dim = len(first_hash) if isinstance(first_hash, np.ndarray) else len(first_hash.split(','))

        def build_matrix(indices, minhash_dict):
            """向量化 MinHash 解析 — np.fromstring 替代 Python 逐值循环"""
            values_for_indices = [minhash_dict[i] for i in indices if i in minhash_dict]
            if values_for_indices and isinstance(values_for_indices[0], np.ndarray):
                try:
                    return np.stack(values_for_indices).astype(np.int64, copy=False)
                except Exception as e:
                    logger.debug(f"MinHash BLOB 堆叠失败, 回退字符串解析: {e}")
            hashes = [v for v in values_for_indices if isinstance(v, str) and v]
            if not hashes:
                return np.zeros((len(indices), dim), dtype=np.int64)
            try:
                flat = ','.join(hashes)
                values = np.fromstring(flat, sep=',', dtype=np.int64)
                mat = values.reshape(len(hashes), dim) % (2**63 - 1)
                # 补齐缺行（某些段落 MinHash 为空）
                full = np.zeros((len(indices), dim), dtype=np.int64)
                full[:len(hashes)] = mat
                return full
            except Exception as e:
                logger.debug(f"MinHash 向量化解析失败, 回退逐值: {e}")
                # Fallback
                mat = np.zeros((len(indices), dim), dtype=np.int64)
                for idx, para_idx in enumerate(indices):
                    h_str = minhash_dict.get(para_idx, '')
                    if not h_str: continue
                    try:
                        vals = np.fromstring(h_str, sep=',', dtype=np.int64)
                        if len(vals) == dim:
                            mat[idx] = vals
                    except Exception:
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
    '招标文件要求', '技术规格', '商务条款', '采购文件要求',
    '响应招标文件', '响应采购文件', '按照招标文件', '严格按照',
    '采购人要求', '招标人要求', '本项目要求', '本项目',
    # 格式模板类
    '序号', '项目名称', '规格型号', '技术参数', '备注',
    '投标人名称', '法定代表人', '授权代表', '盖章', '签字',
    '日期', '联系人', '联系电话',
    # 通用管理类
    '质量保证体系', '安全管理', '环境保护', '文明施工',
    '项目管理', '进度计划', '资源配置', '人员配备',
    '施工方案', '施工组织设计', '施工组织', '确保质量',
    '确保安全', '按时完成', '保质保量', '法律法规',
]


# 预编译模板语正则（单次扫描替代逐 pattern 循环）
import re as _re
_BOILERPLATE_RE = _re.compile('|'.join(_re.escape(p) for p in _BID_BOILERPLATE_PATTERNS))


def _compute_boilerplate_ratio(text_a: str, text_b: str) -> float:
    """单次正则扫描替代逐 pattern 循环（~50× 加速）"""
    if not text_a or not text_b:
        return 0.0
    a_hits = sum(len(m.group()) for m in _BOILERPLATE_RE.finditer(text_a))
    b_hits = sum(len(m.group()) for m in _BOILERPLATE_RE.finditer(text_b))
    return max(min(1.0, a_hits / max(len(text_a), 1)),
               min(1.0, b_hits / max(len(text_b), 1)))


def _normalize_template_text(text: str) -> str:
    """归一化模板段落文本，用于跨文档统计同源模板句。"""
    text = _re.sub(r"\s+", "", text or "")
    text = _re.sub(r"[A-Za-z0-9_.+-]+@[A-Za-z0-9_.+-]+", "EMAIL", text)
    text = _re.sub(r"1[3-9]\d{9}", "PHONE", text)
    text = _re.sub(r"\d+(?:\.\d+)?", "#", text)
    return text[:500]


def _template_text_hash(text: str) -> str:
    norm = _normalize_template_text(text)
    if not norm:
        return ''
    return hashlib.md5(norm.encode('utf-8')).hexdigest()


def _compute_informativeness_score(text: str, boilerplate_ratio: float = 0.0) -> float:
    """估算段落证据价值。

    分数越高，越像包含项目特异信息；分数越低，越像通用承诺、格式字段、
    招标文件原文或模板套话。
    """
    if not text:
        return 0.0

    compact = _re.sub(r"\s+", "", text)
    tokens = [w for w in jieba.cut(compact) if len(w) >= 2]
    unique_tokens = set(tokens)
    long_tokens = [w for w in unique_tokens if len(w) >= 4]
    digit_groups = _re.findall(r"\d+(?:\.\d+)?", compact)
    specific_markers = [
        '型号', '设备', '品牌', '参数', '编号', '节点', '工期',
        '人员', '岗位', '证书', '系统', '模块', '接口', '数据库',
        '算法', '流程', '清单', '报价', '金额', '数量', '规格',
    ]
    marker_hits = sum(1 for kw in specific_markers if kw in compact)

    token_score = min(0.25, len(unique_tokens) / 40)
    long_token_score = min(0.20, len(long_tokens) / 25)
    digit_score = min(0.20, len(digit_groups) / 20)
    marker_score = min(0.25, marker_hits / 12)
    length_score = min(0.10, len(compact) / 800)
    penalty = min(0.35, boilerplate_ratio * 0.35)

    score = token_score + long_token_score + digit_score + marker_score + length_score - penalty
    return max(0.0, min(1.0, score))


def _should_hard_filter_template_match(result: Dict, config: DetectionConfig) -> bool:
    """判断段落匹配是否只属于模板套话，不进入相似证据。"""
    if not getattr(config, 'BID_TEMPLATE_HARD_FILTER', True):
        return False

    bp_ratio = result.get('boilerplate_ratio', 0.0)
    info_score = result.get('informativeness_score', 1.0)
    batch_common = result.get('batch_common_template', False)
    sim = result.get('similarity', 0.0)
    text_a = result.get('paragraph_a', '') or ''
    text_b = result.get('paragraph_b', '') or ''
    avg_len = (len(text_a) + len(text_b)) / 2

    bp_threshold = getattr(config, 'BID_TEMPLATE_RATIO_THRESHOLD', 0.55)
    info_threshold = getattr(config, 'BID_TEMPLATE_MIN_INFO_SCORE', 0.28)

    if avg_len < 8:
        return True

    # 批次高频模板段落：即便相似度高，也更像公共模板。
    if batch_common and info_score < max(0.40, info_threshold + 0.08):
        return True

    # 高模板覆盖 + 低信息量：直接不作为证据。
    if bp_ratio >= bp_threshold and info_score < info_threshold:
        return True

    # 短格式字段对相似度特别敏感，低信息量时直接过滤。
    if avg_len < 80 and bp_ratio >= 0.35 and info_score < 0.35 and sim < 0.98:
        return True

    return False
