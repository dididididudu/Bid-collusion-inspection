"""
Three-stage paragraph matching engine

Stage 1:  Vectorized MinHash Jaccard fast screening
Stage 2a: Exact word Jaccard matching (finds identical text)
Stage 2b: SBERT semantic verification (finds reworded/synonymous content)
"""

import logging
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

        minhashes_a = {k: v['minhash'] for k, v in para_full_a.items() if v['minhash']}
        minhashes_b = {k: v['minhash'] for k, v in para_full_b.items() if v['minhash']}

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

        if self.semantic_matcher.is_available:
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
                    seq_ratio = SequenceMatcher(None, text_a, text_b).ratio() if text_a and text_b else 0.0

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

        # 确保所有 minhash 值是字符串（数据库可能返回非字符串类型）
        for k, v in list(minhashes_a.items()):
            if not isinstance(v, str):
                minhashes_a[k] = str(v)
            elif not v:
                del minhashes_a[k]
        for k, v in list(minhashes_b.items()):
            if not isinstance(v, str):
                minhashes_b[k] = str(v)
            elif not v:
                del minhashes_b[k]

        indices_a = list(minhashes_a.keys())
        indices_b = list(minhashes_b.keys())
        if not indices_a or not indices_b:
            return []

        first_hash = minhashes_a[indices_a[0]]
        if not first_hash:
            return []
        dim = len(first_hash.split(','))

        def build_matrix(indices, minhash_dict):
            """向量化 MinHash 解析 — np.fromstring 替代 Python 逐值循环"""
            hashes = [minhash_dict[i] for i in indices if minhash_dict.get(i, '')]
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
    '招标文件要求', '技术规格', '商务条款',
    # 格式模板类
    '序号', '项目名称', '规格型号', '技术参数', '备注',
    '投标人名称', '法定代表人', '授权代表',
    # 通用管理类
    '质量保证体系', '安全管理', '环境保护', '文明施工',
    '项目管理', '进度计划', '资源配置', '人员配备',
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
