"""
候选文档对选择器 — 多层级筛选策略 (v2)

Level 1: datasketch MinHashLSH 文档级筛选（O(n)）
Level 2: 元数据指纹匹配（software_fingerprint / time_bucket 相同 → 强制候选）
Level 3: 文档向量余弦预筛（过滤文档级不相似的对）
Level 4: 图片哈希匹配

支持回退到全量比对（当文档数较小时）
"""

import logging
from typing import List, Tuple, Set, Optional, Dict
from collections import defaultdict

import numpy as np

from data_structures import BidFeature
from config import DetectionConfig
from matching.lsh_index import DocLevelLSHIndex

logger = logging.getLogger(__name__)


class CandidatePairSelector:
    """候选文档对选择器（多层级筛选 + 文档向量预筛 + 元数据指纹）"""

    def __init__(self, config: DetectionConfig):
        self.config = config

    def select(
        self,
        features: List[BidFeature],
        cache=None,  # Optional[DocumentCache]
    ) -> List[Tuple[str, str, str, float]]:
        """选择候选文档对

        Args:
            features: 文档特征列表
            cache: DocumentCache 实例（可选，用于元数据指纹和文档向量查询）

        Returns:
            [(doc_a_id, doc_b_id, method, similarity), ...]
        """
        candidate_pairs: dict = {}  # (id_a, id_b) -> (method, similarity)

        valid_features = [f for f in features if not f.is_scanned]
        all_features = features

        # 小规模文档集直接全量比对
        if len(valid_features) <= 5:
            logger.info(
                f"文档数较少 ({len(valid_features)} 个)，直接全量比对"
            )
            for i in range(len(valid_features)):
                for j in range(i + 1, len(valid_features)):
                    pair = tuple(sorted([
                        valid_features[i].doc_id, valid_features[j].doc_id
                    ]))
                    candidate_pairs[pair] = ("all_pairs", 0.0)
        else:
            # Level 1: datasketch MinHashLSH
            try:
                lsh_pairs = self._datasketch_lsh(valid_features)
                logger.info(f"Level 1 LSH 初筛: {len(lsh_pairs)} 对")
                for pair, sim in lsh_pairs.items():
                    if pair not in candidate_pairs:
                        candidate_pairs[pair] = ("lsh", sim)
            except ImportError:
                logger.warning("datasketch 不可用，使用 SimHash 回退方案")
                simhash_pairs = self._simhash_fallback(valid_features)
                for pair in simhash_pairs:
                    if pair not in candidate_pairs:
                        candidate_pairs[pair] = ("simhash", 0.0)

            # Level 2: 元数据指纹匹配（新增）
            if self.config.METADATA_FILTER_ENABLED and cache is not None:
                metadata_pairs = self._metadata_fingerprint_filter(cache)
                logger.info(f"Level 2 元数据指纹: {len(metadata_pairs)} 对")
                for pair in metadata_pairs:
                    if pair not in candidate_pairs:
                        candidate_pairs[pair] = ("metadata_fingerprint", 0.0)

            # 传统元数据聚类（保留作为补充）
            metadata_pairs = self._metadata_clustering(all_features)
            logger.info(f"元数据聚类补充: {len(metadata_pairs)} 对")
            for pair in metadata_pairs:
                if pair not in candidate_pairs:
                    candidate_pairs[pair] = ("metadata", 0.0)

            # Level 3: 文档向量余弦预筛（新增）
            if (self.config.DOC_VECTOR_FILTER_ENABLED
                    and cache is not None
                    and len(candidate_pairs) > 0):
                before = len(candidate_pairs)
                candidate_pairs = self._document_vector_filter(
                    candidate_pairs, cache, self.config.DOC_VECTOR_THRESHOLD
                )
                logger.info(
                    f"Level 3 文档向量预筛: {before} → {len(candidate_pairs)} 对 "
                    f"(阈值={self.config.DOC_VECTOR_THRESHOLD})"
                )

            # Level 4: 图片哈希匹配
            image_pairs = self._image_hash_matching(all_features)
            logger.info(f"Level 4 图片哈希: {len(image_pairs)} 对")
            for pair in image_pairs:
                if pair not in candidate_pairs:
                    candidate_pairs[pair] = ("image", 0.0)

            # 回退
            if not candidate_pairs:
                logger.warning("所有筛选方法均未找到候选对，回退到全量比对")
                for i in range(len(valid_features)):
                    for j in range(i + 1, len(valid_features)):
                        pair = tuple(sorted([
                            valid_features[i].doc_id, valid_features[j].doc_id
                        ]))
                        candidate_pairs[pair] = ("fallback", 0.0)

        result = [
            (id_a, id_b, method, sim)
            for (id_a, id_b), (method, sim) in candidate_pairs.items()
        ]
        result.sort(key=lambda x: x[3], reverse=True)

        logger.info(f"候选选择完成: {len(result)} 对")
        return result

    # ================================================================
    # 新增筛选方法
    # ================================================================

    def _metadata_fingerprint_filter(self, cache) -> Set[Tuple[str, str]]:
        """元数据指纹匹配：software_fingerprint 或 time_bucket 相同的文档对

        从 metadata_fingerprints 表批量查询，强制将同源文档加入候选。
        """
        fingerprints = cache.load_metadata_fingerprints()
        if not fingerprints:
            return set()

        # 按 software_fingerprint 分组
        sw_groups = defaultdict(list)
        tb_groups = defaultdict(list)
        for doc_id, fp in fingerprints.items():
            sw = fp.get('software_fingerprint', '')
            tb = fp.get('time_bucket', '')
            if sw:
                sw_groups[sw].append(doc_id)
            if tb:
                tb_groups[tb].append(doc_id)

        pairs = set()
        # 同一软件指纹的文档对
        for group in sw_groups.values():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    pairs.add(tuple(sorted([group[i], group[j]])))
        # 同一时间桶的文档对
        for group in tb_groups.values():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    pairs.add(tuple(sorted([group[i], group[j]])))

        return pairs

    def _document_vector_filter(
        self,
        candidate_pairs: dict,
        cache,
        threshold: float = 0.3,
    ) -> dict:
        """文档向量余弦预筛：过滤文档级不相似的对

        加载 document_embeddings 表中的文档级向量，
        对候选对计算余弦相似度，低于阈值的移除。
        """
        doc_embeddings = cache.load_all_document_embeddings()
        if not doc_embeddings:
            logger.warning("无文档嵌入缓存，跳过向量预筛")
            return candidate_pairs

        filtered = {}
        for (doc_a_id, doc_b_id), (method, sim) in candidate_pairs.items():
            emb_a = doc_embeddings.get(doc_a_id)
            emb_b = doc_embeddings.get(doc_b_id)

            if emb_a is not None and emb_b is not None:
                # 余弦相似度
                norm_a = np.linalg.norm(emb_a)
                norm_b = np.linalg.norm(emb_b)
                if norm_a > 0 and norm_b > 0:
                    cos_sim = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
                    if cos_sim >= threshold:
                        filtered[(doc_a_id, doc_b_id)] = (
                            method, max(float(sim), cos_sim)
                        )
                    # else: 过滤掉 — 文档级向量不相似
                else:
                    filtered[(doc_a_id, doc_b_id)] = (method, sim)
            else:
                # 缺少嵌入 → 保留（安全回退）
                filtered[(doc_a_id, doc_b_id)] = (method, sim)

        return filtered

    def _datasketch_lsh(
        self, features: List[BidFeature],
    ) -> dict:
        """使用 datasketch MinHashLSH 进行文档级筛选"""
        if not features:
            return {}

        try:
            lsh_index = DocLevelLSHIndex(self.config)
            pairs = lsh_index.query_all_pairs(features)
            return {pair: 0.5 for pair in pairs}  # LSH 命中的估计相似度
        except ImportError:
            raise

    def _simhash_fallback(
        self, features: List[BidFeature],
    ) -> Set[Tuple[str, str]]:
        """SimHash 汉明距离回退方案（保留原有逻辑）"""
        pairs = set()
        valid = [f for f in features if f.text_simhash]

        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                hamming = self._hamming_distance(
                    valid[i].text_simhash, valid[j].text_simhash
                )
                if hamming <= 3:
                    pair = tuple(sorted([valid[i].doc_id, valid[j].doc_id]))
                    pairs.add(pair)

        return pairs

    def _hamming_distance(self, hash1: str, hash2: str) -> int:
        """计算 SimHash 汉明距离"""
        try:
            int1 = int(hash1, 16)
            int2 = int(hash2, 16)
            return bin(int1 ^ int2).count('1')
        except Exception:
            return 64

    def _metadata_clustering(
        self, features: List[BidFeature],
    ) -> Set[Tuple[str, str]]:
        """基于元数据的快速聚类（保留原有逻辑）"""
        pairs = set()
        software_index = defaultdict(list)
        time_bucket_index = defaultdict(list)

        for f in features:
            if f.metadata.software_fingerprint:
                software_index[f.metadata.software_fingerprint].append(f.doc_id)
            if f.metadata.time_bucket:
                time_bucket_index[f.metadata.time_bucket].append(f.doc_id)

        # 同一软件指纹
        for docs in software_index.values():
            if len(docs) > 1:
                for i in range(len(docs)):
                    for j in range(i + 1, len(docs)):
                        pairs.add(tuple(sorted([docs[i], docs[j]])))

        # 同一时间桶
        for docs in time_bucket_index.values():
            if len(docs) > 1:
                for i in range(len(docs)):
                    for j in range(i + 1, len(docs)):
                        pairs.add(tuple(sorted([docs[i], docs[j]])))

        return pairs

    def _image_hash_matching(
        self, features: List[BidFeature],
    ) -> Set[Tuple[str, str]]:
        """基于图片哈希的匹配（保留原有逻辑）"""
        pairs = set()
        image_index = defaultdict(list)

        for f in features:
            for img_hash in f.image_hashes:
                image_index[img_hash].append(f.doc_id)

        for docs in image_index.values():
            if len(docs) > 1:
                for i in range(len(docs)):
                    for j in range(i + 1, len(docs)):
                        pairs.add(tuple(sorted([docs[i], docs[j]])))

        return pairs
