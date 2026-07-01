"""
候选文档对选择器 — 多层级筛选策略

替换原有的 SimHash O(n²) 遍历 + LSH 分桶:
- Level 1: datasketch MinHashLSH 文档级筛选（O(n)）
- Level 2: 元数据聚类（保留原有逻辑）
- Level 3: 图片哈希匹配（保留原有逻辑）

支持回退到全量比对（当文档数较小时）
"""

import logging
from typing import List, Tuple, Set
from collections import defaultdict

from data_structures import BidFeature
from config import DetectionConfig
from matching.lsh_index import DocLevelLSHIndex

logger = logging.getLogger(__name__)


class CandidatePairSelector:
    """候选文档对选择器（多层级筛选）"""

    def __init__(self, config: DetectionConfig):
        self.config = config

    def select(
        self, features: List[BidFeature],
    ) -> List[Tuple[str, str, str, float]]:
        """选择候选文档对

        返回带方法和相似度的候选对，用于后续精细分析和优先级排序。

        Returns:
            [(doc_a_id, doc_b_id, method, similarity), ...]
        """
        candidate_pairs: dict = {}  # (id_a, id_b) -> (method, similarity)

        valid_features = [f for f in features if not f.is_scanned]
        all_features = features  # 包含扫描版用于元数据和图片匹配

        # 小规模文档集直接全量比对
        if len(valid_features) <= 5:
            logger.info(
                f"文档数较少 ({len(valid_features)} 个)，直接全量比对"
            )
            for i in range(len(valid_features)):
                for j in range(i + 1, len(valid_features)):
                    pair = tuple(sorted([valid_features[i].doc_id, valid_features[j].doc_id]))
                    candidate_pairs[pair] = ("all_pairs", 0.0)
        else:
            # Level 1: datasketch MinHashLSH
            try:
                lsh_pairs = self._datasketch_lsh(valid_features)
                logger.info(f"datasketch LSH 初筛: {len(lsh_pairs)} 对")
                for pair, sim in lsh_pairs.items():
                    if pair not in candidate_pairs:
                        candidate_pairs[pair] = ("lsh", sim)
            except ImportError:
                logger.warning("datasketch 不可用，使用 SimHash 回退方案")
                simhash_pairs = self._simhash_fallback(valid_features)
                for pair in simhash_pairs:
                    if pair not in candidate_pairs:
                        candidate_pairs[pair] = ("simhash", 0.0)

            # Level 2: 元数据聚类
            metadata_pairs = self._metadata_clustering(all_features)
            logger.info(f"元数据聚类: {len(metadata_pairs)} 对")
            for pair in metadata_pairs:
                if pair not in candidate_pairs:
                    candidate_pairs[pair] = ("metadata", 0.0)

            # Level 3: 图片哈希匹配
            image_pairs = self._image_hash_matching(all_features)
            logger.info(f"图片哈希匹配: {len(image_pairs)} 对")
            for pair in image_pairs:
                if pair not in candidate_pairs:
                    candidate_pairs[pair] = ("image", 0.0)

            # 如果所有方法都没找到候选对，回退到全量比对
            if not candidate_pairs:
                logger.warning("所有筛选方法均未找到候选对，回退到全量比对")
                for i in range(len(valid_features)):
                    for j in range(i + 1, len(valid_features)):
                        pair = tuple(sorted([valid_features[i].doc_id, valid_features[j].doc_id]))
                        candidate_pairs[pair] = ("fallback", 0.0)

        # 按相似度排序
        result = [
            (id_a, id_b, method, sim)
            for (id_a, id_b), (method, sim) in candidate_pairs.items()
        ]
        result.sort(key=lambda x: x[3], reverse=True)

        logger.info(f"候选选择完成: {len(result)} 对")
        return result

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
