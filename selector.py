"""
模块 B：快速初筛引擎
"""
import logging
from typing import List, Tuple, Set, Dict
from collections import defaultdict

from data_structures import BidFeature
from config import DetectionConfig

logger = logging.getLogger(__name__)


class CandidatePairSelector:
    """候选文档对选择器"""

    def __init__(self, config: DetectionConfig):
        self.config = config

    def select(self, features: List[BidFeature]) -> List[Tuple[str, str]]:
        """选择候选文档对 - 返回所有文档对进行比较"""
        candidate_pairs = set()

        valid_features = [f for f in features if not f.is_scanned]

        for i in range(len(valid_features)):
            for j in range(i + 1, len(valid_features)):
                pair_id = tuple(sorted([valid_features[i].doc_id, valid_features[j].doc_id]))
                candidate_pairs.add(pair_id)

        candidate_list = sorted(list(candidate_pairs))

        logger.info(f"候选对数: {len(candidate_list)}")
        return candidate_list

    def _simhash_screening(self, features: List[BidFeature]) -> Set[Tuple[str, str]]:
        """基于SimHash汉明距离的初筛"""
        pairs = set()

        # 只处理非扫描版文档
        valid_features = [f for f in features if not f.is_scanned and f.text_simhash]

        for i in range(len(valid_features)):
            for j in range(i + 1, len(valid_features)):
                doc_a = valid_features[i]
                doc_b = valid_features[j]

                # 计算汉明距离
                hamming_dist = self._hamming_distance(doc_a.text_simhash, doc_b.text_simhash)

                # 汉明距离 <= 3 认为高度相似
                if hamming_dist <= 3:
                    pair_id = tuple(sorted([doc_a.doc_id, doc_b.doc_id]))
                    pairs.add(pair_id)

        return pairs

    def _hamming_distance(self, hash1: str, hash2: str) -> int:
        """计算汉明距离（针对十六进制字符串）"""
        try:
            # 将十六进制字符串转换为整数
            int1 = int(hash1, 16)
            int2 = int(hash2, 16)
            # 异或运算
            xor = int1 ^ int2
            # 计算二进制中1的个数
            return bin(xor).count('1')
        except:
            return 64

    def _lsh_bucketing(self, features: List[BidFeature]) -> Set[Tuple[str, str]]:
        """LSH桶分桶"""
        pairs = set()

        # 只处理非扫描版文档
        valid_features = [f for f in features if not f.is_scanned and f.text_simhash]

        # 将SimHash分为多个band
        bands = self.config.SIMHASH_BANDS
        rows_per_band = self.config.SIMHASH_ROWS

        # 对每个band建立桶
        for band_idx in range(bands):
            buckets = defaultdict(list)

            for feature in valid_features:
                simhash = feature.text_simhash
                # 提取当前band的部分
                start = band_idx * rows_per_band
                end = start + rows_per_band
                band_hash = simhash[start:end] if start < len(simhash) else ""

                if band_hash:
                    buckets[band_hash].append(feature.doc_id)

            # 同一桶内的文档两两配对
            for bucket_docs in buckets.values():
                if len(bucket_docs) > 1:
                    for i in range(len(bucket_docs)):
                        for j in range(i + 1, len(bucket_docs)):
                            pair_id = tuple(sorted([bucket_docs[i], bucket_docs[j]]))
                            pairs.add(pair_id)

        return pairs

    def _metadata_clustering(self, features: List[BidFeature]) -> Set[Tuple[str, str]]:
        """基于元数据的快速聚类"""
        pairs = set()

        # 构建倒排索引：software_fingerprint -> 文档列表
        software_index = defaultdict(list)
        time_bucket_index = defaultdict(list)

        for feature in features:
            metadata = feature.metadata

            if metadata.software_fingerprint:
                software_index[metadata.software_fingerprint].append(feature.doc_id)

            if metadata.time_bucket:
                time_bucket_index[metadata.time_bucket].append(feature.doc_id)

        # 同一软件指纹的文档配对
        for docs in software_index.values():
            if len(docs) > 1:
                for i in range(len(docs)):
                    for j in range(i + 1, len(docs)):
                        pair_id = tuple(sorted([docs[i], docs[j]]))
                        pairs.add(pair_id)

        # 同一时间桶的文档配对
        for docs in time_bucket_index.values():
            if len(docs) > 1:
                for i in range(len(docs)):
                    for j in range(i + 1, len(docs)):
                        pair_id = tuple(sorted([docs[i], docs[j]]))
                        pairs.add(pair_id)

        return pairs

    def _image_hash_matching(self, features: List[BidFeature]) -> Set[Tuple[str, str]]:
        """基于图片哈希的匹配"""
        pairs = set()

        # 构建图片哈希 -> 文档列表映射
        image_index = defaultdict(list)

        for feature in features:
            for img_hash in feature.image_hashes:
                image_index[img_hash].append(feature.doc_id)

        # 共享同一图片的文档配对
        for docs in image_index.values():
            if len(docs) > 1:
                for i in range(len(docs)):
                    for j in range(i + 1, len(docs)):
                        pair_id = tuple(sorted([docs[i], docs[j]]))
                        pairs.add(pair_id)

        return pairs
