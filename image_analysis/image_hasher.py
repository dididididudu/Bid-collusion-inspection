"""
图片哈希增强 — pHash/dHash 汉明距离 + 模糊匹配

扩展原有 imagehash 库的精确匹配为基于汉明距离的相似度匹配。
"""

import logging
from typing import List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class HashMatchResult:
    """哈希匹配结果"""
    hash_a: str
    hash_b: str
    hamming_dist: int       # 汉明距离（0 = 完全相同）
    is_exact: bool          # 是否完全相同
    is_similar: bool        # 是否相似（汉明距离 ≤ 阈值）


class ImageHasher:
    """图片哈希比较器 — 汉明距离 + 模糊匹配"""

    # 汉明距离阈值
    EXACT_THRESHOLD = 0     # 完全相同
    NEAR_IDENTICAL = 5      # 几乎相同（微小压缩/缩放差异）
    SIMILAR = 10            # 相似（同一图片的不同版本）

    def __init__(self):
        pass

    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> int:
        """计算两个十六进制哈希字符串的汉明距离

        支持 pHash 和 dHash（均为 64 位 = 16 个十六进制字符）。

        Args:
            hash1, hash2: 十六进制哈希字符串（如 "0f1e2d3c4b5a6978"）
                         支持带前缀格式（如 "page_5:p0f1e2d3c4b5a6978"）

        Returns:
            汉明距离（不同位数）
        """
        # 剥离前缀（如有）
        h1 = hash1.split(':')[-1] if ':' in hash1 else hash1
        h2 = hash2.split(':')[-1] if ':' in hash2 else hash2

        if len(h1) != len(h2):
            # 补齐到相同长度
            max_len = max(len(h1), len(h2))
            h1 = h1.zfill(max_len)
            h2 = h2.zfill(max_len)

        try:
            int1 = int(h1, 16)
            int2 = int(h2, 16)
            xor = int1 ^ int2
            return bin(xor).count('1')
        except (ValueError, TypeError):
            # 非十六进制字符串，回退到字符比较
            return sum(c1 != c2 for c1, c2 in zip(h1, h2))

    def match_hashes(
        self,
        hashes_a: List[str],
        hashes_b: List[str],
    ) -> List[HashMatchResult]:
        """在两个哈希列表之间找匹配的图片对

        对每对哈希计算汉明距离，返回所有满足阈值的匹配。

        Args:
            hashes_a: 文档A的图片哈希列表
            hashes_b: 文档B的图片哈希列表

        Returns:
            匹配结果列表（按汉明距离升序 = 越相似越靠前）
        """
        results = []

        for ha in hashes_a:
            for hb in hashes_b:
                dist = self.hamming_distance(ha, hb)
                if dist <= self.SIMILAR:
                    results.append(HashMatchResult(
                        hash_a=ha,
                        hash_b=hb,
                        hamming_dist=dist,
                        is_exact=(dist == self.EXACT_THRESHOLD),
                        is_similar=(dist <= self.NEAR_IDENTICAL),
                    ))

        results.sort(key=lambda x: x.hamming_dist)
        return results
