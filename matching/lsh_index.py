"""
datasketch MinHash LSH 索引封装

替换手写 SimHash/LSH 分桶为专业的 datasketch 库实现:
- 文档级 LSH: 用于候选对快速初筛
- 段落级 LSH: 按需临时构建，用完即释放（不再维护全局段落索引）

内存优化:
- 旧方案: 全局段落 LSH 索引 = 100K 段落 × 8 band = 80万条目
- 新方案: 按需构建，每次仅 ~200 条目（单文档对）
"""

import logging
from typing import List, Dict, Set, Tuple, Optional

import numpy as np

try:
    from datasketch import MinHash, MinHashLSH
    DATASKETCH_AVAILABLE = True
except ImportError:
    DATASKETCH_AVAILABLE = False

from config import DetectionConfig

logger = logging.getLogger(__name__)


class DocLevelLSHIndex:
    """文档级 MinHash LSH 索引

    用于候选对初筛，替代原有的 SimHash 汉明距离 + LSH 分桶两层筛选。
    使用 datasketch.MinHashLSH 实现，O(n) 插入 + O(1) 查询。
    """

    def __init__(self, config: DetectionConfig):
        if not DATASKETCH_AVAILABLE:
            raise ImportError(
                "datasketch 未安装，请运行: pip install datasketch"
            )
        self.config = config
        self.threshold = config.MINHASH_LSH_THRESHOLD
        self.num_perm = config.MINHASH_NUM_HASHES
        self.lsh = MinHashLSH(
            threshold=self.threshold,
            num_perm=self.num_perm,
            weights=(0.5, 0.5),  # 平衡精确度和召回率
        )
        self._inserted = set()
        self._mh_cache = {}  # 缓存 MinHash 对象，避免重复构建
        logger.info(
            f"文档级 LSH 索引已创建: threshold={self.threshold}, "
            f"num_perm={self.num_perm}"
        )

    def _build_minhash(self, key: str, minhash_values: List[int]) -> MinHash:
        """构建或从缓存获取 MinHash 对象"""
        if key not in self._mh_cache:
            mh = MinHash(num_perm=self.num_perm)
            for v in minhash_values:
                mh.update(str(v).encode('utf-8'))
            self._mh_cache[key] = mh
        return self._mh_cache[key]

    def insert(self, doc_id: str, minhash_values: List[int]) -> bool:
        """插入文档的 MinHash 签名到索引"""
        if not minhash_values or doc_id in self._inserted:
            return False

        mh = self._build_minhash(doc_id, minhash_values)
        self.lsh.insert(doc_id, mh)
        self._inserted.add(doc_id)
        return True

    def query(self, doc_id: str, minhash_values: List[int]) -> Set[str]:
        """查询与指定文档相似的文档"""
        if not minhash_values:
            return set()

        mh = self._build_minhash(doc_id, minhash_values)
        results = self.lsh.query(mh)
        return {r for r in results if r != doc_id}

    def query_all_pairs(self, features: List) -> Set[Tuple[str, str]]:
        """查询所有特征中的相似文档对"""
        # 先插入所有文档
        for f in features:
            if f.doc_minhash and not f.is_scanned:
                self.insert(f.doc_id, f.doc_minhash)

        # 查询每对
        pairs = set()
        for f in features:
            if f.is_scanned or not f.doc_minhash:
                continue
            matches = self.query(f.doc_id, f.doc_minhash)
            for match_id in matches:
                pair = tuple(sorted([f.doc_id, match_id]))
                pairs.add(pair)

        logger.info(f"LSH 查询完成: {len(pairs)} 对候选")
        return pairs

    def __len__(self):
        return len(self._inserted)


class ParagraphLevelLSH:
    """段落级 MinHash LSH 索引（按需构建，用完释放）

    仅在分析单个文档对时临时构建 doc_b 的段落索引，
    然后用 doc_a 的段落查询。分析完成后释放。

    相比旧方案的内存节省:
    - 旧: 全局 100K 段落 × 8 band = 80万条目
    - 新: 每次 ~200 段落 × 4 band = 800 条目
    """

    def __init__(self, config: DetectionConfig):
        if not DATASKETCH_AVAILABLE:
            raise ImportError("datasketch 未安装")
        self.config = config
        self.threshold = config.PARAGRAPH_LSH_THRESHOLD
        self.num_perm = config.MINHASH_NUM_HASHES_PARAGRAPH

    def build_and_query(
        self,
        paras_a: List[Dict],
        paras_b: List[Dict],
    ) -> List[Tuple[int, int, float]]:
        """构建 doc_b 的段落 LSH 索引，用 doc_a 查询

        Args:
            paras_a: doc_a 段落列表 [{'para_index': int, 'minhash': str}, ...]
            paras_b: doc_b 段落列表 [{'para_index': int, 'minhash': str}, ...]

        Returns:
            [(para_a_index, para_b_index, estimated_jaccard), ...]
            按估计 Jaccard 降序排列
        """
        if len(paras_a) == 0 or len(paras_b) == 0:
            return []

        # 构建 doc_b 的 LSH 索引
        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)

        para_b_map = {}  # LSH key -> para_index
        b_minhashes = {}
        mh_cache = {}  # 缓存 MinHash 对象：key_str -> MinHash

        def get_cached_minhash(key_str, values):
            """构建或从缓存获取 MinHash 对象"""
            if key_str not in mh_cache:
                mh = MinHash(num_perm=self.num_perm)
                for v in values[:self.num_perm]:
                    mh.update(str(v).encode('utf-8'))
                mh_cache[key_str] = mh
            return mh_cache[key_str]

        for para in paras_b:
            minhash_str = para.get('minhash', '')
            if not minhash_str:
                continue
            values = [int(v) for v in minhash_str.split(',')]
            if len(values) < self.num_perm:
                continue

            key = f"b_{para['para_index']}"
            mh = get_cached_minhash(key, values)
            lsh.insert(key, mh)
            para_b_map[key] = para['para_index']
            b_minhashes[para['para_index']] = np.array(values[:self.num_perm])

        # 用 doc_a 的每个段落查询
        candidates = {}

        for para in paras_a:
            minhash_str = para.get('minhash', '')
            if not minhash_str:
                continue
            values = [int(v) for v in minhash_str.split(',')]
            if len(values) < self.num_perm:
                continue

            key = f"a_{para['para_index']}"
            mh = get_cached_minhash(key, values)

            results = lsh.query(mh)
            for result_key in results:
                b_idx = para_b_map.get(result_key)
                if b_idx is not None:
                    pair = (para['para_index'], b_idx)
                    if pair not in candidates:
                        # 估算 Jaccard 相似度
                        a_vals = np.array(values[:self.num_perm])
                        b_vals = b_minhashes.get(b_idx)
                        if b_vals is not None and len(a_vals) == len(b_vals):
                            jaccard = float(np.mean(a_vals == b_vals))
                        else:
                            jaccard = 0.3  # 保守估计
                        candidates[pair] = jaccard

        # 按估计 Jaccard 排序
        result = sorted(
            [(i, j, sim) for (i, j), sim in candidates.items()],
            key=lambda x: x[2],
            reverse=True
        )

        logger.debug(
            f"段落 LSH: {len(paras_a)}×{len(paras_b)} → {len(result)} 候选"
        )
        return result


def minhash_from_values(values: List[int], num_perm: int = 128) -> 'MinHash':
    """辅助函数：从整数列表创建 datasketch MinHash 对象"""
    if not DATASKETCH_AVAILABLE:
        return None
    mh = MinHash(num_perm=num_perm)
    for v in values:
        mh.update(str(v).encode('utf-8'))
    return mh


def estimate_jaccard(minhash_a: List[int], minhash_b: List[int]) -> float:
    """从两个 MinHash 签名估算 Jaccard 相似度

    Args:
        minhash_a: 文档 A 的 MinHash 签名
        minhash_b: 文档 B 的 MinHash 签名

    Returns:
        估计的 Jaccard 相似度 (0.0 ~ 1.0)
    """
    if not minhash_a or not minhash_b:
        return 0.0

    if len(minhash_a) != len(minhash_b):
        return 0.0

    matches = sum(1 for a, b in zip(minhash_a, minhash_b) if a == b)
    return matches / len(minhash_a)
