"""
图片哈希增强 — 三级级联匹配（哈希 + ORB + 直方图）

第一级：多哈希共识（pHash + dHash + 长宽比）      ← 快速预筛
第二级：ORB 特征匹配                               ← 确认结构一致性
第三级：直方图相关性                                ← 确认颜色一致性

三级都通过才判为匹配，大幅降低"布局相似但内容不同"或"颜色不同"的误报。
"""

import logging
import re
import io
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ================================================================
# ORB 特征匹配（全局共享，cv2 可能无 GPU 但特征匹配量很小）
# ================================================================

_ORB_DETECTORS = {}

def _get_orb(nfeatures=200):
    """获取 ORB 检测器（按 nfeatures 分级缓存 — 方案4）

    根据需要的特征点数缓存多个 ORB 实例，避免重复创建。
    """
    global _ORB_DETECTORS
    if nfeatures not in _ORB_DETECTORS:
        try:
            import cv2
            # scaleFactor=1.2: 金字塔尺度
            # fastThreshold=10: 降低阈值以在低纹理图上也能检测到关键点
            _ORB_DETECTORS[nfeatures] = cv2.ORB_create(
                nfeatures=nfeatures, scaleFactor=1.2, fastThreshold=10
            )
        except ImportError:
            logger.warning("OpenCV (cv2) 未安装，ORB 特征匹配不可用")
            _ORB_DETECTORS[nfeatures] = False
    return _ORB_DETECTORS[nfeatures] if _ORB_DETECTORS[nfeatures] is not False else None


# ================================================================
# 基础数据结构
# ================================================================


@dataclass
class HashMatchResult:
    """单哈希匹配结果（保留向后兼容）"""
    hash_a: str
    hash_b: str
    hamming_dist: int       # 汉明距离（0 = 完全相同）
    is_exact: bool          # 是否完全相同
    is_similar: bool        # 是否相似（汉明距离 ≤ 阈值）


@dataclass
class ImageSignature:
    """一张图片的完整签名（三级级联所需全部信息）"""
    source_id: str = ""             # 来源标识（如 "page_5" 或 "xref_7"）
    page_num: int = -1              # 页码

    # L1: 多哈希指纹
    phash: str = ""                 # 感知哈希
    dhash: str = ""                 # 差异哈希
    whash: str = ""                 # 小波哈希（备用）

    # L2+L3: 缩略图（JPEG 字节流）
    thumbnail: bytes = b""          # 64×64 JPEG 缩略图

    # 图片元数据
    width: int = 0                  # 图片宽度（像素）
    height: int = 0                 # 图片高度（像素）

    # 原始哈希字符串（未拆分的，用于展示）
    raw_hashes: List[str] = None

    # ORB 特征缓存（懒加载）
    _orb_kp: list = None
    _orb_des: np.ndarray = None

    def __post_init__(self):
        if self.raw_hashes is None:
            self.raw_hashes = []
        # 方案5：直方图缓存初始化（lazy 计算）
        self._histogram_cache = None

    @property
    def aspect_ratio(self) -> float:
        """长宽比，用于尺寸合理性校验"""
        if self.height == 0:
            return 0.0
        return self.width / self.height

    @property
    def has_any_hash(self) -> bool:
        return bool(self.phash or self.dhash or self.whash)

    @property
    def has_thumbnail(self) -> bool:
        return bool(self.thumbnail and len(self.thumbnail) > 100)

    def get_orb_features(self, nfeatures=None):
        """懒加载计算 ORB 特征

        Args:
            nfeatures: 期望的特征点数，None 表示使用默认值 200

        Returns:
            (keypoints, descriptors) or (None, None) if unavailable
        """
        if self._orb_des is not None:
            return self._orb_kp, self._orb_des
        if not self.has_thumbnail:
            return None, None

        orb = _get_orb(nfeatures=nfeatures or 200)
        if orb is None:
            return None, None

        try:
            img = Image.open(io.BytesIO(self.thumbnail))
            img_array = np.array(img.convert('RGB'))
            # cv2 需要 BGR
            import cv2
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            kp, des = orb.detectAndCompute(img_bgr, None)
            self._orb_kp = kp
            self._orb_des = des
            return kp, des
        except Exception as e:
            logger.debug(f"ORB 特征计算失败: {e}")
            return None, None

    def get_histogram(self) -> Optional[np.ndarray]:
        """获取缩略图的归一化颜色直方图（缓存版 — 方案5）

        第一次调用时计算并缓存，后续 O(1) 返回。
        Returns:
            shape (24,) 的 numpy 数组（RGB 各 8 bins）或 None
        """
        if self._histogram_cache is not None:
            return self._histogram_cache
        if not self.has_thumbnail:
            return None
        self._histogram_cache = self._compute_histogram()
        return self._histogram_cache

    def _compute_histogram(self) -> Optional[np.ndarray]:
        """计算缩略图的归一化颜色直方图"""
        try:
            img = Image.open(io.BytesIO(self.thumbnail)).convert('RGB')
            arr = np.array(img)  # (H, W, 3)
            hist_r = np.histogram(arr[:, :, 0], bins=8, range=(0, 256))[0]
            hist_g = np.histogram(arr[:, :, 1], bins=8, range=(0, 256))[0]
            hist_b = np.histogram(arr[:, :, 2], bins=8, range=(0, 256))[0]
            hist = np.concatenate([hist_r, hist_g, hist_b]).astype(np.float64)
            s = hist.sum()
            if s > 0:
                hist /= s
            return hist
        except Exception as e:
            logger.debug(f"直方图计算失败: {e}")
            return None


@dataclass
class ImageMatchVerdict:
    """两张图片的完整匹配判决（三级级联结果）"""
    sig_a: ImageSignature
    sig_b: ImageSignature

    # L1: 各哈希类型的距离
    phash_dist: int = -1
    dhash_dist: int = -1
    whash_dist: int = -1

    # L2: ORB 特征匹配结果
    orb_match_count: int = 0
    orb_total_kp: int = 0
    orb_match_ratio: float = 0.0     # 0~1

    # L3: 直方图相关性
    histogram_correlation: float = -1.0  # -1~1

    # 长宽比
    aspect_ratio_a: float = 0.0
    aspect_ratio_b: float = 0.0

    # 各级判决
    l1_pass: bool = False             # 哈希层通过
    l2_pass: bool = False             # ORB 层通过
    l3_pass: bool = False             # 直方图层通过

    # 最终判决
    is_match: bool = False
    confidence: float = 0.0           # 0.0 ~ 1.0
    reasons: List[str] = None

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []


class ImageHasher:
    """图片哈希比较器 — 多哈希共识 + 图像签名匹配

    相比旧版单哈希匹配，改进点：
    1. 将属于同一张图片的多个 hash 聚合为 ImageSignature
    2. 两张图片匹配要求多个哈希类型同时达标
    3. 加入尺寸/长宽比过滤
    4. 输出置信度而非简单的 0/1
    """

    # 汉明距离阈值
    EXACT_DIST = 0          # 完全相同
    NEAR_IDENTICAL_DIST = 5 # 几乎相同
    SIMILAR_DIST = 10       # 相似

    # 长宽比容差（比值差异超过此值则不能是同一张图）
    ASPECT_RATIO_TOLERANCE = 0.15  # 15%

    def __init__(self):
        pass

    # ================================================================
    # 哈希字符串解析
    # ================================================================

    HASH_PREFIX_PATTERN = re.compile(
        r'^(?:page_(\d+):)?'            # 可选页码前缀
        r'([pdw])'                       # 哈希类型: p=phash, d=dhash, w=whash
        r'([0-9a-fA-F]+)$'              # 哈希值（十六进制）
    )

    @classmethod
    def parse_hash_string(cls, hash_str: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """解析哈希字符串，返回 (page_num_or_none, hash_type, hash_value)

        Examples:
            "page_5:p0f1e2" → ("5", "p", "0f1e2")
            "d0f1e2"       → (None, "d", "0f1e2")
            "p0f1e2"       → (None, "p", "0f1e2")
        """
        # 先尝试带命名空间前缀的格式
        if ':' in hash_str:
            prefix, rest = hash_str.split(':', 1)
            # page_N 格式
            if prefix.startswith('page_'):
                page_num = prefix[5:]
                if rest and rest[0] in ('p', 'd', 'w') and len(rest) > 1:
                    return page_num, rest[0], rest[1:]
                return page_num, None, rest
            # 其他前缀
            return prefix, None, rest

        # 无前缀，直接解析类型
        if hash_str and hash_str[0] in ('p', 'd', 'w') and len(hash_str) > 1:
            return None, hash_str[0], hash_str[1:]

        # 兼容旧数据库：无前缀的16位十六进制字符串视为 pHash
        if hash_str and len(hash_str) == 16:
            try:
                int(hash_str, 16)  # 验证是否为合法十六进制
                return None, 'p', hash_str
            except ValueError:
                pass

        return None, None, hash_str

    @classmethod
    def hamming_distance(cls, hash1: str, hash2: str) -> int:
        """计算两个十六进制哈希字符串的汉明距离"""
        h1 = hash1.split(':')[-1] if ':' in hash1 else hash1
        h2 = hash2.split(':')[-1] if ':' in hash2 else hash2

        # 跳过类型前缀字符（如 p, d, w）
        content1 = h1[1:] if h1 and h1[0] in ('p', 'd', 'w') else h1
        content2 = h2[1:] if h2 and h2[0] in ('p', 'd', 'w') else h2

        if len(content1) != len(content2):
            max_len = max(len(content1), len(content2))
            content1 = content1.zfill(max_len)
            content2 = content2.zfill(max_len)

        try:
            int1 = int(content1, 16)
            int2 = int(content2, 16)
            xor = int1 ^ int2
            return bin(xor).count('1')
        except (ValueError, TypeError):
            return sum(c1 != c2 for c1, c2 in zip(content1, content2))

    @classmethod
    def parse_hashes_into_signatures(
        cls, hash_strings: List[str],
        width_map: Dict[str, int] = None,
        height_map: Dict[str, int] = None,
    ) -> List[ImageSignature]:
        """将扁平哈希列表解析为按图片分组的签名列表

        相同 page_N 前缀的 hashes 属于同一张图片（扫描版）。
        无前缀的每个 hash 独立为一张图片（嵌入位图）。

        Args:
            hash_strings: 原始哈希字符串列表
            width_map: 可选的 source_id → 宽度映射
            height_map: 可选的 source_id → 高度映射

        Returns:
            按图片分组的签名列表
        """
        if width_map is None:
            width_map = {}
        if height_map is None:
            height_map = {}

        # 第一步：按 page 分组
        page_groups: Dict[str, Dict[str, str]] = {}
        standalone: List[Dict[str, str]] = []

        for hs in hash_strings:
            page_num, htype, hval = cls.parse_hash_string(hs)
            if page_num is not None:
                # 有页码 → 属于某页的所有 hash（可能有多个）
                key = f"page_{page_num}"
                if key not in page_groups:
                    page_groups[key] = {'_page': page_num}
                if htype:
                    if htype == 'p':
                        page_groups[key]['phash'] = hval
                    elif htype == 'd':
                        page_groups[key]['dhash'] = hval
                    elif htype == 'w':
                        page_groups[key]['whash'] = hval
                    page_groups[key].setdefault('raw', []).append(hs)
                else:
                    page_groups[key].setdefault('raw', []).append(hs)
            else:
                # 嵌入位图
                entry = {'_page': '-1'}
                if htype == 'p':
                    entry['phash'] = hval
                elif htype == 'd':
                    entry['dhash'] = hval
                elif htype == 'w':
                    entry['whash'] = hval
                entry.setdefault('raw', []).append(hs)
                standalone.append(entry)

        # 第二步：构建签名
        signatures = []

        for key, group in page_groups.items():
            sig = ImageSignature(
                source_id=key,
                page_num=int(group.get('_page', -1)),
                phash=group.get('phash', ''),
                dhash=group.get('dhash', ''),
                whash=group.get('whash', ''),
                raw_hashes=group.get('raw', []),
                width=width_map.get(key, 0),
                height=height_map.get(key, 0),
            )
            signatures.append(sig)

        for entry in standalone:
            sig = ImageSignature(
                source_id='',
                page_num=-1,
                phash=entry.get('phash', ''),
                dhash=entry.get('dhash', ''),
                whash=entry.get('whash', ''),
                raw_hashes=entry.get('raw', []),
                width=width_map.get('', 0),
                height=height_map.get('', 0),
            )
            signatures.append(sig)

        return signatures

    # ================================================================
    # 单图片对匹配判决
    # ================================================================

    def compare_signatures(
        self,
        sig_a: ImageSignature,
        sig_b: ImageSignature,
    ) -> ImageMatchVerdict:
        "三级级联匹配判决"
        verdict = ImageMatchVerdict(
            sig_a=sig_a, sig_b=sig_b,
            aspect_ratio_a=sig_a.aspect_ratio,
            aspect_ratio_b=sig_b.aspect_ratio,
        )

        # L1: 哈希距离 + 长宽比
        verdict = self._verify_l1_hash(sig_a, sig_b, verdict)
        if not verdict.l1_pass:
            return verdict

        # 方案2a：L1 通过且距离极小（≤2）→ 跳过 L2/L3
        max_l1_dist = max(
            d for d in (verdict.phash_dist, verdict.dhash_dist, verdict.whash_dist)
            if d >= 0
        )
        if max_l1_dist <= 2:
            verdict.confidence = min(0.95, verdict.confidence + 0.2)
            verdict.l2_pass = True
            verdict.l3_pass = True
            verdict.reasons.append(
                f"L1 距离极小(max={max_l1_dist}) → 跳过 L2/L3"
            )
            return verdict

        if not (sig_a.has_thumbnail and sig_b.has_thumbnail):
            return verdict

        # L2: ORB 特征匹配
        verdict = self._verify_l2_orb(sig_a, sig_b, verdict)
        if not verdict.l2_pass:
            verdict.reasons.append("三级级联: L1通过, L2(ORB)未通过")
            return verdict

        # L3: 直方图相关性
        verdict = self._verify_l3_histogram(sig_a, sig_b, verdict)
        return verdict

    def _verify_l1_hash(
        self, sig_a, sig_b, verdict
    ):
        if sig_a.phash and sig_b.phash:
            verdict.phash_dist = self.hamming_distance("p" + sig_a.phash, "p" + sig_b.phash)
        if sig_a.dhash and sig_b.dhash:
            verdict.dhash_dist = self.hamming_distance("d" + sig_a.dhash, "d" + sig_b.dhash)
        if sig_a.whash and sig_b.whash:
            verdict.whash_dist = self.hamming_distance("w" + sig_a.whash, "w" + sig_b.whash)

        if sig_a.width > 0 and sig_b.width > 0 and sig_a.height > 0 and sig_b.height > 0:
            ar_a = sig_a.aspect_ratio
            ar_b = sig_b.aspect_ratio
            if ar_a > 0 and ar_b > 0:
                ar_ratio = max(ar_a, ar_b) / min(ar_a, ar_b) - 1.0
                if ar_ratio > self.ASPECT_RATIO_TOLERANCE:
                    verdict.reasons.append(f"长宽比不匹配: {ar_a:.2f} vs {ar_b:.2f}")
                    return verdict

        has_phash = bool(sig_a.phash and sig_b.phash)
        has_dhash = bool(sig_a.dhash and sig_b.dhash)
        if not (has_phash or has_dhash):
            verdict.reasons.append("无可比对的哈希类型")
            return verdict

        phash_ok = bool(has_phash and verdict.phash_dist <= self.SIMILAR_DIST)
        dhash_ok = bool(has_dhash and verdict.dhash_dist <= self.SIMILAR_DIST)

        if has_phash and has_dhash:
            if phash_ok and dhash_ok:
                max_dist = max(verdict.phash_dist, verdict.dhash_dist)
                verdict.confidence = max(0.0, 1.0 - max_dist / self.SIMILAR_DIST)
                verdict.reasons.append(f"L1: pHash={verdict.phash_dist}, dHash={verdict.dhash_dist}")
            else:
                verdict.reasons.append(f"L1: pHash={'OK' if phash_ok else verdict.phash_dist}, "
                                       f"dHash={'OK' if dhash_ok else verdict.dhash_dist}")
                return verdict
        elif has_phash:
            if phash_ok:
                verdict.confidence = max(0.0, 1.0 - verdict.phash_dist / self.SIMILAR_DIST)
                verdict.reasons.append(f"L1: pHash={verdict.phash_dist}")
            else:
                verdict.reasons.append(f"L1: pHash={verdict.phash_dist} 未通过")
                return verdict
        elif has_dhash:
            if dhash_ok:
                verdict.confidence = max(0.0, 1.0 - verdict.dhash_dist / self.SIMILAR_DIST)
                verdict.reasons.append(f"L1: dHash={verdict.dhash_dist}")
            else:
                verdict.reasons.append(f"L1: dHash={verdict.dhash_dist} 未通过")
                return verdict

        verdict.l1_pass = True
        verdict.is_match = True
        return verdict

    ORB_MIN_MATCHES = 5
    ORB_MIN_MATCH_RATIO = 0.20

    def _verify_l2_orb(self, sig_a, sig_b, verdict):
        # 方案4：根据 L1 置信度自适应调整 ORB 特征数
        orb_nfeatures = 200
        if verdict.confidence >= 0.7:
            orb_nfeatures = 50   # 高度相似 → 轻量确认
        elif verdict.confidence >= 0.4:
            orb_nfeatures = 100  # 中等相似 → 适度确认

        kp_a, des_a = sig_a.get_orb_features(nfeatures=orb_nfeatures)
        kp_b, des_b = sig_b.get_orb_features(nfeatures=orb_nfeatures)
        if des_a is None or des_b is None:
            verdict.l2_pass = True
            return verdict
        verdict.orb_total_kp = min(len(kp_a) or 0, len(kp_b) or 0)
        if verdict.orb_total_kp < 5:
            verdict.l2_pass = True
            return verdict
        try:
            import cv2
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des_a, des_b)
        except Exception as e:
            logger.debug(f"ORB 失败: {e}")
            verdict.l2_pass = True
            return verdict
        if matches:
            verdict.orb_match_count = len(matches)
            match_ratio = len(matches) / max(1, verdict.orb_total_kp)
            verdict.orb_match_ratio = match_ratio
            if len(matches) >= self.ORB_MIN_MATCHES and match_ratio >= self.ORB_MIN_MATCH_RATIO:
                verdict.l2_pass = True
                verdict.reasons.append(f"L2(ORB): {len(matches)}/{verdict.orb_total_kp} ratio={match_ratio:.2f}")
            else:
                verdict.l2_pass = False
                verdict.is_match = False
                verdict.confidence = 0.0
                verdict.reasons.append(f"L2(ORB): {len(matches)}/{verdict.orb_total_kp} ratio={match_ratio:.2f} — 结构不同")
        else:
            verdict.l2_pass = False
            verdict.is_match = False
            verdict.confidence = 0.0
            verdict.reasons.append("L2(ORB): 无匹配")
        return verdict

    HIST_CORR_PASS = 0.70
    HIST_CORR_LOW = 0.40

    def _verify_l3_histogram(self, sig_a, sig_b, verdict):
        hist_a = sig_a.get_histogram()
        hist_b = sig_b.get_histogram()
        if hist_a is None or hist_b is None:
            verdict.l3_pass = True
            return verdict
        try:
            import cv2
            correlation = cv2.compareHist(hist_a.astype(np.float32), hist_b.astype(np.float32), cv2.HISTCMP_CORREL)
        except Exception:
            correlation = float(np.corrcoef(hist_a, hist_b)[0, 1]) if len(hist_a) > 1 else 0.0
        verdict.histogram_correlation = correlation
        if correlation >= self.HIST_CORR_PASS:
            verdict.l3_pass = True
            verdict.confidence *= min(1.0, correlation / 0.85)
            verdict.reasons.append(f"L3(直方图): 相关性={correlation:.3f}")
        elif correlation >= self.HIST_CORR_LOW:
            verdict.l3_pass = True
            verdict.confidence *= 0.6
            verdict.reasons.append(f"L3(直方图): 低相关={correlation:.3f}")
        else:
            verdict.l3_pass = False
            verdict.is_match = False
            verdict.confidence = 0.0
            verdict.reasons.append(f"L3(直方图): 相关性={correlation:.3f} — 颜色不同")
        return verdict

    # ================================================================
    # 批量匹配（文档级）
    # ================================================================

    def match_images(
        self,
        sigs_a: List[ImageSignature],
        sigs_b: List[ImageSignature],
        max_matches: int = 0,
        boilerplate_hashes: set = None,
    ) -> List[ImageMatchVerdict]:
        """批量匹配两个文档的图片签名列表（方案1+2b+6 优化版）

        方案1：哈希分桶预筛（40-bit pHash + 邻居桶），避免 O(n²) 暴力对比。
        方案2b：max_matches 提前终止。
        方案6：boilerplate_hashes 跳过已知模板哈希。

        Args:
            sigs_a: 文档 A 的图片签名列表
            sigs_b: 文档 B 的图片签名列表
            max_matches: >0 时找到此数量的匹配后提前终止
            boilerplate_hashes: 已知模板哈希集合，匹配时跳过

        Returns:
            按置信度降序排列的匹配结果列表
        """
        if not sigs_a or not sigs_b:
            return []

        # 方案6：过滤掉模板图
        if boilerplate_hashes:
            sigs_a = [s for s in sigs_a if s.phash not in boilerplate_hashes]
            sigs_b = [s for s in sigs_b if s.phash not in boilerplate_hashes]
            if not sigs_a or not sigs_b:
                return []

        # 方案1：哈希分桶预筛 — 2 hex 字符(8 bits)桶 + 比特翻转邻居
        from collections import defaultdict
        buckets = defaultdict(list)
        for sig in sigs_b:
            if sig.phash:
                buckets[sig.phash[:2]].append(sig)

        def neighbor_keys(key):
            """生成主桶 + 邻居桶 key（单比特翻转，覆盖比特翻转导致的所有 hex 变化）

            相比旧版 hex±1 邻居，本方法能覆盖更多情况：
            - hex±1 只能覆盖某比特变化恰好使 hex 值 ±1 的单比特翻转（约 25% 情况）
            - 比特翻转覆盖全部 4 种可能的单比特翻转（100% 情况）
            """
            keys = {key}
            for i, ch in enumerate(key):
                v = int(ch, 16)
                for bit in range(4):
                    nv = v ^ (1 << bit)
                    keys.add(key[:i] + hex(nv)[2] + key[i+1:])
            return keys

        results = []
        for sa in sigs_a:
            if not sa.phash:
                continue

            # 收集所有候选桶中的图片（用 dict 去重，避免 unhashable type 问题）
            candidates = {}
            for nk in neighbor_keys(sa.phash[:2]):
                for sig in buckets.get(nk, []):
                    candidates[id(sig)] = sig

            for sb in candidates.values():
                # 方案2b：提前终止
                if max_matches > 0 and len(results) >= max_matches:
                    break

                verdict = self.compare_signatures(sa, sb)
                if verdict.is_match and verdict.confidence > 0:
                    results.append(verdict)

        # 按置信度降序排列
        results.sort(key=lambda v: v.confidence, reverse=True)
        return results

    # ================================================================
    # 向后兼容：旧版 match_hashes
    # ================================================================

    def match_hashes(
        self,
        hashes_a: List[str],
        hashes_b: List[str],
    ) -> List[HashMatchResult]:
        """旧版单哈希匹配（保留向后兼容）"""
        results = []

        for ha in hashes_a:
            for hb in hashes_b:
                dist = self.hamming_distance(ha, hb)
                if dist <= self.SIMILAR_DIST:
                    results.append(HashMatchResult(
                        hash_a=ha,
                        hash_b=hb,
                        hamming_dist=dist,
                        is_exact=(dist == self.EXACT_DIST),
                        is_similar=(dist <= self.NEAR_IDENTICAL_DIST),
                    ))

        results.sort(key=lambda x: x.hamming_dist)
        return results

    @classmethod
    def filter_ocr_hashes(
        cls,
        ocr_results: List,
        hash_key: str = 'image_hash',
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """从 OCR 结果构建 source_id → 尺寸 映射

        兼容 OCRResult 对象和字典两种输入格式。
        """
        width_map = {}
        height_map = {}
        for r in ocr_results:
            if hasattr(r, hash_key):
                h = getattr(r, hash_key, '')
                w = getattr(r, 'image_width', 0)
                hh = getattr(r, 'image_height', 0)
            else:
                h = r.get(hash_key, '')
                w = r.get('image_width', 0)
                hh = r.get('image_height', 0)
            if h:
                page_num, htype, hval = cls.parse_hash_string(h)
                if page_num is not None:
                    source_id = f"page_{page_num}"
                else:
                    source_id = h[:16]
                if w:
                    width_map[source_id] = w
                if hh:
                    height_map[source_id] = hh
        return width_map, height_map
