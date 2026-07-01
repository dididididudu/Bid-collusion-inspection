"""
分块文本处理器 — 聚合多个文本块的特征为文档级签名

设计目标:
- 每个块独立处理（内存友好）
- 聚合策略：文档级 SimHash 使用 bitwise-AND（保守匹配）
- 文档级 MinHash 使用所有段落 MinHash 的逐位最小值
- 报价和图片跨块合并去重
"""

import logging
from typing import List, Dict
from collections import Counter

import numpy as np

from data_structures import BidFeature, ChunkResult, MetadataFeature, QuoteSignature
from config import DetectionConfig

logger = logging.getLogger(__name__)


class ChunkedTextProcessor:
    """分块文本处理器 — 聚合多块特征为文档级签名"""

    def __init__(self, config: DetectionConfig):
        self.config = config

    def aggregate_chunks(
        self,
        doc_id: str,
        filename: str,
        file_size: int,
        chunks: List[ChunkResult],
        metadata: MetadataFeature,
        is_scanned: bool = False,
        page_count: int = 0,
    ) -> BidFeature:
        """聚合所有文本块的特征为文档级描述符

        Args:
            doc_id: 文档 ID
            filename: 文件名
            file_size: 文件大小（字节）
            chunks: 所有文本块的处理结果
            metadata: 从 Phase 0 提取的元数据
            is_scanned: 是否为扫描版
            page_count: 总页数

        Returns:
            BidFeature: 文档级特征（轻量级，不含文本内容）
        """
        logger.info(f"聚合 {len(chunks)} 个块的特征 -> {filename}")

        # 聚合 SimHash（bitwise-AND 策略：保守匹配）
        doc_simhash = self._aggregate_simhash(chunks)

        # 聚合 MinHash（所有段落的签名）
        all_paragraph_hashes = []
        for chunk in chunks:
            all_paragraph_hashes.extend(chunk.paragraph_hashes)

        doc_minhash = self._aggregate_minhash(all_paragraph_hashes)

        # 聚合报价
        all_quotes = []
        seen_quotes = set()
        for chunk in chunks:
            for q in chunk.quotes:
                if q not in seen_quotes:
                    all_quotes.append(q)
                    seen_quotes.add(q)
        all_quotes.sort()

        quote_signature = self._compute_quote_signature(all_quotes)

        # 聚合图片哈希
        all_image_hashes = []
        seen_hashes = set()
        for chunk in chunks:
            for h in chunk.image_hashes:
                if h not in seen_hashes:
                    all_image_hashes.append(h)
                    seen_hashes.add(h)

        # 计算总文本长度
        total_text_length = sum(
            len(getattr(chunk, 'text', '')) or
            sum(len(p) for p in chunk.paragraphs)
            for chunk in chunks
        )

        # 总段落数
        total_paragraphs = sum(len(chunk.paragraphs) for chunk in chunks)

        return BidFeature(
            doc_id=doc_id,
            filename=filename,
            file_size=file_size,
            text_content="",  # 不存储在内存中
            text_length=total_text_length,
            text_simhash=doc_simhash,
            paragraphs=[],  # 不存储
            paragraph_hashes=[],  # 不存储
            metadata=metadata,
            quotes=all_quotes,
            quote_signature=quote_signature,
            image_hashes=all_image_hashes,
            extracted_at=chunks[0].text[:0] if chunks else "",  # placeholder
            is_scanned=is_scanned,
            page_count=page_count,
            doc_minhash=doc_minhash,
            chunk_count=len(chunks),
        )

    def _aggregate_simhash(self, chunks: List[ChunkResult]) -> str:
        """聚合多块的 SimHash（保守策略：bitwise-AND）

        只有所有块都为 1 的位才设为 1，这样只有两份文档在所有块中都相似时
        才会匹配。适合检测全局高度相似的文档。
        """
        valid_hashes = [c.simhash for c in chunks if c.simhash and c.simhash != "0" * 16]

        if not valid_hashes:
            return "0" * 16

        if len(valid_hashes) == 1:
            return valid_hashes[0]

        # 将所有 SimHash 预转换为整数，逐位 AND
        ints = [int(h, 16) for h in valid_hashes]
        result_int = ints[0]
        for x in ints[1:]:
            result_int &= x

        return format(result_int, '016x')

    def _aggregate_minhash(self, all_paragraph_hashes: List[str]) -> List[int]:
        """聚合所有段落 MinHash 签名为文档级 MinHash

        策略：对每个哈希维度取所有段落的最小值。
        这确保了语义相似的文档在 MinHash LSH 中会被放入同一个桶。
        """
        if not all_paragraph_hashes:
            return []

        # 解析所有有效的 MinHash 签名
        parsed_hashes = []
        for h in all_paragraph_hashes:
            if h:
                try:
                    values = [int(v) for v in h.split(',')]
                    if values:
                        parsed_hashes.append(values)
                except (ValueError, TypeError):
                    continue

        if not parsed_hashes:
            return []

        # 使用 numpy 向量化取每个维度的最小值（比 Python 循环快 10-50x）
        import numpy as np
        arr = np.array(parsed_hashes, dtype=np.int64)
        result = np.min(arr, axis=0).tolist()
        return result

    def _compute_quote_signature(self, quotes: List[float]) -> QuoteSignature:
        """计算报价统计特征"""
        if not quotes:
            return QuoteSignature()

        tail_distribution = {}
        integer_count = 0

        for quote in quotes:
            decimal_part = quote - int(quote)
            if decimal_part < 0.01:
                integer_count += 1
                tail_key = "00"
            else:
                tail_key = str(int(decimal_part * 100)).zfill(2)
            tail_distribution[tail_key] = tail_distribution.get(tail_key, 0) + 1

        integer_ratio = integer_count / len(quotes)
        mean = float(np.mean(quotes))
        std = float(np.std(quotes))

        return QuoteSignature(
            count=len(quotes),
            values=quotes,
            tail_distribution=tail_distribution,
            integer_ratio=integer_ratio,
            mean=mean,
            std=std,
        )
