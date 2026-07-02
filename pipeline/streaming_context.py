"""
LRU 流式上下文管理器

设计目标:
- 仅保持在分析中的文档的活跃文本块（默认每个文档最多 5 个块）
- 使用 OrderedDict 实现 LRU 淘汰策略
- 未命中时从 SQLite 自动加载
- 分析完一对后释放相关文档的内存
"""

import logging
from collections import OrderedDict
from typing import Dict, Optional

from extraction.feature_cache import DocumentCache
from data_structures import ChunkMetadata

logger = logging.getLogger(__name__)


class ChunkData:
    """单个文本块的完整数据（含解压后的文本）"""

    def __init__(self, metadata: ChunkMetadata, text: str):
        self.metadata = metadata
        self.text = text


class StreamingContext:
    """LRU 缓存 — 管理活跃文档的文本块

    工作方式:
    1. 开始分析一对文档时，调用 activate_document(doc_id)
    2. 需要文本时，调用 get_chunk(doc_id, chunk_index)
    3. 分析完成后，调用 release_document(doc_id)
    """

    def __init__(self, cache: DocumentCache, max_chunks_per_doc: int = 5):
        """
        Args:
            cache: SQLite 特征缓存（用于惰性加载）
            max_chunks_per_doc: 每个文档保留在内存中的最大块数
        """
        self.cache = cache
        self.max_chunks_per_doc = max_chunks_per_doc

        # _chunks[doc_id] = OrderedDict[chunk_index, ChunkData]
        self._chunks: Dict[str, OrderedDict[int, ChunkData]] = {}
        self._active_docs: set = set()

    def activate_document(self, doc_id: str):
        """标记文档为活跃状态（预加载其块元数据）"""
        self._active_docs.add(doc_id)
        if doc_id not in self._chunks:
            self._chunks[doc_id] = OrderedDict()

    def get_chunk(self, doc_id: str, chunk_index: int) -> Optional[ChunkData]:
        """获取文本块数据（LRU 缓存 + SQLite 惰性加载）

        Args:
            doc_id: 文档 ID
            chunk_index: 块序号

        Returns:
            ChunkData 或 None（如果块不存在）
        """
        # 确保文档已激活
        if doc_id not in self._chunks:
            self._chunks[doc_id] = OrderedDict()

        chunks = self._chunks[doc_id]

        # 命中缓存 — 移到末尾（最近使用）
        if chunk_index in chunks:
            chunks.move_to_end(chunk_index)
            return chunks[chunk_index]

        # 未命中 — 从 SQLite 惰性加载
        metadata = self.cache.load_chunk_metadata(doc_id, chunk_index)
        if metadata is None:
            return None

        text = self.cache.load_chunk_text(doc_id, chunk_index)
        if text is None:
            return None

        chunk_data = ChunkData(metadata=metadata, text=text)
        chunks[chunk_index] = chunk_data

        # LRU 淘汰：如果超出容量，移除最久未使用的
        while len(chunks) > self.max_chunks_per_doc:
            evicted_idx, evicted_data = chunks.popitem(last=False)
            logger.debug(f"LRU 淘汰: doc={doc_id}, chunk={evicted_idx}")

        return chunk_data

    def get_paragraph_text(self, doc_id: str, para_index: int) -> Optional[str]:
        """直接从 SQLite 加载单个段落文本（跳过块缓存）"""
        return self.cache.load_paragraph_text(doc_id, para_index)

    def release_document(self, doc_id: str):
        """释放文档的所有缓存块（分析完一对后调用）"""
        if doc_id in self._chunks:
            chunk_count = len(self._chunks[doc_id])
            del self._chunks[doc_id]
            logger.debug(f"释放文档 {doc_id} 的 {chunk_count} 个缓存块")
        self._active_docs.discard(doc_id)

    @property
    def memory_usage_estimate(self) -> int:
        """估算当前内存使用量（字节）"""
        total = 0
        for doc_chunks in self._chunks.values():
            for chunk_data in doc_chunks.values():
                total += len(chunk_data.text.encode('utf-8'))
        return total

    @property
    def active_document_count(self) -> int:
        """当前活跃的文档数"""
        return len(self._active_docs)

    def clear(self):
        """清空所有缓存"""
        self._chunks.clear()
        self._active_docs.clear()
        logger.debug("StreamingContext 已清空")
