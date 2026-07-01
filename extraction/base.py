"""
PDF 提取器抽象基类

定义所有 PDF 提取器必须实现的接口，支持:
- pymupdf (fitz): 高速 PDF 解析（推荐，10x 速度提升）
- pdfplumber: 传统解析器（回退方案）
"""

from abc import ABC, abstractmethod
from typing import List, Generator, Tuple, Optional
from data_structures import MetadataFeature, ChunkResult


class BasePDFExtractor(ABC):
    """PDF 提取器抽象基类"""

    @abstractmethod
    def extract_metadata(self, file_path: str) -> Tuple[MetadataFeature, int, bool]:
        """提取 PDF 元数据和页数

        Phase 0 调用，仅读取元数据，不解析文本。

        Args:
            file_path: PDF 文件路径

        Returns:
            (metadata, page_count, is_scanned)
        """
        pass

    @abstractmethod
    def extract_chunks(
        self,
        file_path: str,
        chunk_size: int,
        start_page: int = 0,
    ) -> Generator[ChunkResult, None, None]:
        """按块流式提取文本内容

        Phase 1 调用，每次返回一个块的文本。

        Args:
            file_path: PDF 文件路径
            chunk_size: 每块的页数
            start_page: 起始页码（用于恢复）

        Yields:
            ChunkResult: 每个块的处理结果
        """
        pass

    @abstractmethod
    def get_page_count(self, file_path: str) -> int:
        """快速获取 PDF 页数（不加载全部内容）"""
        pass
