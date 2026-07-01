"""
全局 SBERT 嵌入编码引擎 — Phase 1.5

一次性编码所有段落的 SBERT 嵌入并持久化到 SQLite，
后续 Phase 3 只做查表+点积，不再调用模型。
"""

import os
import logging
from typing import List, Dict, Optional

import numpy as np

from config import DetectionConfig

logger = logging.getLogger(__name__)

# 延迟导入 SBERT（避免启动开销）
SBERT_AVAILABLE = False
SentenceTransformer = None
try:
    from sentence_transformers import SentenceTransformer as ST
    SentenceTransformer = ST
    SBERT_AVAILABLE = True
except ImportError:
    pass


class EmbeddingEngine:
    """全局嵌入编码引擎

    职责：
    1. 加载 SBERT 模型（单例）
    2. 为一个文档的所有段落编码嵌入
    3. 计算文档级嵌入（均值池化）
    4. 通过 DocumentCache 持久化
    """

    def __init__(self, config: DetectionConfig):
        self.config = config
        self._model = None
        self._device = None

    @property
    def is_available(self) -> bool:
        return SBERT_AVAILABLE

    @property
    def device(self) -> str:
        if self._device is None:
            self._resolve_device()
        return self._device

    def _resolve_device(self):
        if self.config.SBERT_DEVICE == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    self._device = "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self._device = "mps"
                else:
                    self._device = "cpu"
            except ImportError:
                self._device = "cpu"
        else:
            self._device = self.config.SBERT_DEVICE

    @property
    def model(self):
        """延迟加载模型"""
        if self._model is None and SBERT_AVAILABLE:
            self._load_model()
        return self._model

    def _load_model(self):
        logger.info(f"EmbeddingEngine: 加载 SBERT 模型 (设备: {self.device})...")
        os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
        local_only = os.environ.get('TRANSFORMERS_OFFLINE', '') == '1'

        # 先尝试离线加载（速度快，不依赖网络）
        try:
            self._model = SentenceTransformer(
                'paraphrase-multilingual-MiniLM-L12-v2',
                device=self.device,
                cache_folder='./models',
                trust_remote_code=True,
                local_files_only=True,
            )
            logger.info(f"EmbeddingEngine: 模型离线加载完成")
            return
        except Exception as e:
            logger.debug(f"EmbeddingEngine: 离线加载失败 ({e})，尝试在线...")

        # 回退：在线加载
        if not local_only:
            try:
                self._model = SentenceTransformer(
                    'paraphrase-multilingual-MiniLM-L12-v2',
                    device=self.device,
                    cache_folder='./models',
                    trust_remote_code=True,
                    local_files_only=False,
                )
                logger.info(f"EmbeddingEngine: 模型在线加载完成")
                return
            except Exception as e:
                logger.error(f"EmbeddingEngine: 模型加载失败: {e}")

        self._model = None

    def encode_document(
        self,
        doc_id: str,
        paragraphs: List[str],
        cache,
    ) -> int:
        """为一个文档的所有段落编码并持久化

        Args:
            doc_id: 文档 ID
            paragraphs: 段落文本列表（按 para_index 顺序）
            cache: DocumentCache 实例

        Returns:
            成功编码的段落数
        """
        if not paragraphs or self.model is None:  # self.model 触发延迟加载
            return 0

        batch_size = max(1, self.config.SBERT_BATCH_SIZE)
        logger.info(
            f"EmbeddingEngine: 编码文档 {doc_id[:12]}... "
            f"({len(paragraphs)} 段落, batch_size={batch_size})"
        )

        # SBERT 编码
        embeddings = self._model.encode(
            paragraphs,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # 持久化到 SQLite
        para_indices = list(range(len(paragraphs)))
        cache.store_paragraph_embeddings(doc_id, para_indices, embeddings)

        # 计算并存储文档级嵌入（均值池化）
        doc_embedding = np.mean(embeddings, axis=0).astype(np.float32)
        cache.store_document_embedding(doc_id, doc_embedding)

        logger.debug(
            f"EmbeddingEngine: {doc_id[:12]}... — "
            f"{len(paragraphs)} 嵌入已存储, "
            f"文档向量维度={doc_embedding.shape[0]}"
        )
        return len(paragraphs)

    def compute_doc_embedding_from_cache(
        self, doc_id: str, cache
    ) -> Optional[np.ndarray]:
        """从缓存的段落嵌入计算文档级嵌入（均值池化）

        用于从已缓存的段落嵌入重新计算文档向量。
        """
        para_embs = cache.load_all_paragraph_embeddings(doc_id)
        if not para_embs:
            return None
        all_embs = np.stack(list(para_embs.values()))
        return np.mean(all_embs, axis=0).astype(np.float32)
