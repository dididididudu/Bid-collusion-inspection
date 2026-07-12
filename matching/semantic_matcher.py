"""
SBERT 语义推理引擎 — 支持 GPU / ONNX 加速

设备自动检测: CUDA → MPS → CPU
可选 ONNX 路径: 通过 optimum.onnxruntime 实现 2-3x GPU 加速
智能去重: 多次出现的段落只编码一次 (减少 ~50% 编码次数)
"""

import os
os.environ.setdefault('USE_TF', 'FALSE')
import logging
from typing import List, Dict, Tuple, Optional

import numpy as np

from config import DetectionConfig

logger = logging.getLogger(__name__)

# 尝试导入 SBERT
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class SemanticMatcher:
    """SBERT 语义匹配引擎"""

    def __init__(self, config: DetectionConfig, model=None):
        self.config = config
        self._model = model  # 允许外部注入已加载的模型，避免重复加载
        self._device = None

    def set_model(self, model):
        """注入外部已加载的 SBERT 模型，跳过模型加载"""
        self._model = model
        if model is not None:
            logger.info("SemanticMatcher: 复用外部模型实例")

    @property
    def is_available(self) -> bool:
        """SBERT 模型是否可用"""
        return self.model is not None

    @property
    def model(self):
        """延迟加载 SBERT 模型"""
        if self._model is None and SBERT_AVAILABLE:
            self._load_model()
        return self._model

    @property
    def device(self) -> str:
        if self._device is None:
            self._resolve_device()
        return self._device

    def _resolve_device(self):
        """解析运行设备"""
        if self.config.SBERT_DEVICE == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    self._device = "cuda"
                    logger.info("自动检测到 CUDA GPU")
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self._device = "mps"
                    logger.info("自动检测到 Apple MPS")
                else:
                    self._device = "cpu"
                    logger.info("未检测到 GPU，使用 CPU")
            except ImportError:
                self._device = "cpu"
        else:
            self._device = self.config.SBERT_DEVICE

    def _load_model(self):
        """加载 SBERT 模型"""
        logger.info(f"正在加载 SBERT 模型 (设备: {self.device})...")

        try:
            # 设置镜像
            os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

            if self.config.USE_ONNX and self.config.ONNX_MODEL_PATH:
                self._load_onnx_model()
            else:
                self._load_sbert_model()

            if self._model is not None:
                logger.info(f"SBERT 模型加载完成 ({self.device})")
            else:
                logger.warning("SBERT 模型加载失败，将使用基础方法")

        except Exception as e:
            logger.error(f"SBERT 模型加载失败: {e}")
            self._model = None

    def _load_sbert_model(self):
        """加载原始 SBERT 模型（优先本地缓存，失败后再尝试在线）"""
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
            logger.info(f"SBERT 模型离线加载完成 ({self.device})")
            return
        except Exception as e:
            logger.debug(f"SBERT 离线加载失败，尝试在线: {e}")

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
                logger.info(f"SBERT 模型在线加载完成 ({self.device})")
                return
            except Exception as e:
                logger.error(f"SBERT 在线加载也失败: {e}")

        self._model = None

    def _load_onnx_model(self):
        """加载 ONNX 优化模型"""
        try:
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            from transformers import AutoTokenizer

            provider = (
                "CUDAExecutionProvider" if self.device == "cuda"
                else "CPUExecutionProvider"
            )

            self._model = ORTModelForFeatureExtraction.from_pretrained(
                self.config.ONNX_MODEL_PATH,
                file_name="model.onnx",
                provider=provider,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.ONNX_MODEL_PATH
            )
            logger.info("ONNX 模型加载完成")
        except ImportError:
            logger.warning("optimum.onnxruntime 不可用，回退到原始 SBERT")
            self._load_sbert_model()
        except Exception as e:
            logger.error(f"ONNX 模型加载失败: {e}")
            self._load_sbert_model()

    def encode(self, texts: List[str]) -> np.ndarray:
        """编码文本列表为向量

        Args:
            texts: 文本列表

        Returns:
            shape (n_texts, embedding_dim) 的 numpy 数组
        """
        if self._model is None:
            raise RuntimeError("SBERT 模型未加载")

        if not texts:
            return np.array([])

        batch_size = self.config.SBERT_BATCH_SIZE

        if hasattr(self._model, 'encode'):
            # 原始 SBERT
            return self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=getattr(self.config, 'NORMALIZE_EMBEDDINGS', True),
            )
        else:
            # ONNX 路径
            return self._encode_onnx(texts, batch_size)

    def _encode_onnx(self, texts: List[str], batch_size: int) -> np.ndarray:
        """使用 ONNX 模型编码（GPU 端累积，最后一次性传输到 CPU）"""
        import torch

        all_embeddings = []  # 在 GPU 设备上累积 torch 张量
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = self._model(**inputs)

            # 使用 mean pooling
            if hasattr(outputs, 'last_hidden_state'):
                hidden = outputs.last_hidden_state
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                masked = hidden * attention_mask
                summed = masked.sum(dim=1)
                counts = attention_mask.sum(dim=1)
                embeddings = summed / counts
            else:
                embeddings = outputs

            # GPU 端累积，避免逐批 PCIe 传输
            all_embeddings.append(embeddings)

        # 一次性拼接并传输到 CPU
        return torch.cat(all_embeddings, dim=0).cpu().numpy()

    def score_pairs(
        self,
        candidates: List[Tuple[int, int, float]],
        para_texts_a: Dict[int, str],
        para_texts_b: Dict[int, str],
    ) -> List[Dict]:
        """对候选段落对进行 SBERT 语义评分

        优化：去重编码 — 多次出现的段落只编码一次。

        Args:
            candidates: [(para_a_index, para_b_index, jaccard_sim), ...]
            para_texts_a: {para_index: text, ...}
            para_texts_b: {para_index: text, ...}

        Returns:
            [{
                'similarity': float,
                'paragraph_a_index': int,
                'paragraph_b_index': int,
                'detection_method': str,
                'paragraph_a': str,
                'paragraph_b': str,
            }, ...]
        """
        if self._model is None:
            return []

        # === 跨语料库去重：相同文本在 doc_a 和 doc_b 中只编码一次 ===
        # 构建全局唯一文本 → 编码索引映射
        text_to_global_idx = {}
        global_texts = []
        a_global_map = {}  # para_index -> global_idx
        b_global_map = {}  # para_index -> global_idx

        for i, j, _ in candidates:
            if i not in a_global_map and i in para_texts_a:
                text = para_texts_a[i].strip()
                if len(text) >= 15:
                    if text not in text_to_global_idx:
                        text_to_global_idx[text] = len(global_texts)
                        global_texts.append(text)
                    a_global_map[i] = text_to_global_idx[text]
            if j not in b_global_map and j in para_texts_b:
                text = para_texts_b[j].strip()
                if len(text) >= 15:
                    if text not in text_to_global_idx:
                        text_to_global_idx[text] = len(global_texts)
                        global_texts.append(text)
                    b_global_map[j] = text_to_global_idx[text]

        # === 一次性批量编码所有唯一文本 ===
        logger.debug(f"编码 {len(global_texts)} 个唯一段落（跨语料库去重）")
        global_embeddings = self.encode(global_texts)

        # === 批量计算余弦相似度（向量化，消除逐对循环） ===
        # 收集所有有效候选对的嵌入索引
        valid_indices = []
        valid_candidates = []
        for i, j, jaccard_sim in candidates:
            if i in a_global_map and j in b_global_map:
                valid_indices.append((a_global_map[i], b_global_map[j]))
                valid_candidates.append((i, j, jaccard_sim))

        if not valid_indices:
            return []

        # 提取嵌入矩阵（m 行 × d 列）
        idx_a_list = [p[0] for p in valid_indices]
        idx_b_list = [p[1] for p in valid_indices]
        embs_a = global_embeddings[idx_a_list]  # (m, d)
        embs_b = global_embeddings[idx_b_list]  # (m, d)

        # 向量化余弦相似度：逐行归一化后点积
        norms_a = np.linalg.norm(embs_a, axis=1, keepdims=True)
        norms_b = np.linalg.norm(embs_b, axis=1, keepdims=True)
        # 避免除零
        norms_a[norms_a == 0] = 1.0
        norms_b[norms_b == 0] = 1.0
        embs_a_norm = embs_a / norms_a
        embs_b_norm = embs_b / norms_b
        sims = np.sum(embs_a_norm * embs_b_norm, axis=1)  # (m,)

        # === 阈值过滤并构建结果 ===
        # 自适应阈值（已禁用：小候选集时 -0.03 调整会导致短段落阈值从 0.90 降至 0.87，
        # 使模板化短句如"承诺方名称：A公司（盖章）" vs "投标人名称：B公司（公章）"误判为相似）
        base_threshold_adj = 0.0

        results = []
        for k, (i, j, jaccard_sim) in enumerate(valid_candidates):
            sim = float(sims[k])

            text_a = para_texts_a[i]
            text_b = para_texts_b[j]
            avg_len = (len(text_a) + len(text_b)) / 2

            if avg_len < self.config.SBERT_SHORT_PARAGRAPH_LEN:
                threshold = self.config.SBERT_SHORT_PARAGRAPH_THRESHOLD
            else:
                threshold = self.config.SBERT_BASE_THRESHOLD

            threshold = max(0.75, min(0.90, threshold + base_threshold_adj))

            if sim >= threshold:
                results.append({
                    'similarity': sim,
                    'paragraph_a_index': i,
                    'paragraph_b_index': j,
                    'detection_method': 'SBERT',
                    'paragraph_a': text_a,
                    'paragraph_b': text_b,
                    'is_continuous_clone': False,
                    'continuous_clone_group_id': '',
                    'highlighted_text_a': '',
                    'highlighted_text_b': '',
                    'common_parts': [],
                })

        results.sort(key=lambda x: x['similarity'], reverse=True)
        logger.debug(
            f"SBERT 评分完成: {len(candidates)} 候选 → {len(results)} 匹配 "
            f"(唯一段落: {len(a_global_map)}+{len(b_global_map)}, "
            f"全局去重: {len(global_texts)})"
        )

        return results

    def score_pairs_from_cache(
        self,
        candidates: List[Tuple[int, int, float]],
        para_texts_a: Dict[int, str],
        para_texts_b: Dict[int, str],
        cache,
        doc_a_id: str = '',
        doc_b_id: str = '',
        preloaded_embeddings_a: Dict[int, np.ndarray] = None,
        preloaded_embeddings_b: Dict[int, np.ndarray] = None,
    ) -> List[Dict]:
        """从预计算嵌入缓存评分（Phase 3 查表模式，不调 SBERT 模型）

        与 score_pairs 不同，此方法从 SQLite 加载预计算的嵌入向量，
        而不是实时调用 SBERT 编码。适用于 Phase 1.5 已编码所有段落后的场景。

        Args:
            candidates: [(para_a_index, para_b_index, jaccard_sim), ...]
            para_texts_a: {para_index: text}
            para_texts_b: {para_index: text}
            cache: DocumentCache 实例

        Returns:
            与 score_pairs 相同格式的匹配列表
        """
        if not candidates:
            return []

        # 收集所有需要嵌入的段落索引
        all_a_indices = set()
        all_b_indices = set()
        for i, j, _ in candidates:
            if i in para_texts_a:
                all_a_indices.add(i)
            if j in para_texts_b:
                all_b_indices.add(j)

        # 批量从 SQLite 加载预计算嵌入（关键：不调模型）
        if preloaded_embeddings_a is not None:
            embs_a = {
                idx: preloaded_embeddings_a[idx]
                for idx in all_a_indices
                if idx in preloaded_embeddings_a
            }
        else:
            embs_a = cache.load_paragraph_embeddings(
                doc_a_id, list(all_a_indices)
            ) if all_a_indices and doc_a_id else {}
        if preloaded_embeddings_b is not None:
            embs_b = {
                idx: preloaded_embeddings_b[idx]
                for idx in all_b_indices
                if idx in preloaded_embeddings_b
            }
        else:
            embs_b = cache.load_paragraph_embeddings(
                doc_b_id, list(all_b_indices)
            ) if all_b_indices and doc_b_id else {}

        # 构建对齐嵌入矩阵（向量化批量余弦相似度）
        valid_pairs = []
        for i, j, jaccard_sim in candidates:
            ea, eb = embs_a.get(i), embs_b.get(j)
            if ea is not None and eb is not None:
                valid_pairs.append((i, j, jaccard_sim, ea, eb))

        if not valid_pairs:
            return []

        emb_a_mat = np.stack([p[3] for p in valid_pairs])
        emb_b_mat = np.stack([p[4] for p in valid_pairs])
        if getattr(self.config, 'NORMALIZE_EMBEDDINGS', True):
            all_sims = np.sum(emb_a_mat * emb_b_mat, axis=1)
        else:
            na = np.linalg.norm(emb_a_mat, axis=1); na[na == 0] = 1.0
            nb = np.linalg.norm(emb_b_mat, axis=1); nb[nb == 0] = 1.0
            all_sims = np.sum(emb_a_mat * emb_b_mat, axis=1) / (na * nb)

        # 自适应阈值（已禁用：小候选集时 -0.03 调整会导致短段落阈值从 0.90 降至 0.87，
        # 使模板化短句如"承诺方名称：A公司（盖章）" vs "投标人名称：B公司（公章）"误判为相似）
        base_threshold_adj = 0.0

        results = []
        for idx, (i, j, jaccard_sim, _ea, _eb) in enumerate(valid_pairs):
            sim = float(all_sims[idx])

            # 阈值判断
            text_a = para_texts_a[i]
            text_b = para_texts_b[j]
            avg_len = (len(text_a) + len(text_b)) / 2

            if avg_len < self.config.SBERT_SHORT_PARAGRAPH_LEN:
                threshold = self.config.SBERT_SHORT_PARAGRAPH_THRESHOLD
            else:
                threshold = self.config.SBERT_BASE_THRESHOLD

            threshold = max(0.75, min(0.90, threshold + base_threshold_adj))

            if sim >= threshold:
                results.append({
                    'similarity': sim,
                    'paragraph_a_index': i,
                    'paragraph_b_index': j,
                    'detection_method': 'SBERT',
                    'paragraph_a': text_a,
                    'paragraph_b': text_b,
                    'is_continuous_clone': False,
                    'continuous_clone_group_id': '',
                    'highlighted_text_a': '',
                    'highlighted_text_b': '',
                    'common_parts': [],
                })

        results.sort(key=lambda x: x['similarity'], reverse=True)
        logger.debug(
            f"score_pairs_from_cache: {len(valid_pairs)} valid_pairs -> {len(results)} results "
            f"(嵌入命中: A={len(embs_a)}, B={len(embs_b)})"
        )
        return results
