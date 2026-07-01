"""
OCR 引擎封装 — 支持 EasyOCR（主）和 PaddleOCR（备选）

从 PDF 渲染的图片中提取文字内容，输出文本 + 分词 + 文字位置。
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """单张图片的 OCR 结果"""
    text: str = ""                          # 完整文本
    words: List[str] = field(default_factory=list)   # 识别到的词列表
    bboxes: List[Dict] = field(default_factory=list) # [{text, x, y, w, h}, ...]
    confidence: float = 0.0                 # 平均置信度


class ImageOCREngine:
    """OCR 引擎封装

    优先使用 EasyOCR（安装简单，中文效果好）。
    可选 PaddleOCR（需额外安装 paddlepaddle）。
    """

    def __init__(
        self,
        use_gpu: bool = False,
        languages: List[str] = None,
    ):
        self.use_gpu = use_gpu
        self.languages = languages or ['ch_sim', 'en']
        self._reader = None
        self._available = False

    @property
    def is_available(self) -> bool:
        """OCR 引擎是否可用"""
        if self._reader is not None:
            return self._available
        self._init_engine()
        return self._available

    def _init_engine(self):
        """初始化 OCR 引擎"""
        # 尝试 EasyOCR
        try:
            import easyocr
            self._reader = easyocr.Reader(
                self.languages,
                gpu=self.use_gpu,
                verbose=False,
            )
            self._engine_type = 'easyocr'
            self._available = True
            logger.info(
                f"EasyOCR 引擎已初始化 (语言: {self.languages}, "
                f"GPU: {self.use_gpu})"
            )
            return
        except ImportError:
            logger.debug("EasyOCR 不可用，尝试 PaddleOCR...")
        except Exception as e:
            logger.warning(f"EasyOCR 初始化失败: {e}")

        # 回退 PaddleOCR
        try:
            from paddleocr import PaddleOCR
            self._reader = PaddleOCR(
                lang='ch',
                use_angle_cls=False,
                show_log=False,
                use_gpu=self.use_gpu,
            )
            self._engine_type = 'paddleocr'
            self._available = True
            logger.info(f"PaddleOCR 引擎已初始化 (GPU: {self.use_gpu})")
            return
        except ImportError:
            logger.warning("PaddleOCR 不可用")
        except Exception as e:
            logger.warning(f"PaddleOCR 初始化失败: {e}")

        self._reader = None
        self._available = False
        logger.warning("无可用 OCR 引擎，图片文字提取功能禁用")

    def extract(self, image: 'np.ndarray') -> OCRResult:
        """从图片中提取文字

        Args:
            image: numpy 数组 (H, W, 3) RGB 格式

        Returns:
            OCRResult: 包含文本、词列表、位置和置信度
        """
        if not self.is_available:
            return OCRResult()

        try:
            if self._engine_type == 'easyocr':
                return self._extract_easyocr(image)
            elif self._engine_type == 'paddleocr':
                return self._extract_paddleocr(image)
        except Exception as e:
            logger.error(f"OCR 提取失败: {e}")

        return OCRResult()

    def _extract_easyocr(self, image: np.ndarray) -> OCRResult:
        """EasyOCR 提取"""
        raw = self._reader.readtext(image, detail=1)

        if not raw:
            return OCRResult()

        texts = []
        words = []
        bboxes = []
        confidences = []

        for bbox, text, conf in raw:
            if text and len(text.strip()) > 0:
                texts.append(text.strip())
                words.extend(text.strip().split())
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                bboxes.append({
                    'text': text.strip(),
                    'x': int(min(x_coords)),
                    'y': int(min(y_coords)),
                    'w': int(max(x_coords) - min(x_coords)),
                    'h': int(max(y_coords) - min(y_coords)),
                })
                confidences.append(conf)

        full_text = '\n'.join(texts)
        avg_conf = float(np.mean(confidences)) if confidences else 0.0

        return OCRResult(
            text=full_text,
            words=words,
            bboxes=bboxes,
            confidence=avg_conf,
        )

    def _extract_paddleocr(self, image: np.ndarray) -> OCRResult:
        """PaddleOCR 提取"""
        raw = self._reader.ocr(image, cls=False)

        if not raw or not raw[0]:
            return OCRResult()

        texts = []
        words = []
        bboxes = []
        confidences = []

        for line in raw[0]:
            bbox = line[0]
            text = line[1][0]
            conf = line[1][1]

            if text and len(text.strip()) > 0:
                texts.append(text.strip())
                words.extend(text.strip().split())
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                bboxes.append({
                    'text': text.strip(),
                    'x': int(min(x_coords)),
                    'y': int(min(y_coords)),
                    'w': int(max(x_coords) - min(x_coords)),
                    'h': int(max(y_coords) - min(y_coords)),
                })
                confidences.append(conf)

        full_text = '\n'.join(texts)
        avg_conf = float(np.mean(confidences)) if confidences else 0.0

        return OCRResult(
            text=full_text,
            words=words,
            bboxes=bboxes,
            confidence=avg_conf,
        )
