"""
模型常驻服务 — API 部署时预加载并保持模型在内存中

用法:
    from deploy.model_server import ModelServer

    # 启动时加载（仅一次）
    server = ModelServer()
    server.warmup()  # 加载 SBERT + OCR，保持常驻

    # 处理请求
    result = server.detect(input_dir, output_dir)

    # 关闭时释放
    server.shutdown()
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ModelServer:
    """模型常驻服务器

    SBERT 语义模型 (~118MB) 和 EasyOCR 模型 (~200MB) 在 warmup() 时加载，
    之后常驻内存，所有检测请求共享同一模型实例，无需反复加载。

    适用场景:
        - FastAPI / Flask API 服务
        - 批量处理多个请求
        - 长时间运行的后台服务
    """

    def __init__(self, config_path: str = None):
        from config import load_config, DetectionConfig
        self.config = load_config(config_path) if config_path else DetectionConfig()

        # 强制启用 GPU 自动检测
        self.config.SBERT_DEVICE = "auto"

        # 模型实例（单例，常驻内存）
        self._sb_embedding_engine = None
        self._ocr_engine = None
        self._orchestrator = None
        self._warmed = False

    def warmup(self) -> bool:
        """预热：加载所有模型到内存并保持常驻

        在 API 服务启动时调用一次。所有模型加载完成后，
        后续 detect() 调用无需重新加载。

        Returns:
            True 如果全部加载成功
        """
        logger.info("=" * 60)
        logger.info("模型预热中... (SBERT + OCR)")
        logger.info("=" * 60)

        success = True

        # 1. SBERT 语义模型（用于文本相似度）
        try:
            from embedding.embedding_engine import EmbeddingEngine
            logger.info("[1/2] 加载 SBERT 语义模型...")
            self._sb_embedding_engine = EmbeddingEngine(self.config)
            # 触发模型加载
            _ = self._sb_embedding_engine.model
            if self._sb_embedding_engine.model is not None:
                logger.info("  SBERT 模型加载完成 ✓")
            else:
                logger.warning("  SBERT 模型加载失败 ✗")
                success = False
        except Exception as e:
            logger.error(f"  SBERT 模型加载异常: {e}")
            success = False

        # 2. OCR 模型（用于图片文字提取）
        if self.config.ENABLE_OCR:
            try:
                from image_analysis.image_ocr import ImageOCREngine
                logger.info("[2/2] 加载 OCR 模型...")
                self._ocr_engine = ImageOCREngine(
                    use_gpu=self.config.USE_GPU,
                    languages=self.config.OCR_LANGUAGES or ['ch_sim', 'en'],
                )
                if self._ocr_engine.is_available:
                    logger.info("  OCR 模型加载完成 ✓")
                else:
                    logger.warning("  OCR 模型加载失败 ✗")
                    success = False
            except Exception as e:
                logger.error(f"  OCR 模型加载异常: {e}")
                success = False

        # 3. 创建检测编排器（注入已加载的模型）
        from pipeline.orchestrator import BidDetectionOrchestrator
        self._orchestrator = BidDetectionOrchestrator(self.config)
        # 替换为已加载的 OCR 引擎（避免重复加载）
        if self._ocr_engine is not None:
            self._orchestrator.ocr_engine = self._ocr_engine

        self._warmed = True

        logger.info("=" * 60)
        logger.info("模型预热完成！服务就绪。")
        logger.info("=" * 60)

        return success

    def detect(self, input_dir: str, output_dir: str):
        """执行检测（使用已加载的模型，无冷启动延迟）

        Args:
            input_dir: PDF 输入目录
            output_dir: 报告输出目录

        Returns:
            GlobalReport
        """
        if not self._warmed:
            logger.warning("模型未预热，正在自动加载...")
            self.warmup()

        if self._orchestrator is None:
            raise RuntimeError("检测引擎初始化失败")

        return self._orchestrator.detect(input_dir, output_dir)

    def shutdown(self):
        """释放模型内存"""
        logger.info("释放模型...")
        self._sb_embedding_engine = None
        self._ocr_engine = None
        self._orchestrator = None
        self._warmed = False
        logger.info("模型已释放")

    @property
    def status(self) -> dict:
        """返回当前模型状态"""
        return {
            'warmed': self._warmed,
            'sbert_loaded': self._sb_embedding_engine is not None
                and self._sb_embedding_engine.model is not None,
            'ocr_loaded': self._ocr_engine is not None
                and self._ocr_engine.is_available,
            'ocr_engine_type': getattr(
                self._ocr_engine, '_engine_type', 'none'
            ) if self._ocr_engine else 'none',
        }
