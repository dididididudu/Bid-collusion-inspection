"""
GPU 资源管理器 — 单例进程，统一管理所有 GPU 操作

问题: Phase 1 多进程各自创建 OCR 引擎 → N 份模型抢 GPU → OOM
解决: 独立进程持有唯一 OCR 引擎，Worker 通过队列提交图片批量推理

架构:
    GPUManager (主进程)          GPUManagerClient (worker 进程)
    ┌──────────────────┐         ┌─────────────────────┐
    │  _run() 进程      │◄─队列──│  batch_ocr(images)   │
    │  唯一 OCR 引擎     │──队列──►  阻塞等待结果          │
    └──────────────────┘         └─────────────────────┘

兼容性: 通过 ImageOCREngine 抽象层支持 EasyOCR（当前）和 PaddleOCR（将来）
         切换引擎只需修改 config.OCR_ENGINE，无需改动本模块
"""

import os
import time
import logging
import multiprocessing
from multiprocessing import Process, Queue
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# 哨兵值 — 通知 GPU Manager 进程优雅退出
_SENTINEL = '__GPU_MANAGER_SHUTDOWN__'
_QUEUE_TIMEOUT = 60  # 1 分钟（单批次 OCR 推理不应超过此时间）


# ────────────────────────────────────────────────────────────
# Client — 在 worker 进程中使用，仅持有队列引用
# ────────────────────────────────────────────────────────────

class GPUManagerClient:
    """GPU Manager 客户端 — 在 worker 进程中通过队列与 Manager 通信

    可 pickle 序列化，通过 ProcessPoolExecutor 传递给 worker。
    不创建进程、不加载 GPU 模型，只收发消息。
    """

    def __init__(self, task_queue: Queue, result_queue: Queue, enabled: bool = True):
        self._task_queue = task_queue
        self._result_queue = result_queue
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    def batch_ocr(
        self,
        images: List[np.ndarray],
        metadata: List[dict] = None,
    ) -> Tuple[List, List[dict]]:
        """提交批量 OCR 任务，阻塞等待结果

        Args:
            images: 图片列表
            metadata: 每张图的元信息

        Returns:
            (ocr_results, metadata) — Tuple，避免 pickle 复杂嵌套
        """
        if not self._enabled or not images:
            return ([], metadata or [])

        import uuid
        task_id = uuid.uuid4().hex
        meta = metadata or [{}] * len(images)

        self._task_queue.put((task_id, images, meta), timeout=_QUEUE_TIMEOUT)

        try:
            result_id, results = self._result_queue.get(timeout=_QUEUE_TIMEOUT)
            if result_id != task_id:
                logger.error("GPU Manager 返回的 task_id 不匹配: %s != %s", result_id, task_id)
                return ([], meta)
            return results  # (ocr_results, metadata)
        except Exception as e:
            logger.error("等待 GPU Manager 结果超时: %s", e)
            return ([], meta)


# ────────────────────────────────────────────────────────────
# Manager — 在主进程中使用，管理 GPU 进程
# ────────────────────────────────────────────────────────────

class GPUManager:
    """GPU 资源管理器 — 主进程单例

    用法:
        manager = GPUManager(config)
        manager.start()
        client = manager.client  # 传递给 worker
        ...
        manager.shutdown()
    """

    def __init__(self, config):
        self.config = config
        self._enabled = (
            getattr(config, 'GPU_MANAGER_ENABLED', False)
            and config.USE_GPU
            and config.ENABLE_OCR
        )

        self._process: Optional[Process] = None
        self._started = False
        self._client: Optional[GPUManagerClient] = None

        if self._enabled:
            ctx = multiprocessing.get_context('spawn')
            self._task_queue: Queue = ctx.Queue()
            self._result_queue: Queue = ctx.Queue()
            self._client = GPUManagerClient(self._task_queue, self._result_queue, enabled=True)
        else:
            self._task_queue = None
            self._result_queue = None
            self._client = GPUManagerClient(None, None, enabled=False)

    @property
    def enabled(self) -> bool:
        return self._enabled and self._started

    @property
    def client(self) -> GPUManagerClient:
        """获取可 pickle 的客户端对象，传递给 worker 进程"""
        return self._client

    def start(self):
        if not self._enabled:
            logger.debug("GPU Manager 未启用")
            return
        if self._started:
            return

        self._process = Process(target=self._run, daemon=True, name="gpu-manager")
        self._process.start()
        self._started = True
        logger.info("GPU Manager 进程已启动 (PID: %d)", self._process.pid)

    def shutdown(self, timeout: float = 30.0):
        if not self._started:
            return

        logger.info("GPU Manager 正在关闭...")
        try:
            self._task_queue.put(_SENTINEL, timeout=5)
        except Exception:
            pass

        if self._process and self._process.is_alive():
            self._process.join(timeout=timeout)
            if self._process.is_alive():
                logger.warning("GPU Manager 未在超时内退出，强制终止")
                self._process.terminate()
                self._process.join(timeout=5)

        self._started = False
        logger.info("GPU Manager 已关闭")

    # ── 进程主循环 ──

    def _run(self):
        self._setup_child_logging()
        logger.info("GPU Manager 子进程启动 (PID: %d)", os.getpid())

        engine = None
        try:
            engine = self._init_ocr_engine()
            logger.info("GPU Manager: OCR 引擎就绪，等待任务...")

            while True:
                batch = self._collect_tasks()
                if batch is None:
                    break
                if not batch:
                    continue
                count = self._run_batch_inference(engine, batch)
                if count > 0:
                    logger.info("GPU Manager: 完成 %d 张图片 OCR", count)
        except Exception as exc:
            logger.error("GPU Manager 子进程异常: %s", exc, exc_info=True)
        finally:
            if engine is not None:
                self._cleanup_engine(engine)
            logger.info("GPU Manager 子进程退出 (PID: %d)", os.getpid())

    # ── OCR 引擎管理 ──

    def _init_ocr_engine(self):
        from image_analysis.image_ocr import ImageOCREngine
        t0 = time.perf_counter()
        engine = ImageOCREngine(
            use_gpu=self.config.USE_GPU,
            engine=self.config.OCR_ENGINE,
            model_dir=self.config.OCR_MODEL_DIR,
            offline=self.config.OCR_OFFLINE_MODE,
            retry_count=self.config.OCR_RETRY_COUNT,
        )
        if not engine.is_available:
            raise RuntimeError(f"OCR 引擎 ({self.config.OCR_ENGINE}) 初始化失败")
        elapsed = time.perf_counter() - t0
        logger.info("GPU Manager: %s 就绪 (GPU=%s, %.1fs)", engine.engine_type, self.config.USE_GPU, elapsed)
        return engine

    def _collect_tasks(self) -> Optional[Tuple[str, List, List]]:
        """收集 OCR 任务（支持跨 worker 批量聚合）"""
        try:
            item = self._task_queue.get(timeout=1.0)
        except Exception:
            return None

        if item == _SENTINEL:
            return None

        task_id, images, metadata = item

        batch_size = getattr(self.config, 'OCR_BATCH_SIZE', 8)
        batch_timeout = getattr(self.config, 'OCR_BATCH_TIMEOUT', 2.0)
        deadline = time.monotonic() + batch_timeout

        while len(images) < batch_size and time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                item = self._task_queue.get(timeout=min(0.3, remaining))
                if item == _SENTINEL:
                    break
                _, more_images, more_meta = item
                images.extend(more_images)
                metadata.extend(more_meta)
            except Exception:
                break

        return (task_id, images, metadata)

    def _run_batch_inference(self, engine, batch) -> int:
        task_id, images, metadata = batch
        ocr_results = engine.extract_batch(images, min_confidence=self.config.OCR_MIN_CONFIDENCE)

        for i, (result, meta) in enumerate(zip(ocr_results, metadata)):
            result.image_hash = meta.get('phash', '')
            result.image_width = meta.get('img_w', 0)
            result.image_height = meta.get('img_h', 0)

        self._result_queue.put((task_id, (ocr_results, metadata)))
        return len(ocr_results)

    @staticmethod
    def _cleanup_engine(engine):
        try:
            import gc
            import torch
            del engine
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @staticmethod
    def _setup_child_logging():
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(logging.INFO)
