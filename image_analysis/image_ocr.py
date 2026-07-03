"""
OCR 引擎封装 — 支持 PaddleOCR（主，仅文字识别）和 EasyOCR（备选）

从 PDF 渲染的图片中提取文字内容，输出文本 + 分词 + 文字位置。
PaddleOCR 默认禁用目标检测 (det=False)，仅使用文字识别功能。

部署特性:
- 支持自定义模型目录 (model_dir) 用于离线部署
- warmup() 预热方法提前加载模型
- health_check() 验证引擎实际可用性
- diagnose() 输出详细环境诊断信息
- 内置重试机制 (retry_count)
"""

import os
import sys
import time
import logging
import traceback
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
    image_hash: str = ""                    # 页面图片哈希 (pHash)，供 PS 检测使用
    non_text_hash: str = ""                 # 去除文字区域后的图片哈希（涂掉文字再 hash）
    image_width: int = 0                    # 原始图片宽度（像素）
    image_height: int = 0                   # 原始图片高度（像素）
    thumbnail: bytes = b""                  # 64×64 JPEG 缩略图（供 ORB 匹配+直方图比较）


class ImageOCREngine:
    """OCR 引擎封装 — 部署友好

    优先使用 PaddleOCR（中文识别效果好，仅文字识别，禁用检测）。
    可选 EasyOCR（需额外安装，作为备选）。

    部署用法::

        # 离线模式
        engine = ImageOCREngine(
            engine="paddleocr",
            model_dir="/data/models/ocr",
            offline=True,
        )
        engine.warmup()  # 提前加载模型

        # 健康检查
        ok, msg = engine.health_check()
        if not ok:
            raise RuntimeError(f"OCR 引擎不可用: {msg}")
    """

    # 用于健康检查的测试图片（200x100 白色背景）
    _TEST_IMAGE: np.ndarray = None

    def __init__(
        self,
        use_gpu: bool = False,
        languages: List[str] = None,
        engine: str = "paddleocr",
        model_dir: Optional[str] = None,
        offline: bool = False,
        retry_count: int = 3,
    ):
        self.use_gpu = use_gpu
        self.languages = languages or ['ch_sim', 'en']
        self._engine = engine  # "paddleocr" or "easyocr"
        self._model_dir = model_dir
        self._offline = offline
        self._retry_count = max(0, retry_count)
        self._reader = None
        self._available = False
        self._engine_type = None   # 实际使用的引擎类型
        self._paddle_version = 0   # PaddleOCR 版本号
        self._init_time = 0.0      # 初始化耗时

    # ================================================================
    # 公共属性
    # ================================================================

    @property
    def engine_type(self) -> Optional[str]:
        """实际使用的引擎类型: 'paddleocr' / 'easyocr' / None"""
        return self._engine_type

    @property
    def paddle_version(self) -> int:
        """PaddleOCR 版本号 (2 或 3)"""
        return self._paddle_version

    @property
    def init_time(self) -> float:
        """引擎初始化耗时（秒）"""
        return self._init_time

    @property
    def is_available(self) -> bool:
        """OCR 引擎是否可用"""
        if self._reader is not None:
            return self._available
        self._init_engine()
        return self._available

    # ================================================================
    # 预热与健康检查
    # ================================================================

    def warmup(self) -> bool:
        """预热引擎：强制初始化 + 试运行一次

        在服务启动时调用，确保模型已加载到内存。
        避免首次请求时的冷启动延迟。

        Returns:
            True 表示预热成功
        """
        logger.info(f"正在预热 OCR 引擎 (engine={self._engine})...")
        t0 = time.time()

        if not self.is_available:
            logger.error("OCR 引擎预热失败: 引擎不可用")
            return False

        # 用测试图片试运行
        test_img = self._get_test_image()
        try:
            result = self.extract(test_img)
            elapsed = time.time() - t0
            if self._engine_type == 'paddleocr' and self._paddle_version >= 3:
                # PaddleOCR 3.x 首次调用会触发模型加载
                # 测试图片无文字是正常的
                logger.info(
                    f"OCR 引擎预热完成 ({self._engine_type} v{self._paddle_version}, "
                    f"耗时 {elapsed:.1f}s)"
                )
            else:
                logger.info(
                    f"OCR 引擎预热完成 ({self._engine_type}, "
                    f"耗时 {elapsed:.1f}s)"
                )
            return True
        except Exception as e:
            logger.error(f"OCR 引擎预热失败: {e}")
            return False

    def health_check(self) -> Tuple[bool, str]:
        """健康检查：验证引擎是否真正可用

        Returns:
            (ok, message): ok=True 表示健康，message 包含详细信息
        """
        if not self.is_available:
            return False, f"引擎不可用 (类型: {self._engine_type or 'None'})"

        # 尝试在测试图片上运行
        test_img = self._get_test_image()
        try:
            result = self.extract(test_img)
            return True, (
                f"引擎正常 ({self._engine_type}"
                f"{' v' + str(self._paddle_version) if self._paddle_version else ''}"
                f", GPU: {self.use_gpu}"
                f", 初始化耗时: {self._init_time:.1f}s)"
            )
        except Exception as e:
            return False, f"引擎运行时异常: {type(e).__name__}: {e}"

    @staticmethod
    def diagnose() -> str:
        """输出详细的环境诊断报告（不初始化引擎）

        Returns:
            多行诊断文本
        """
        lines = []
        lines.append("=" * 60)
        lines.append("OCR 环境诊断报告")
        lines.append("=" * 60)

        # Python 信息
        lines.append(f"Python 版本: {sys.version}")
        lines.append(f"平台: {sys.platform}")

        # PaddleOCR
        lines.append("\n--- PaddleOCR ---")
        try:
            import paddleocr
            lines.append(f"  版本: {paddleocr.__version__}")
            lines.append(f"  路径: {paddleocr.__file__}")
        except ImportError:
            lines.append("  ❌ 未安装")
        except Exception as e:
            lines.append(f"  ❌ 导入失败: {e}")

        # PaddlePaddle
        lines.append("\n--- PaddlePaddle ---")
        try:
            import paddle
            lines.append(f"  版本: {paddle.__version__}")
            if hasattr(paddle, 'is_compiled_with_cuda'):
                lines.append(f"  CUDA: {paddle.is_compiled_with_cuda()}")
        except ImportError:
            lines.append("  ❌ 未安装")
        except Exception as e:
            lines.append(f"  ❌ 导入失败: {e}")

        # EasyOCR (备选)
        lines.append("\n--- EasyOCR (备选) ---")
        try:
            import easyocr
            lines.append(f"  版本: {easyocr.__version__ if hasattr(easyocr, '__version__') else 'N/A'}")
        except ImportError:
            lines.append("  ❌ 未安装")
        except Exception as e:
            lines.append(f"  ❌ 导入失败: {e}")

        # PyTorch
        lines.append("\n--- PyTorch ---")
        try:
            import torch
            lines.append(f"  版本: {torch.__version__}")
            lines.append(f"  CUDA: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                lines.append(f"  CUDA 版本: {torch.version.cuda}")
                lines.append(f"  GPU: {torch.cuda.get_device_name(0)}")
        except ImportError:
            lines.append("  ❌ 未安装")
        except Exception as e:
            lines.append(f"  ❌ 导入失败: {e}")

        # 环境变量
        lines.append("\n--- 环境变量 ---")
        for var in ['PADDLEOCR_HOME', 'OCR_MODEL_DIR', 'OCR_OFFLINE',
                     'PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK',
                     'HF_ENDPOINT', 'HF_HOME', 'TRANSFORMERS_OFFLINE']:
            val = os.environ.get(var, '')
            lines.append(f"  {var}: {val if val else '(未设置)'}")

        # 模型缓存目录
        lines.append("\n--- 模型缓存 ---")
        for name, path in [
            ('PaddleOCR 2.x', os.path.expanduser('~/.paddleocr')),
            ('Paddlex 3.x', os.path.expanduser('~/.paddlex')),
        ]:
            if os.path.exists(path):
                size_mb = 0
                for root, dirs, files in os.walk(path):
                    for f in files:
                        try:
                            size_mb += os.path.getsize(os.path.join(root, f))
                        except OSError:
                            pass
                size_mb /= (1024 * 1024)
                lines.append(f"  {name}: {path} ({size_mb:.1f} MB)")
                # 列出模型
                for item in sorted(os.listdir(path)):
                    if os.path.isdir(os.path.join(path, item)) and not item.startswith('.'):
                        lines.append(f"    - {item}")
            else:
                lines.append(f"  {name}: {path} (不存在)")

        lines.append("=" * 60)
        return '\n'.join(lines)

    # ================================================================
    # 引擎初始化
    # ================================================================

    def _init_engine(self):
        """初始化 OCR 引擎（带版本检测）"""
        t0 = time.time()

        if self._engine == "paddleocr":
            self._init_paddleocr()
            if self._available:
                self._init_time = time.time() - t0
                return
            self._cleanup_failed_modules()
            logger.info("PaddleOCR 不可用，回退到 EasyOCR...")
            self._init_easyocr()
        elif self._engine == "easyocr":
            self._init_easyocr()
            if self._available:
                self._init_time = time.time() - t0
                return
            logger.info("EasyOCR 不可用，回退到 PaddleOCR...")
            self._init_paddleocr()
        else:
            self._init_paddleocr()
            if not self._available:
                self._cleanup_failed_modules()
                self._init_easyocr()

        if not self._available:
            self._reader = None
            logger.warning("无可用 OCR 引擎，图片文字提取功能禁用")

        self._init_time = time.time() - t0

    @staticmethod
    def _cleanup_failed_modules():
        """清理导入失败后残留的部分模块，避免影响后续引擎初始化"""
        import sys
        modules_to_clean = []
        for name in sys.modules:
            if name.startswith('paddleocr') or name.startswith('paddle') or \
               name.startswith('paddlex') or name.startswith('ppocr'):
                modules_to_clean.append(name)
        for name in modules_to_clean:
            try:
                del sys.modules[name]
            except (KeyError, AttributeError):
                pass

    def _init_paddleocr(self):
        """初始化 PaddleOCR（仅文字识别，禁用目标检测）

        自动检测版本并选择合适的 API:
        - PaddleOCR 2.x: 使用 det=False, rec=True 参数
        - PaddleOCR 3.x: 使用 pipelines API
        """
        try:
            from paddleocr import PaddleOCR

            # 检测版本
            paddle_major = 2
            try:
                import paddleocr as poc
                version_str = getattr(poc, '__version__', '2.0.0')
                paddle_major = int(str(version_str).split('.')[0])
            except Exception:
                pass

            # 构建通用参数
            common_kwargs = {}
            if self._model_dir:
                # 检测子目录结构：PaddleOCR 模型通常放在 det/、rec/ 子目录下
                # 如果用户只指定了父目录，自动补全子目录路径
                det_dir = os.path.join(self._model_dir, 'det')
                rec_dir = os.path.join(self._model_dir, 'rec')
                if os.path.exists(det_dir) and os.path.exists(rec_dir):
                    common_kwargs['det_model_dir'] = det_dir
                    common_kwargs['rec_model_dir'] = rec_dir
                    logger.info(f"使用自定义模型目录 (子目录): det={det_dir}, rec={rec_dir}")
                elif os.path.isdir(self._model_dir) and any(
                    f.endswith('.pdiparams') or f.endswith('.pdmodel')
                    for f in os.listdir(self._model_dir)
                ):
                    # 如果目录直接包含模型文件，直接使用
                    common_kwargs['det_model_dir'] = self._model_dir
                    common_kwargs['rec_model_dir'] = self._model_dir
                    logger.info(f"使用自定义模型目录: {self._model_dir}")
                else:
                    # 目录不存在或结构不明确，让 PaddleOCR 使用内置模型
                    logger.info(
                        f"OCR_MODEL_DIR={self._model_dir} 未检测到标准模型子目录，"
                        f"将使用 PaddleOCR 内置模型（首次运行自动下载）"
                    )

            if paddle_major < 3:
                # === PaddleOCR 2.x ===
                self._reader = PaddleOCR(
                    lang='ch',
                    use_angle_cls=False,
                    show_log=False,
                    use_gpu=self.use_gpu,
                    det=False,       # 禁用目标检测，仅文字识别
                    rec=True,        # 启用文字识别
                    **common_kwargs,
                )
                self._paddle_version = 2
            else:
                # === PaddleOCR 3.x (PP-OCRv6) ===
                # 3.x 使用 paddlex pipelines，支持 PP-OCRv5/v6 等新版模型
                # GPU 环境下检测速度很快，使用完整的检测+识别 pipeline
                self._reader = PaddleOCR(
                    lang='ch',
                    ocr_version='PP-OCRv6',
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    **common_kwargs,
                )
                self._paddle_version = 3

            self._engine_type = 'paddleocr'
            self._available = True
            logger.info(
                f"PaddleOCR v{paddle_major} 引擎已初始化 "
                f"(仅文字识别, GPU: {self.use_gpu}"
                f"{', model_dir: ' + self._model_dir if self._model_dir else ''})"
            )

        except ImportError:
            logger.debug("PaddleOCR 未安装")
            self._available = False
        except TypeError as e:
            logger.warning(f"PaddleOCR 参数不兼容: {e}")
            self._available = False
        except Exception as e:
            logger.warning(f"PaddleOCR 初始化失败: {e}")
            self._available = False

    def _init_easyocr(self):
        """初始化 EasyOCR（备选引擎）"""
        try:
            import easyocr
            # 模型存放到项目根目录下的 models/easyocr/ 中
            # 用脚本自身路径定位根目录，避免 worker 进程 cwd 不一致
            _script_dir = os.path.dirname(os.path.abspath(__file__))
            _project_root = os.path.dirname(_script_dir)  # 上一级 = 项目根
            model_dir = self._model_dir or os.path.join(_project_root, 'models', 'easyocr')
            os.makedirs(model_dir, exist_ok=True)
            self._reader = easyocr.Reader(
                self.languages,
                gpu=self.use_gpu,
                verbose=False,
                model_storage_directory=model_dir,
            )
            self._engine_type = 'easyocr'
            self._available = True
            logger.info(
                f"EasyOCR 引擎已初始化 (语言: {self.languages}, "
                f"GPU: {self.use_gpu})"
            )
        except ImportError:
            logger.debug("EasyOCR 未安装")
        except Exception as e:
            logger.warning(f"EasyOCR 初始化失败: {e}")

    # ================================================================
    # OCR 提取（带重试）
    # ================================================================

    def extract(self, image: 'np.ndarray') -> OCRResult:
        """从图片中提取文字（带重试机制）

        Args:
            image: numpy 数组 (H, W, 3) RGB 格式

        Returns:
            OCRResult: 包含文本、词列表、位置和置信度
        """
        if not self.is_available:
            return OCRResult()

        last_error = None
        for attempt in range(self._retry_count + 1):
            try:
                if self._engine_type == 'easyocr':
                    return self._extract_easyocr(image)
                elif self._engine_type == 'paddleocr':
                    return self._extract_paddleocr(image)
            except Exception as e:
                last_error = e
                if attempt < self._retry_count:
                    wait = 0.5 * (attempt + 1)
                    logger.debug(
                        f"OCR 提取失败 (尝试 {attempt + 1}/{self._retry_count + 1}): "
                        f"{e}, {wait:.1f}s 后重试..."
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        f"OCR 提取失败 (已重试 {self._retry_count} 次): {e}"
                    )

        return OCRResult()

    def _extract_easyocr(self, image: np.ndarray) -> OCRResult:
        """EasyOCR 提取（含检测+识别）"""
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
        """PaddleOCR 提取（仅文字识别，无检测）

        兼容 PaddleOCR 2.x (det=False/rec=True) 和 3.x (pipelines 模式)。
        """
        paddle_version = getattr(self, '_paddle_version', 2)

        try:
            if paddle_version >= 3:
                raw = self._reader.predict(image)
            else:
                raw = self._reader.ocr(image, cls=False)
        except Exception as e:
            logger.error(f"PaddleOCR 调用失败: {e}")
            return OCRResult()

        if not raw or not raw[0]:
            return OCRResult()

        texts = []
        words = []
        bboxes = []
        confidences = []

        for item in raw[0]:
            # PaddleOCR 3.x 返回 dict 格式
            if isinstance(item, dict):
                rec_texts = item.get('rec_texts', item.get('text', ''))
                rec_scores = item.get('rec_scores', item.get('confidence', 0.0))
                if isinstance(rec_texts, list):
                    for t, c in zip(rec_texts,
                                    rec_scores if isinstance(rec_scores, list)
                                    else [rec_scores] * len(rec_texts)):
                        if t and len(str(t).strip()) > 0:
                            t = str(t).strip()
                            texts.append(t)
                            words.extend(t.split())
                            confidences.append(float(c) if c else 0.5)
                elif isinstance(rec_texts, str) and rec_texts.strip():
                    texts.append(rec_texts.strip())
                    words.extend(rec_texts.strip().split())
                    confidences.append(float(rec_scores) if rec_scores else 0.5)
                continue

            # PaddleOCR 2.x det=True 格式: [[[x1,y1],...], [text, conf]]
            if isinstance(item[0], (list, tuple)) and len(item) >= 2:
                first_elem = item[0]
                if isinstance(first_elem, (list, tuple)) and len(first_elem) > 0:
                    first_nested = first_elem[0]
                    if isinstance(first_nested, (list, tuple)) and len(first_nested) == 2:
                        bbox = item[0]
                        text_info = item[1]
                        text = (text_info[0] if isinstance(text_info, (list, tuple))
                                else str(text_info))
                        conf = (text_info[1] if isinstance(text_info, (list, tuple))
                                and len(text_info) > 1 else 0.5)

                        if text and len(str(text).strip()) > 0:
                            text = str(text).strip()
                            texts.append(text)
                            words.extend(text.split())
                            x_coords = [p[0] for p in bbox]
                            y_coords = [p[1] for p in bbox]
                            bboxes.append({
                                'text': text,
                                'x': int(min(x_coords)),
                                'y': int(min(y_coords)),
                                'w': int(max(x_coords) - min(x_coords)),
                                'h': int(max(y_coords) - min(y_coords)),
                            })
                            confidences.append(float(conf))
                        continue

            # PaddleOCR 2.x det=False 格式: [text, confidence]
            if len(item) >= 2:
                text = str(item[0]).strip() if item[0] else ""
                conf = float(item[1]) if item[1] else 0.0
            elif len(item) == 1:
                text = str(item[0]).strip() if item[0] else ""
                conf = 0.5
            else:
                continue

            if text and len(text) > 0:
                texts.append(text)
                words.extend(text.split())
                confidences.append(conf)

        full_text = '\n'.join(texts)
        avg_conf = float(np.mean(confidences)) if confidences else 0.0

        return OCRResult(
            text=full_text,
            words=words,
            bboxes=bboxes,
            confidence=avg_conf,
        )

    # ================================================================
    # 辅助方法
    # ================================================================

    @classmethod
    def _get_test_image(cls) -> np.ndarray:
        """获取用于健康检查的测试图片（缓存的白色图片）"""
        if cls._TEST_IMAGE is None:
            cls._TEST_IMAGE = np.ones((100, 200, 3), dtype=np.uint8) * 255
        return cls._TEST_IMAGE
