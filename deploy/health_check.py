#!/usr/bin/env python3
"""
独立健康检查脚本 — 在部署前/后验证所有组件是否正常

检测项目:
  1. Python 版本和平台信息
  2. 核心依赖: PyMuPDF, jieba, numpy, PIL
  3. OCR 引擎: PaddleOCR / EasyOCR 可用性
  4. SBERT 嵌入引擎: SentenceTransformer 可用性
  5. PDF 提取: PyMuPDF 文本和图片提取
  6. GPU: CUDA/MPS 可用性
  7. 模型缓存: 已下载的模型文件
  8. 磁盘空间: 输入/输出目录可用空间

用法:
    python deploy/health_check.py
    python deploy/health_check.py --input test_data/input --verbose
    python deploy/health_check.py --json  # JSON 格式输出（供 CI/CD）
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path

# 确保项目根目录在 sys.path 中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HealthChecker:
    """系统健康检查器"""

    def __init__(self, input_dir: str = None, verbose: bool = False):
        self.input_dir = input_dir
        self.verbose = verbose
        self.results = {}
        self.start_time = time.time()

    def check_all(self) -> dict:
        """运行所有检查"""
        logger.info("=" * 60)
        logger.info("系统健康检查开始")
        logger.info("=" * 60)

        checks = [
            ('python', self.check_python),
            ('core_deps', self.check_core_dependencies),
            ('pymupdf', self.check_pymupdf),
            ('ocr_engine', self.check_ocr_engine),
            ('sbert', self.check_sbert),
            ('gpu', self.check_gpu),
            ('model_cache', self.check_model_cache),
            ('pdf_extraction', self.check_pdf_extraction),
            ('disk_space', self.check_disk_space),
        ]

        all_ok = True
        for name, check_fn in checks:
            try:
                ok, msg = check_fn()
                self.results[name] = {'ok': ok, 'message': msg}
                status = '✅' if ok else '❌'
                logger.info(f"  {status} {name}: {msg}")
                if not ok:
                    all_ok = False
            except Exception as e:
                self.results[name] = {'ok': False, 'message': str(e)}
                logger.error(f"  ❌ {name}: 检查异常 - {e}")
                all_ok = False

        elapsed = time.time() - self.start_time
        logger.info("=" * 60)
        if all_ok:
            logger.info(f"✅ 所有检查通过 (耗时 {elapsed:.1f}s)")
        else:
            logger.warning(f"⚠️  部分检查未通过 (耗时 {elapsed:.1f}s)")
        logger.info("=" * 60)

        return self.results

    def check_python(self) -> tuple:
        import platform
        py_ver = platform.python_version()
        min_ver = (3, 9)
        current = tuple(map(int, py_ver.split('.')))
        if current >= min_ver:
            return True, f"Python {py_ver} ({platform.platform()})"
        return False, f"Python {py_ver} < 3.9 (不满足最低要求)"

    def check_core_dependencies(self) -> tuple:
        deps = []
        for name, import_name in [
            ('numpy', 'numpy'),
            ('PIL', 'PIL'),
            ('jieba', 'jieba'),
            ('imagehash', 'imagehash'),
        ]:
            try:
                mod = __import__(import_name)
                ver = getattr(mod, '__version__', 'N/A')
                deps.append(f"{name} {ver}")
            except ImportError:
                return False, f"缺少依赖: {name}"
        return True, ', '.join(deps)

    def check_pymupdf(self) -> tuple:
        try:
            import fitz
            return True, f"PyMuPDF {fitz.version[0]}"
        except ImportError:
            return False, "PyMuPDF 未安装"

    def check_ocr_engine(self) -> tuple:
        try:
            from image_analysis.image_ocr import ImageOCREngine
            engine = ImageOCREngine(use_gpu=False, engine='paddleocr')
            if engine.is_available:
                ok, msg = engine.health_check()
                if ok:
                    return True, f"PaddleOCR: {msg}"
                else:
                    # 尝试 EasyOCR
                    engine2 = ImageOCREngine(use_gpu=False, engine='easyocr')
                    if engine2.is_available:
                        ok2, msg2 = engine2.health_check()
                        if ok2:
                            return True, f"EasyOCR (回退): {msg2}"
                    return False, msg
            return False, "所有 OCR 引擎均不可用"
        except Exception as e:
            return False, f"OCR 引擎检查异常: {e}"

    def check_sbert(self) -> tuple:
        try:
            from sentence_transformers import SentenceTransformer
            return True, f"SentenceTransformer 可用"
        except ImportError:
            return False, "sentence-transformers 未安装"
        except Exception as e:
            return False, f"SBERT 检查异常: {e}"

    def check_gpu(self) -> tuple:
        try:
            import torch
            if torch.cuda.is_available():
                return True, f"CUDA: {torch.cuda.get_device_name(0)}"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return True, "Apple MPS"
            else:
                return True, "CPU 模式"
        except ImportError:
            return True, "PyTorch 未安装 (CPU 模式)"

    def check_model_cache(self) -> tuple:
        cache_dirs = [
            os.path.expanduser('~/.paddleocr'),
            os.path.expanduser('~/.paddlex'),
            os.path.expanduser('~/.cache/huggingface'),
            './models',
        ]
        found = []
        for d in cache_dirs:
            if os.path.exists(d):
                size_mb = sum(
                    os.path.getsize(os.path.join(r, f))
                    for r, _, files in os.walk(d)
                    for f in files
                ) / (1024 * 1024)
                found.append(f"{d} ({size_mb:.0f}MB)")

        if found:
            return True, f"缓存: {'; '.join(found)}"
        return True, "无本地模型缓存 (首次运行将自动下载)"

    def check_pdf_extraction(self) -> tuple:
        if not self.input_dir or not os.path.exists(self.input_dir):
            return True, "跳过 (未指定输入目录)"

        pdf_files = list(Path(self.input_dir).glob('*.pdf'))
        if not pdf_files:
            return True, "跳过 (无 PDF 文件)"

        try:
            import fitz
            test_file = str(pdf_files[0])
            doc = fitz.open(test_file)
            page_count = doc.page_count
            text = doc[0].get_text('text') if page_count > 0 else ''
            images = doc[0].get_images() if page_count > 0 else []
            doc.close()
            return True, (
                f"测试 PDF 提取正常: {os.path.basename(test_file)} "
                f"({page_count} 页, {len(text)} 字符, {len(images)} 图片)"
            )
        except Exception as e:
            return False, f"PDF 提取失败: {e}"

    def check_disk_space(self) -> tuple:
        try:
            import shutil
            cwd = os.getcwd()
            usage = shutil.disk_usage(cwd)
            free_gb = usage.free / (1024 ** 3)
            if free_gb < 1:
                return False, f"磁盘空间不足: {free_gb:.1f} GB"
            return True, f"磁盘可用: {free_gb:.1f} GB"
        except Exception:
            return True, "无法检查磁盘空间"


def main():
    parser = argparse.ArgumentParser(
        description='系统健康检查脚本'
    )
    parser.add_argument(
        '--input', type=str, default=None,
        help='测试 PDF 输入目录 (可选)'
    )
    parser.add_argument(
        '--verbose', action='store_true', default=False,
        help='详细输出'
    )
    parser.add_argument(
        '--json', action='store_true', default=False,
        help='JSON 格式输出 (供 CI/CD 使用)'
    )

    args = parser.parse_args()

    checker = HealthChecker(
        input_dir=args.input,
        verbose=args.verbose,
    )
    results = checker.check_all()

    if args.json:
        output = {
            'timestamp': datetime.now().isoformat(),
            'all_ok': all(r['ok'] for r in results.values()),
            'checks': results,
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))

    # 返回码
    all_ok = all(r['ok'] for r in results.values())
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
