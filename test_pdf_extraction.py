"""
PDF 提取测试脚本
测试 test_data/input/ 中的 PDF 文件能否正确提取文本和图片
每个测试文件只包含 1 页和若干嵌入图片，以此验证提取功能。
"""

import os
import sys
import io
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_fitz_available():
    """测试 PyMuPDF 是否可用"""
    try:
        import fitz
        logger.info(f"✅ PyMuPDF (fitz) 已安装, 版本: {fitz.version}")
        return True
    except ImportError:
        logger.error("❌ PyMuPDF (fitz) 未安装!")
        return False


def test_paddleocr_available():
    """测试 PaddleOCR 是否可用"""
    try:
        from paddleocr import PaddleOCR
        logger.info("✅ PaddleOCR 已安装")
        return True
    except ImportError:
        logger.warning("⚠️  PaddleOCR 未安装")
        return False


def test_easyocr_available():
    """测试 EasyOCR 是否可用"""
    try:
        import easyocr
        logger.info("✅ EasyOCR 已安装")
        return True
    except ImportError:
        logger.warning("⚠️  EasyOCR 未安装")
        return False


def test_pdf_text_extraction(pdf_path):
    """测试从 PDF 提取文本内容"""
    import fitz

    doc = fitz.open(pdf_path)
    try:
        page_count = doc.page_count
        logger.info(f"  PDF 页数: {page_count}")

        total_text = ""
        for page_num in range(page_count):
            page = doc[page_num]
            text = page.get_text("text")
            total_text += text
            logger.info(f"  第 {page_num + 1} 页: {len(text)} 字符")

        logger.info(f"  总文本: {len(total_text)} 字符")
        if total_text.strip():
            preview = total_text.strip()[:200].replace('\n', '\\n')
            logger.info(f"  文本预览: {preview}...")
        return page_count, total_text
    finally:
        doc.close()


def test_pdf_image_extraction(pdf_path):
    """测试从 PDF 提取嵌入图片"""
    import fitz
    from PIL import Image
    import imagehash

    doc = fitz.open(pdf_path)
    try:
        total_images = 0
        valid_images = 0

        for page_num in range(doc.page_count):
            page = doc[page_num]

            # 方法1: get_images() - 嵌入图片列表
            image_list = page.get_images(full=True)
            logger.info(f"  第 {page_num + 1} 页: get_images() 找到 {len(image_list)} 个嵌入图片")

            for img_info in image_list:
                total_images += 1
                xref = img_info[0]
                width, height = img_info[2], img_info[3]
                logger.info(f"    图片 {total_images}: xref={xref}, 尺寸={width}x{height}")

                if width >= 100 and height >= 100:
                    try:
                        base_image = doc.extract_image(xref)
                        if base_image:
                            image_bytes = base_image.get("image")
                            if image_bytes and len(image_bytes) >= 1024:
                                img = Image.open(io.BytesIO(image_bytes))
                                phash = str(imagehash.phash(img))
                                logger.info(f"      ✅ 有效图片: 格式={img.format}, "
                                          f"模式={img.mode}, pHash={phash}")
                                valid_images += 1
                    except Exception as e:
                        logger.warning(f"      ⚠️  提取失败: {e}")

            # 方法2: get_image_info() - 页面图片区域信息
            image_info_list = page.get_image_info()
            valid_info = [i for i in image_info_list
                         if i.get('width', 0) >= 100 and i.get('height', 0) >= 100]
            logger.info(f"    get_image_info() 找到 {len(image_info_list)} 个图片区域 "
                      f"({len(valid_info)} 个有效)")

        logger.info(f"  总计: {total_images} 个嵌入图片, {valid_images} 个有效(>=100x100)")
        return total_images, valid_images
    finally:
        doc.close()


def test_ocr_engine():
    """测试 OCR 引擎初始化和可用性"""
    from image_analysis.image_ocr import ImageOCREngine

    # 测试 PaddleOCR
    logger.info("\n--- 测试 PaddleOCR (det=False, 仅识别) ---")
    try:
        engine = ImageOCREngine(use_gpu=False, engine="paddleocr")
        if engine.is_available:
            logger.info(f"✅ PaddleOCR 可用 (引擎类型: {engine._engine_type})")
        else:
            logger.warning("⚠️  PaddleOCR 不可用")
    except Exception as e:
        logger.error(f"❌ PaddleOCR 初始化异常: {e}")

    # 测试 EasyOCR
    logger.info("\n--- 测试 EasyOCR (备选) ---")
    try:
        engine = ImageOCREngine(use_gpu=False, engine="easyocr")
        if engine.is_available:
            logger.info(f"✅ EasyOCR 可用 (引擎类型: {engine._engine_type})")
        else:
            logger.warning("⚠️  EasyOCR 不可用")
    except Exception as e:
        logger.error(f"❌ EasyOCR 初始化异常: {e}")

    # 测试自动模式
    logger.info("\n--- 测试自动模式 (默认: paddleocr) ---")
    try:
        engine = ImageOCREngine(use_gpu=False)
        if engine.is_available:
            logger.info(f"✅ 自动模式可用 (引擎类型: {engine._engine_type})")
        else:
            logger.warning("⚠️  自动模式不可用")
    except Exception as e:
        logger.error(f"❌ 自动模式初始化异常: {e}")


def test_ocr_on_image(pdf_path):
    """测试对 PDF 中嵌入图片进行 OCR"""
    import fitz
    import numpy as np
    from PIL import Image
    from image_analysis.image_ocr import ImageOCREngine

    logger.info(f"\n--- OCR 测试: {os.path.basename(pdf_path)} ---")

    engine = ImageOCREngine(use_gpu=False, engine="paddleocr")
    if not engine.is_available:
        logger.warning("OCR 引擎不可用，跳过 OCR 测试")
        return

    doc = fitz.open(pdf_path)
    try:
        ocr_count = 0
        for page_num in range(doc.page_count):
            page = doc[page_num]

            # 渲染页面为图片
            OCR_DPI = 200
            scale = OCR_DPI / 72.0
            pix = page.get_pixmap(dpi=OCR_DPI)
            full_img = Image.open(io.BytesIO(pix.tobytes("png")))

            # 获取有效图片区域
            image_info_list = page.get_image_info()
            valid_images = [
                i for i in image_info_list
                if i.get('width', 0) >= 100 and i.get('height', 0) >= 100
            ]

            logger.info(f"  第 {page_num + 1} 页: {len(valid_images)} 个有效图片区域")

            for idx, info in enumerate(valid_images[:5]):  # 只测试前5个
                bbox = info.get('bbox', (0, 0, 0, 0))
                x0, y0, x1, y1 = bbox
                px0 = int(x0 * scale)
                py0 = int(y0 * scale)
                px1 = int(x1 * scale)
                py1 = int(y1 * scale)

                crop = full_img.crop((px0, py0, px1, py1))
                if crop.size[0] < 10 or crop.size[1] < 10:
                    continue

                img_array = np.array(crop)
                result = engine.extract(img_array)

                if result.text.strip():
                    logger.info(f"    图片 {idx + 1} ({crop.size}): "
                              f"置信度={result.confidence:.3f}, "
                              f"文字=\"{result.text[:100]}\"")
                    ocr_count += 1
                else:
                    logger.info(f"    图片 {idx + 1} ({crop.size}): 无文字")

        logger.info(f"  OCR 结果: {ocr_count} 张图片提取到文字")
        return ocr_count
    finally:
        doc.close()


def test_config_loading():
    """测试配置加载"""
    from config import DetectionConfig

    logger.info("\n--- 配置加载测试 ---")

    # 默认配置
    config = DetectionConfig()
    logger.info(f"默认 OCR_ENGINE: {config.OCR_ENGINE}")
    logger.info(f"默认 ENABLE_OCR: {config.ENABLE_OCR}")

    # 从 JSON 加载
    config_files = ["test_config.json", "test_config_ocr.json", "test_config_full.json"]
    for cf in config_files:
        if os.path.exists(cf):
            try:
                c = DetectionConfig.from_json(cf)
                logger.info(f"{cf}: OCR_ENGINE={c.OCR_ENGINE}, "
                          f"ENABLE_OCR={c.ENABLE_OCR}")
            except Exception as e:
                logger.error(f"{cf} 加载失败: {e}")


def test_full_pipeline():
    """测试完整的 Phase 1 提取流程 (单文档，串行模式)"""
    from config import DetectionConfig
    from extraction.pdf_extractor import PyMuPDFExtractor
    from extraction.text_processor import ChunkedTextProcessor
    from extraction.feature_cache import DocumentCache
    from image_analysis.image_ocr import ImageOCREngine

    input_dir = os.path.join("test_data", "input")
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]

    if not pdf_files:
        logger.error(f"test_data/input/ 中没有 PDF 文件!")
        return

    logger.info(f"\n--- 完整 Pipeline 测试 ({len(pdf_files)} 个文件) ---")

    config = DetectionConfig()
    config.ENABLE_OCR = True
    config.OCR_ENGINE = "paddleocr"
    config.DISABLE_CACHE = True
    config.PHASE1_WORKERS = 1

    cache_dir = "./cache_test"
    os.makedirs(cache_dir, exist_ok=True)

    cache = DocumentCache(cache_dir, config)
    extractor = PyMuPDFExtractor(config)
    text_processor = ChunkedTextProcessor(config)
    ocr_engine = ImageOCREngine(use_gpu=False, engine=config.OCR_ENGINE)

    try:
        for pdf_file in sorted(pdf_files):
            file_path = os.path.join(input_dir, pdf_file)
            logger.info(f"\n处理: {pdf_file}")

            # Phase 0: 元数据
            metadata, page_count, is_scanned = extractor.extract_metadata(file_path)
            doc_id = extractor._generate_doc_id(file_path)
            file_size = os.path.getsize(file_path)

            logger.info(f"  doc_id: {doc_id}")
            logger.info(f"  页数: {page_count}")
            logger.info(f"  大小: {file_size} 字节")
            logger.info(f"  扫描版: {is_scanned}")
            logger.info(f"  作者: {metadata.author}")
            logger.info(f"  创建工具: {metadata.creator}")
            logger.info(f"  生成工具: {metadata.producer}")

            # Phase 1: 文本提取
            chunks = []
            for chunk in extractor.extract_chunks(file_path, config.CHUNK_PAGE_SIZE, 0):
                cache.store_chunk(chunk)
                chunks.append(chunk)
                logger.info(f"  块 {chunk.chunk_index}: 页 {chunk.start_page}-{chunk.end_page}, "
                          f"{len(chunk.paragraphs)} 段, "
                          f"{len(chunk.image_hashes)} 图片哈希, "
                          f"{len(chunk.text)} 字符")

            # 聚合特征
            if chunks:
                feature = text_processor.aggregate_chunks(
                    doc_id=doc_id, filename=pdf_file, file_size=file_size,
                    chunks=chunks, metadata=metadata,
                    is_scanned=False, page_count=page_count,
                )

                all_img_hashes = set()
                for c in chunks:
                    all_img_hashes.update(c.image_hashes)
                feature.image_hashes = list(all_img_hashes)

                cache.store_document(feature)

                logger.info(f"  聚合结果:")
                logger.info(f"    文本长度: {feature.text_length}")
                logger.info(f"    段落数: {len(feature.paragraphs)}")
                logger.info(f"    图片哈希数: {len(feature.image_hashes)}")
                logger.info(f"    报价数: {len(feature.quotes)}")
                logger.info(f"    SimHash: {feature.text_simhash[:16] if feature.text_simhash else 'N/A'}...")

                # 验证结果
                logger.info(f"\n  验证:")
                # 检查页数
                assert page_count == 1, f"期望 1 页, 实际 {page_count} 页"
                logger.info(f"    ✅ 页数正确: {page_count}")

                # 检查文本内容
                assert feature.text_length > 0, "文本内容为空!"
                logger.info(f"    ✅ 文本提取成功: {feature.text_length} 字符")

                # 检查图片提取
                assert len(feature.image_hashes) > 0, "未提取到任何图片!"
                logger.info(f"    ✅ 图片提取成功: {len(feature.image_hashes)} 哈希")

                # 检查段落（段落存储在 SQLite 中，不在 BidFeature.paragraphs 字段）
                total_paras = sum(len(c.paragraphs) for c in chunks)
                logger.info(f"    ✅ 段落分割成功: {total_paras} 段 (存储在 SQLite)")

                # 尝试 OCR（如果有 OCR 引擎）
                if ocr_engine.is_available:
                    logger.info(f"\n  OCR 测试:")
                    import fitz
                    import numpy as np
                    from PIL import Image

                    doc = fitz.open(file_path)
                    ocr_count = 0
                    try:
                        for page_num in range(page_count):
                            page = doc[page_num]
                            image_info_list = page.get_image_info()
                            valid = [i for i in image_info_list
                                    if i.get('width', 0) >= 100 and i.get('height', 0) >= 100]

                            if valid:
                                OCR_DPI = 200
                                scale = OCR_DPI / 72.0
                                pix = page.get_pixmap(dpi=OCR_DPI)
                                full_img = Image.open(io.BytesIO(pix.tobytes("png")))

                                for info in valid[:3]:  # 只测试前3个
                                    bbox = info.get('bbox', (0, 0, 0, 0))
                                    x0, y0, x1, y1 = bbox
                                    crop = full_img.crop((
                                        int(x0 * scale), int(y0 * scale),
                                        int(x1 * scale), int(y1 * scale)
                                    ))
                                    if crop.size[0] >= 10 and crop.size[1] >= 10:
                                        result = ocr_engine.extract(np.array(crop))
                                        if result.text.strip():
                                            ocr_count += 1
                                            logger.info(f"      {ocr_count}. \"{result.text[:80]}...\" "
                                                      f"(conf={result.confidence:.2f})")
                    finally:
                        doc.close()

                    if ocr_count > 0:
                        logger.info(f"    ✅ OCR 成功: {ocr_count} 张图片")
                    else:
                        logger.warning(f"    ⚠️  OCR 未提取到文字 (可能是图片不含文字)")
                else:
                    logger.warning(f"    ⚠️  OCR 引擎不可用，跳过 OCR 测试")

        logger.info(f"\n{'='*50}")
        logger.info("✅ 所有测试通过!")
        logger.info(f"{'='*50}")

    except AssertionError as e:
        logger.error(f"❌ 验证失败: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ 测试异常: {e}", exc_info=True)
        raise
    finally:
        cache.close()
        # 清理测试缓存
        import shutil
        if os.path.exists(cache_dir):
            try:
                shutil.rmtree(cache_dir)
            except Exception:
                pass


def main():
    logger.info("=" * 60)
    logger.info("PDF 提取功能测试")
    logger.info("=" * 60)

    # 1. 检查依赖
    logger.info("\n=== 第1步: 检查依赖库 ===")
    fitz_ok = test_fitz_available()
    paddle_ok = test_paddleocr_available()
    easy_ok = test_easyocr_available()

    if not fitz_ok:
        logger.error("PyMuPDF 是必需依赖, 请安装: pip install PyMuPDF")
        sys.exit(1)

    # 2. 检查测试文件
    logger.info("\n=== 第2步: 检查测试文件 ===")
    input_dir = os.path.join("test_data", "input")
    if not os.path.exists(input_dir):
        logger.error(f"测试目录不存在: {input_dir}")
        sys.exit(1)

    pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]
    logger.info(f"找到 {len(pdf_files)} 个 PDF 文件:")
    for f in sorted(pdf_files):
        file_path = os.path.join(input_dir, f)
        size_kb = os.path.getsize(file_path) / 1024
        logger.info(f"  - {f} ({size_kb:.1f} KB)")

    # 3. 测试文本提取
    logger.info("\n=== 第3步: 测试 PDF 文本提取 ===")
    for f in sorted(pdf_files):
        file_path = os.path.join(input_dir, f)
        logger.info(f"\n文件: {f}")
        page_count, text = test_pdf_text_extraction(file_path)

    # 4. 测试图片提取
    logger.info("\n=== 第4步: 测试 PDF 图片提取 ===")
    for f in sorted(pdf_files):
        file_path = os.path.join(input_dir, f)
        logger.info(f"\n文件: {f}")
        total_img, valid_img = test_pdf_image_extraction(file_path)

    # 5. 测试 OCR 引擎
    logger.info("\n=== 第5步: 测试 OCR 引擎 ===")
    test_ocr_engine()

    # 6. 测试 OCR 实际使用
    if paddle_ok or easy_ok:
        logger.info("\n=== 第6步: 测试 OCR 实际提取 ===")
        for f in sorted(pdf_files):
            file_path = os.path.join(input_dir, f)
            test_ocr_on_image(file_path)

    # 7. 测试配置
    logger.info("\n=== 第7步: 测试配置加载 ===")
    test_config_loading()

    # 8. 完整 Pipeline 测试
    logger.info("\n=== 第8步: 完整 Pipeline 测试 ===")
    test_full_pipeline()


if __name__ == "__main__":
    main()
