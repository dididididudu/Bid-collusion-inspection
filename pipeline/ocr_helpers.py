"""
OCR 辅助函数 — 供 orchestrator 和 parallel_workers 共用

提取 PDF 页面/嵌入图片的文字，聚合为段落注入文本匹配管线。
"""

import os
import io
import logging
from typing import List, Dict

import numpy as np
from PIL import Image
import imagehash

from config import DetectionConfig
from data_structures import ChunkResult

logger = logging.getLogger(__name__)

# OCR 聚合段落的虚拟块索引
OCR_CHUNK_INDEX: int = 1000000


def _compute_non_text_hash(img_array: np.ndarray, bboxes: List[Dict]) -> str:
    """计算去除文字区域后的图片哈希

    用 OCR bbox 坐标将文字区域涂白，再计算 pHash。
    用于 PS 嫌疑检测：如果两张图文字不同但 non_text_hash 相同，
    说明背景/结构未变，只有文字被修改过。

    Args:
        img_array: 原始图片 (H, W, 3) numpy 数组
        bboxes: OCR bbox 列表 [{x, y, w, h}, ...]

    Returns:
        去除文字后的 pHash 字符串
    """
    if not bboxes or img_array.size == 0:
        return ''

    masked = img_array.copy()
    h_img, w_img = masked.shape[:2]

    for bbox in bboxes:
        x = max(0, int(bbox.get('x', 0)))
        y = max(0, int(bbox.get('y', 0)))
        w = min(int(bbox.get('w', 0)), w_img - x)
        h = min(int(bbox.get('h', 0)), h_img - y)
        if w > 0 and h > 0:
            # 用区域周边的平均色填充（取四边采样）
            edge_pixels = []
            sample_count = min(10, max(1, (w + h) // 10))
            for s in range(sample_count):
                sx = x + (w * s) // sample_count
                # 上边缘
                if y > 0:
                    edge_pixels.append(masked[y - 1, sx])
                # 下边缘
                if y + h < h_img - 1:
                    edge_pixels.append(masked[y + h, sx])
            bg_color = np.median(edge_pixels, axis=0).astype(np.uint8) if edge_pixels else np.array([255, 255, 255], dtype=np.uint8)
            masked[y:y + h, x:x + w] = bg_color

    return str(imagehash.phash(Image.fromarray(masked)))


def _make_thumbnail(img_array: np.ndarray, size: int = 64) -> bytes:
    """生成图片的缩略图（JPEG 格式），供 ORB 特征匹配和直方图比较使用

    Args:
        img_array: numpy 数组 (H, W, 3)
        size: 缩略图边长（保持宽高比）

    Returns:
        JPEG 字节流
    """
    if img_array.size == 0:
        return b''
    img = Image.fromarray(img_array)
    img.thumbnail((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    # JPEG 质量 85 折中体积和质量
    img.save(buf, format='JPEG', quality=85)
    return buf.getvalue()


def ocr_pages(
    file_path: str,
    doc_id: str,
    page_count: int,
    cache,
    config: DetectionConfig,
    ocr_engine,
    force: bool = False,
) -> int:
    """对 PDF 中的图片运行 OCR 提取文字

    Args:
        file_path: PDF 文件路径
        doc_id: 文档 ID
        page_count: 总页数
        cache: DocumentCache 实例
        config: DetectionConfig
        ocr_engine: ImageOCREngine 实例
        force: True=扫描版全页OCR, False=嵌入图片OCR

    Returns:
        成功 OCR 的图片数
    """
    if not config.ENABLE_OCR:
        return 0
    if not ocr_engine.is_available:
        logger.debug("OCR 引擎不可用，跳过图片文字提取")
        return 0

    if not force:
        existing_ocr = cache.load_image_ocr_results(doc_id)
        if existing_ocr:
            logger.debug(f"OCR: {os.path.basename(file_path)} 已有 {len(existing_ocr)} 条结果，跳过")
            return len(existing_ocr)

    import fitz

    logger.info(
        f"OCR: {'扫描版全页' if force else '嵌入图片'}模式 "
        f"{os.path.basename(file_path)} ({page_count} 页)..."
    )

    ocr_count = 0
    sample_step = config.OCR_SAMPLE_STEP
    min_conf = config.OCR_MIN_CONFIDENCE
    min_img_size = getattr(config, 'IMAGE_MIN_SIZE', 50)

    try:
        doc = fitz.open(file_path)
    except Exception as e:
        logger.error(f"OCR: 无法打开 PDF ({file_path}): {e}")
        return 0

    try:
        for page_num in range(0, page_count, sample_step):
            try:
                page = doc[page_num]

                if force:
                    # 扫描版：渲染整页 → OCR
                    pix = page.get_pixmap(dpi=150)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    phash = str(imagehash.phash(img))
                    img_array = np.array(img)

                    ocr_result = ocr_engine.extract(img_array)
                    ocr_result.image_hash = phash
                    ocr_result.image_width = img.width
                    ocr_result.image_height = img.height
                    ocr_result.thumbnail = _make_thumbnail(img_array)
                    ocr_result.non_text_hash = _compute_non_text_hash(
                        img_array, ocr_result.bboxes
                    )

                    if ocr_result.confidence >= min_conf and ocr_result.text.strip():
                        cache.store_image_ocr_result(
                            doc_id=doc_id, page_num=page_num,
                            image_hash=phash, ocr_text=ocr_result.text,
                            ocr_words=ocr_result.words, bboxes=ocr_result.bboxes,
                            confidence=ocr_result.confidence,
                            non_text_hash=ocr_result.non_text_hash,
                            image_width=ocr_result.image_width,
                            image_height=ocr_result.image_height,
                            thumbnail=ocr_result.thumbnail,
                        )
                        ocr_count += 1
                else:
                    # 文本版：渲染页面 → 裁剪图片区域 → OCR
                    image_info_list = page.get_image_info()
                    if not image_info_list:
                        continue

                    valid_images = [
                        i for i in image_info_list
                        if i.get('width', 0) >= min_img_size
                        and i.get('height', 0) >= min_img_size
                    ]
                    if not valid_images:
                        continue

                    OCR_DPI = 200
                    scale = OCR_DPI / 72.0
                    pix = page.get_pixmap(dpi=OCR_DPI)
                    full_img = Image.open(io.BytesIO(pix.tobytes("png")))

                    for info in valid_images:
                        try:
                            bbox = info.get('bbox', (0, 0, 0, 0))
                            x0, y0, x1, y1 = bbox
                            crop = full_img.crop((
                                int(x0 * scale), int(y0 * scale),
                                int(x1 * scale), int(y1 * scale),
                            ))
                            if crop.size[0] < 10 or crop.size[1] < 10:
                                continue

                            phash = str(imagehash.phash(crop))
                            img_array = np.array(crop)

                            ocr_result = ocr_engine.extract(img_array)
                            ocr_result.image_hash = phash
                            ocr_result.image_width = crop.width
                            ocr_result.image_height = crop.height
                            ocr_result.thumbnail = _make_thumbnail(img_array)
                            ocr_result.non_text_hash = _compute_non_text_hash(
                                img_array, ocr_result.bboxes
                            )

                            if ocr_result.confidence < min_conf:
                                continue
                            if not ocr_result.text.strip():
                                continue

                            cache.store_image_ocr_result(
                                doc_id=doc_id, page_num=page_num,
                                image_hash=phash, ocr_text=ocr_result.text,
                                ocr_words=ocr_result.words, bboxes=ocr_result.bboxes,
                                confidence=ocr_result.confidence,
                                non_text_hash=ocr_result.non_text_hash,
                                image_width=ocr_result.image_width,
                                image_height=ocr_result.image_height,
                                thumbnail=ocr_result.thumbnail,
                            )
                            ocr_count += 1

                        except Exception as e:
                            logger.debug(f"OCR: 第 {page_num} 页裁剪区失败: {e}")
                            continue

            except Exception as e:
                logger.debug(f"OCR: 第 {page_num} 页失败 ({os.path.basename(file_path)}): {e}")
                continue

    finally:
        doc.close()

    if ocr_count > 0:
        logger.info(f"OCR: {os.path.basename(file_path)} — {ocr_count} 张图片成功提取文字")

    return ocr_count


def aggregate_ocr_paragraphs(
    doc_id: str,
    page_count: int,
    cache,
    extractor,
    text_processor,
) -> int:
    """将 OCR 结果聚合为段落并注入文本匹配管线

    Returns:
        创建的段落数量
    """
    import jieba

    ocr_results = cache.load_image_ocr_results(doc_id)
    if not ocr_results:
        return 0

    ocr_sorted = sorted(ocr_results, key=lambda r: r.get('page_num', 0))
    all_text = "\n".join(
        r['ocr_text'] for r in ocr_sorted
        if r.get('ocr_text', '').strip()
    )
    if not all_text.strip():
        return 0

    logger.info(f"OCR 聚合: {doc_id} — {len(all_text)} 字符, {len(ocr_results)} 页 OCR 结果")

    # 分段
    paragraphs = extractor._split_paragraphs(all_text)
    if not paragraphs:
        return 0

    # 分词 + 词哈希缓存
    stopwords = extractor.stopwords
    all_tokens = [w for w in jieba.cut(all_text) if w not in stopwords and len(w) > 1]

    word_hash_cache = {}
    for w in set(all_tokens):
        word_hash_cache[w] = [hf(w) for hf in extractor._minhash_funcs]

    # 每段 MinHash
    paragraph_hashes = []
    for para in paragraphs:
        para_words = [w for w in jieba.cut(para) if w not in stopwords and len(w) > 1]
        para_hash = extractor._compute_minhash_cached(para_words, word_hash_cache)
        paragraph_hashes.append(para_hash)

    # SimHash
    simhash = extractor._compute_simhash_from_tokens(all_tokens) if all_tokens else "0" * 16

    # 报价
    quotes = extractor._extract_quotes(all_text)

    # 虚拟 ChunkResult
    chunk_result = ChunkResult(
        doc_id=doc_id,
        chunk_index=OCR_CHUNK_INDEX,
        start_page=0,
        end_page=page_count - 1 if page_count > 0 else 0,
        text=all_text,
        paragraphs=paragraphs,
        paragraph_hashes=paragraph_hashes,
        simhash=simhash,
        quotes=quotes,
        image_hashes=[],
    )
    cache.store_chunk(chunk_result)

    # 更新 BidFeature
    feature = cache.load_document(doc_id)
    if feature:
        doc_minhash = text_processor._aggregate_minhash(paragraph_hashes)
        feature.text_length = len(all_text)
        feature.text_simhash = simhash
        feature.doc_minhash = doc_minhash
        feature.chunk_count = 1
        cache.store_document(feature)
        logger.info(f"OCR 聚合完成: {doc_id} → {len(paragraphs)} 段, MinHash 已填充")
    else:
        logger.warning(f"OCR 聚合: {doc_id} 文档特征未找到")

    return len(paragraphs)
