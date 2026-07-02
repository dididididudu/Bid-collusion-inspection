"""
OCR 辅助函数 — 供 orchestrator 和 parallel_workers 共用

提取 PDF 页面/嵌入图片的文字，聚合为段落注入文本匹配管线。
"""

import os
import io
import logging
from typing import List

import numpy as np
from PIL import Image
import imagehash

from config import DetectionConfig
from data_structures import ChunkResult

logger = logging.getLogger(__name__)

# OCR 聚合段落的虚拟块索引
OCR_CHUNK_INDEX: int = 1000000


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

                    if ocr_result.confidence >= min_conf and ocr_result.text.strip():
                        cache.store_image_ocr_result(
                            doc_id=doc_id, page_num=page_num,
                            image_hash=phash, ocr_text=ocr_result.text,
                            ocr_words=ocr_result.words, bboxes=ocr_result.bboxes,
                            confidence=ocr_result.confidence,
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

                            if ocr_result.confidence < min_conf:
                                continue
                            if not ocr_result.text.strip():
                                continue

                            cache.store_image_ocr_result(
                                doc_id=doc_id, page_num=page_num,
                                image_hash=phash, ocr_text=ocr_result.text,
                                ocr_words=ocr_result.words, bboxes=ocr_result.bboxes,
                                confidence=ocr_result.confidence,
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
