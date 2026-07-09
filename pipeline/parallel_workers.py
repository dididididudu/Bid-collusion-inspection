"""
并行 Worker 函数 — 供 ProcessPoolExecutor / ThreadPoolExecutor 调用

所有函数必须是模块级（可 pickle），参数仅使用可 pickle 类型。
每个 worker 创建自己的 SQLite 连接，WAL 模式 + busy_timeout 处理写冲突。
"""

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pkg_resources')
warnings.filterwarnings('ignore', category=UserWarning, module='jieba')
import os
import logging
import json
import time
import sqlite3
import threading as _threading
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from data_structures import (
    BidFeature, ChunkResult, PairwiseResult, EvidenceChain,
    TextEvidence, MetadataEvidence, ImageEvidence, QuoteSignature,
)
from extraction.feature_cache import DocumentCache
from extraction.pdf_extractor import PyMuPDFExtractor
from extraction.text_processor import ChunkedTextProcessor
from matching.paragraph_matcher import ParagraphMatcher
from pipeline.evidence_builder import (
    build_metadata_evidence, build_text_evidence,
)
from image_analysis.image_ocr import OCRResult
from image_analysis.image_matcher import ImageMatcher

logger = logging.getLogger(__name__)
_tls = _threading.local()


def _begin_immediate_with_retry(conn, max_retries=10, base_delay=0.5):
    """带重试的 BEGIN IMMEDIATE，缓解多进程写冲突

    ProcessPoolExecutor 多 worker 同时写同一个 SQLite 时，
    即使 busy_timeout 也可能竞争失败。重试 + 指数退避更可靠。
    """
    for attempt in range(max_retries):
        try:
            conn.execute("BEGIN IMMEDIATE")
            return
        except sqlite3.OperationalError as e:
            if 'database is locked' in str(e) and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.debug(f"BEGIN IMMEDIATE 冲突 (尝试 {attempt+1}/{max_retries})，等待 {delay:.1f}s")
                time.sleep(delay)
                continue
            raise


def _get_thread_cache(db_dir, config):
    if not hasattr(_tls, 'cache'):
        from extraction.feature_cache import DocumentCache
        _tls.cache = DocumentCache(db_dir, config)
        _tls.doc_cache = {}
        _tls.matcher = None
    return _tls.cache


def _get_thread_doc(doc_id, cache):
    if doc_id not in _tls.doc_cache:
        _tls.doc_cache[doc_id] = cache.load_document(doc_id)
    return _tls.doc_cache[doc_id]


def _build_image_evidence(doc_a, doc_b, cache, config=None, semantic_matcher=None):
    evidence = ImageEvidence()
    hashes_a = doc_a.image_hashes
    hashes_b = doc_b.image_hashes
    common_exact = list(set(hashes_a) & set(hashes_b))
    evidence.common_image_count = len(common_exact)
    evidence.common_image_hashes = common_exact

    ocr_a = cache.load_image_ocr_results(doc_a.doc_id)
    ocr_b = cache.load_image_ocr_results(doc_b.doc_id)
    if ocr_a: evidence.ocr_results_a = ocr_a
    if ocr_b: evidence.ocr_results_b = ocr_b

    ocr_objects_a = [
        OCRResult(text=r['ocr_text'], words=r['ocr_words'], bboxes=r['bboxes'],
                  confidence=r['confidence'], image_hash=r.get('image_hash', ''),
                  non_text_hash=r.get('non_text_hash', ''),
                  image_width=r.get('image_width', 0), image_height=r.get('image_height', 0),
                  thumbnail=r.get('thumbnail', b''))
        for r in ocr_a
    ]
    ocr_objects_b = [
        OCRResult(text=r['ocr_text'], words=r['ocr_words'], bboxes=r['bboxes'],
                  confidence=r['confidence'], image_hash=r.get('image_hash', ''),
                  non_text_hash=r.get('non_text_hash', ''),
                  image_width=r.get('image_width', 0), image_height=r.get('image_height', 0),
                  thumbnail=r.get('thumbnail', b''))
        for r in ocr_b
    ]

    boilerplate_hashes = set(config.IMAGE_BOILERPLATE_HASHES) if config and config.IMAGE_BOILERPLATE_HASHES else None
    matcher = ImageMatcher(semantic_matcher=semantic_matcher)
    match_result = matcher.analyze(
        hashes_a=hashes_a, hashes_b=hashes_b,
        ocr_results_a=ocr_objects_a or None, ocr_results_b=ocr_objects_b or None,
        boilerplate_hashes=boilerplate_hashes,
    )

    evidence.exact_image_count = match_result.exact_image_count
    evidence.near_identical_count = match_result.near_identical_count
    evidence.similar_image_count = match_result.similar_image_count
    evidence.ps_suspicious = match_result.ps_suspicious
    evidence.ps_suspicious_count = match_result.ps_suspicious_count
    evidence.shared_typos = match_result.shared_typos
    evidence.shared_typo_count = match_result.shared_typo_count
    evidence.text_identical_count = match_result.text_identical_count
    evidence.text_similar_count = match_result.text_similar_count
    evidence.image_risk_score = match_result.image_risk_score
    evidence.image_risk_factors = match_result.image_risk_factors

    # 填充 matched_image_pairs（含缩略图 base64，供 PDF 报告 / API 展示）
    for v in match_result.image_verdicts:
        thumb_a_b64 = _thumbnail_to_base64(v.sig_a.thumbnail)
        thumb_b_b64 = _thumbnail_to_base64(v.sig_b.thumbnail)
        ocr_text_a = _find_ocr_text_by_hash(ocr_objects_a, v.sig_a.phash or v.sig_a.dhash)
        ocr_text_b = _find_ocr_text_by_hash(ocr_objects_b, v.sig_b.phash or v.sig_b.dhash)
        evidence.matched_image_pairs.append({
            'source_a': v.sig_a.source_id,
            'source_b': v.sig_b.source_id,
            'phash_dist': v.phash_dist,
            'dhash_dist': v.dhash_dist,
            'orb_match_ratio': round(v.orb_match_ratio, 3),
            'histogram_correlation': round(v.histogram_correlation, 3),
            'confidence': round(v.confidence, 3),
            'reasons': v.reasons,
            'thumbnail_base64_a': thumb_a_b64,
            'thumbnail_base64_b': thumb_b_b64,
            'ocr_text_a': ocr_text_a,
            'ocr_text_b': ocr_text_b,
            'l1_pass': v.l1_pass,
            'l2_pass': v.l2_pass,
            'l3_pass': v.l3_pass,
        })
    evidence.matched_text_pairs = match_result.text_matches
    evidence.ps_detail_list = match_result.ps_details
    return evidence


def _thumbnail_to_base64(thumb_bytes: bytes) -> str:
    """将缩略图字节转为 base64 data URI"""
    if not thumb_bytes:
        return ''
    import base64
    encoded = base64.b64encode(thumb_bytes).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded}"


def _find_ocr_text_by_hash(ocr_objects: list, hash_val: str) -> str:
    """根据哈希值在 OCR 结果列表中找对应文本"""
    if not hash_val:
        return ''
    for obj in ocr_objects:
        img_hash = obj.image_hash if hasattr(obj, 'image_hash') else ''
        text = obj.text if hasattr(obj, 'text') else ''
        if img_hash and (hash_val in img_hash or img_hash in hash_val):
            return text[:200]
    return ''


def extract_single_worker(args: tuple) -> dict:
    """Phase 1 worker: 提取单个文档的特征"""
    file_path, config_dict, db_dir, gpu_manager_client = args
    from config import DetectionConfig
    config = DetectionConfig(**config_dict)
    ocr_engine = None
    use_gpu_mgr = gpu_manager_client is not None and gpu_manager_client.enabled
    t_worker = time.time()

    try:
        from extraction.contact_extractor import extract_contacts_from_sqlite
        cache = _get_thread_cache(db_dir, config)
        extractor = PyMuPDFExtractor(config)
        text_processor = ChunkedTextProcessor(config)

        metadata, page_count, is_scanned = extractor.extract_metadata(file_path)
        doc_id = extractor._generate_doc_id(file_path)
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)

        logger.info(f"[Worker] 开始: {filename} ({page_count} 页, "
                    f"{file_size / 1024:.0f} KB)")

        existing_chunks = cache.load_document_chunks(doc_id)
        processed_chunks = {c.chunk_index for c in existing_chunks}

        # === 文本版 PDF ===
        start_page = max(
            (max(processed_chunks) + 1) * config.CHUNK_PAGE_SIZE
            if processed_chunks else 0, 0
        )

        if start_page >= page_count and page_count > 0:
            logger.info(f"[Worker] 已完全提取: {filename}")
            try:
                fp = extract_contacts_from_sqlite(doc_id, cache)
                cache.store_contact_fingerprint(doc_id, fp.to_json())
            except Exception:
                pass
            cache.conn.commit()
            cache.close()
            return {"doc_id": doc_id, "filename": filename, "success": True}

        chunks = []
        _begin_immediate_with_retry(cache.conn, max_retries=10)
        try:
            for chunk_result in extractor.extract_chunks(file_path, config.CHUNK_PAGE_SIZE, start_page):
                cache.store_chunk(chunk_result, conn=cache.conn)
                chunks.append(chunk_result)
            cache.conn.execute("COMMIT")
        except Exception:
            cache.conn.execute("ROLLBACK")
            raise

        if chunks:
            feature = text_processor.aggregate_chunks(
                doc_id=doc_id, filename=filename, file_size=file_size,
                chunks=chunks, metadata=metadata,
                is_scanned=False, page_count=page_count,
            )
            all_img_hashes = set()
            for c in chunks:
                all_img_hashes.update(c.image_hashes)
            feature.image_hashes = list(all_img_hashes)
            cache.store_document(feature)

            if config.ENABLE_OCR and not use_gpu_mgr:
                from image_analysis.image_ocr import ImageOCREngine
                ocr_engine = ImageOCREngine(
                    use_gpu=config.USE_GPU, engine=config.OCR_ENGINE,
                    model_dir=config.OCR_MODEL_DIR, offline=config.OCR_OFFLINE_MODE,
                    retry_count=config.OCR_RETRY_COUNT,
                )

            if ocr_engine is not None:
                from pipeline.ocr_helpers import ocr_pages as _ocr_pages
                _ocr_pages(
                    file_path, doc_id, page_count, cache,
                    config, ocr_engine, force=False,
                    ocr_workers=config.OCR_WORKERS,
                )
            elif use_gpu_mgr:
                from pipeline.ocr_helpers import ocr_pages as _ocr_pages
                _ocr_pages(
                    file_path, doc_id, page_count, cache,
                    config, ocr_engine, force=False,
                    ocr_workers=config.OCR_WORKERS,
                    gpu_manager=gpu_manager_client,
                )

        try:
            fp = extract_contacts_from_sqlite(doc_id, cache)
            cache.store_contact_fingerprint(doc_id, fp.to_json())
        except Exception:
            pass

        cache.conn.commit()
        elapsed = time.time() - t_worker
        total_paras = sum(len(c.paragraphs) for c in chunks) if chunks else 0
        total_imgs = sum(len(c.image_hashes) for c in chunks) if chunks else 0
        logger.info(f"[Worker] 完成: {filename} "
                    f"({page_count} 页, {len(chunks)} 块, {total_paras} 段, "
                    f"{total_imgs} 图片哈希, {elapsed:.1f}s)")
        return {"doc_id": doc_id, "filename": filename, "success": True}

    except Exception as e:
        logger.error(f"[Worker] 失败 ({file_path}): {e}", exc_info=True)
        return {"doc_id": "", "filename": os.path.basename(file_path),
                "success": False, "error": str(e)}
    finally:
        if cache:
            cache.close()
            for attr in ['cache', 'doc_cache', 'matcher']:
                if hasattr(_tls, attr):
                    delattr(_tls, attr)


def analyze_pair_worker(args: tuple) -> dict:
    """Phase 3 worker: 分析一个候选文档对"""
    doc_a_id, doc_b_id, config_dict, db_dir = args[:4]
    shared_matcher = args[4] if len(args) > 4 else None
    pair_id = "::".join(sorted([doc_a_id, doc_b_id]))

    from config import DetectionConfig
    config = DetectionConfig(**config_dict)
    cache = None
    t_start = time.time()
    try:
        from matching.paragraph_matcher import ParagraphMatcher

        cache = _get_thread_cache(db_dir, config)
        doc_a = _get_thread_doc(doc_a_id, cache)
        doc_b = _get_thread_doc(doc_b_id, cache)

        if not doc_a or not doc_b:
            cache.mark_pair_processed(doc_a_id, doc_b_id)
            cache.conn.commit()
            return {"pair_id": pair_id, "success": False, "match_count": 0, "error": "Doc not found"}

        fname_a, fname_b = doc_a.filename, doc_b.filename

        if _tls.matcher is None:
            _tls.matcher = ParagraphMatcher(config)
        matcher = _tls.matcher
        if shared_matcher is not None:
            matcher.semantic_matcher = shared_matcher

        if not doc_a.doc_minhash or not doc_b.doc_minhash:
            if doc_a.image_hashes or doc_b.image_hashes:
                logger.info(f"[Analyze] 无文本-仅图片: {fname_a} vs {fname_b}")
                image_evidence = _build_image_evidence(doc_a, doc_b, cache, config=config,
                                                        semantic_matcher=getattr(matcher, 'semantic_matcher', None))
                metadata_evidence = build_metadata_evidence(doc_a, doc_b)
                evidence = EvidenceChain(
                    text_evidence=TextEvidence(),
                    metadata_evidence=metadata_evidence,
                    image_evidence=image_evidence,
                )
                result = PairwiseResult(
                    pair_id=pair_id, doc_a_id=doc_a_id, doc_b_id=doc_b_id,
                    similarity_scores={
                        'text_local': 0.0,
                        'metadata_match': len(metadata_evidence.matched_fields),
                        'image_common': image_evidence.common_image_count,
                    },
                    evidence=evidence,
                )
                cache.store_pairwise_result(result)
                cache.mark_pair_processed(doc_a_id, doc_b_id)
                cache.conn.commit()
                return {"pair_id": pair_id, "success": True, "match_count": 0,
                        "clone_block_count": 0, "text_similarity": 0.0,
                        "filename_a": fname_a, "filename_b": fname_b,
                        "error": ""}
            else:
                cache.mark_pair_processed(doc_a_id, doc_b_id)
                cache.conn.commit()
                return {"pair_id": pair_id, "success": True, "match_count": 0,
                        "clone_block_count": 0, "text_similarity": 0.0,
                        "filename_a": fname_a, "filename_b": fname_b,
                        "error": ""}

        t_match = time.time()
        paragraph_matches = matcher.match(doc_a, doc_b, cache)
        match_time = time.time() - t_match

        text_evidence = _build_text_evidence_basic(doc_a, doc_b, paragraph_matches, config)
        metadata_evidence = build_metadata_evidence(doc_a, doc_b)
        image_evidence = _build_image_evidence(doc_a, doc_b, cache, config=config,
                                                semantic_matcher=getattr(matcher, 'semantic_matcher', None))

        evidence = EvidenceChain(
            text_evidence=text_evidence,
            metadata_evidence=metadata_evidence,
            image_evidence=image_evidence,
        )

        result = PairwiseResult(
            pair_id=pair_id, doc_a_id=doc_a_id, doc_b_id=doc_b_id,
            similarity_scores={
                'text_local': text_evidence.local_similarity,
                'metadata_match': len(metadata_evidence.matched_fields),
                'image_common': image_evidence.common_image_count,
            },
            evidence=evidence,
        )
        cache.store_pairwise_result(result)
        cache.mark_pair_processed(doc_a_id, doc_b_id)
        cache.conn.commit()

        clone_block_count = len(text_evidence.continuous_clone_blocks) if hasattr(text_evidence, 'continuous_clone_blocks') else 0

        total_time = time.time() - t_start
        logger.info(f"[Analyze] {fname_a} vs {fname_b} — "
                    f"匹配 {len(paragraph_matches)} 段, "
                    f"相似度 {text_evidence.local_similarity:.3f}, "
                    f"克隆块 {clone_block_count}, "
                    f"匹配耗时 {match_time:.2f}s, 总计 {total_time:.2f}s")
        return {"pair_id": pair_id, "success": True,
                "match_count": len(paragraph_matches),
                "clone_block_count": clone_block_count,
                "text_similarity": text_evidence.local_similarity,
                "filename_a": fname_a, "filename_b": fname_b,
                "error": ""}

    except Exception as e:
        logger.error(f"[Analyze Worker] 失败 ({pair_id}): {e}", exc_info=True)
        try:
            if cache:
                cache.mark_pair_processed(doc_a_id, doc_b_id)
                cache.conn.commit()
        except Exception:
            pass
        return {"pair_id": pair_id, "success": False,
                "match_count": 0, "error": str(e)}
    finally:
        if cache:
            cache.close()
            for attr in ['cache', 'doc_cache', 'matcher']:
                if hasattr(_tls, attr):
                    delattr(_tls, attr)


def _build_text_evidence_basic(doc_a, doc_b, paragraph_matches, config):
    from pipeline.evidence_builder import build_text_evidence
    return build_text_evidence(doc_a, doc_b, paragraph_matches, config, compute_highlight=False)
