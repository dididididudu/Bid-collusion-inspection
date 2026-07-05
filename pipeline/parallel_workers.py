"""
并行 Worker 函数 — 供 ProcessPoolExecutor / ThreadPoolExecutor 调用

所有函数必须是模块级（可 pickle），参数仅使用可 pickle 类型。
每个 worker 创建自己的 SQLite 连接，WAL 模式 + busy_timeout 处理写冲突。
"""

import os
import logging
from typing import List, Dict

from config import DetectionConfig
from data_structures import (
    BidFeature, PairwiseResult, EvidenceChain,
    TextEvidence, MetadataEvidence, ImageEvidence,
    ChunkResult, QuoteSignature,
)
from image_analysis.image_ocr import ImageOCREngine, OCRResult
from image_analysis.image_matcher import ImageMatcher
from pipeline.evidence_builder import (
    build_metadata_evidence, build_image_evidence, build_text_evidence,
)
from pipeline.ocr_helpers import ocr_pages, aggregate_ocr_paragraphs

logger = logging.getLogger(__name__)


# ================================================================
# 共享辅助函数：OCR 页面文字提取
# ================================================================

# _ocr_pages → 委托给 ocr_helpers.ocr_pages()
_ocr_pages = ocr_pages


# ================================================================
# 共享辅助函数：OCR 文字 → 段落聚合
# ================================================================

# _aggregate_ocr_paragraphs → 委托给 ocr_helpers.aggregate_ocr_paragraphs()
_aggregate_ocr_paragraphs = aggregate_ocr_paragraphs


# ================================================================
# 共享辅助函数：证据构建
# ================================================================

# _build_metadata_evidence → 委托给 evidence_builder.build_metadata_evidence()
_build_metadata_evidence = build_metadata_evidence


def _build_image_evidence(
    doc_a: BidFeature, doc_b: BidFeature, cache
) -> ImageEvidence:
    """构建增强图片证据 — 四层检测"""
    evidence = ImageEvidence()

    hashes_a = doc_a.image_hashes
    hashes_b = doc_b.image_hashes
    common_exact = list(set(hashes_a) & set(hashes_b))
    evidence.common_image_count = len(common_exact)
    evidence.common_image_hashes = common_exact

    ocr_a = cache.load_image_ocr_results(doc_a.doc_id)
    ocr_b = cache.load_image_ocr_results(doc_b.doc_id)

    if ocr_a:
        evidence.ocr_results_a = ocr_a
    if ocr_b:
        evidence.ocr_results_b = ocr_b

    ocr_objects_a = [
        OCRResult(
            text=r['ocr_text'],
            words=r['ocr_words'],
            bboxes=r['bboxes'],
            confidence=r['confidence'],
            image_hash=r.get('image_hash', ''),
            non_text_hash=r.get('non_text_hash', ''),
            image_width=r.get('image_width', 0),
            image_height=r.get('image_height', 0),
            thumbnail=r.get('thumbnail', b''),
        ) for r in ocr_a
    ]
    ocr_objects_b = [
        OCRResult(
            text=r['ocr_text'],
            words=r['ocr_words'],
            bboxes=r['bboxes'],
            confidence=r['confidence'],
            image_hash=r.get('image_hash', ''),
            non_text_hash=r.get('non_text_hash', ''),
            image_width=r.get('image_width', 0),
            image_height=r.get('image_height', 0),
            thumbnail=r.get('thumbnail', b''),
        ) for r in ocr_b
    ]

    matcher = ImageMatcher()
    match_result = matcher.analyze(
        hashes_a=hashes_a,
        hashes_b=hashes_b,
        ocr_results_a=ocr_objects_a if ocr_objects_a else None,
        ocr_results_b=ocr_objects_b if ocr_objects_b else None,
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

    return evidence


# ================================================================
# Phase 1 Worker: 单文档提取 (ProcessPoolExecutor)
# ================================================================

def extract_single_worker(args: tuple) -> dict:
    """Phase 1 worker: 提取单个 PDF，结果存入 SQLite

    Args:
        args: (file_path, config_dict, db_dir)

    Returns:
        {"doc_id": str, "filename": str, "success": bool, "error": str}
    """
    file_path, config_dict, db_dir = args

    config = DetectionConfig(**config_dict)
    cache = None
    doc_id = ""

    try:
        from extraction.feature_cache import DocumentCache
        from extraction.pdf_extractor import PyMuPDFExtractor
        from extraction.text_processor import ChunkedTextProcessor

        cache = DocumentCache(db_dir, config)

        extractor = PyMuPDFExtractor(config)
        text_processor = ChunkedTextProcessor(config)

        ocr_engine = None
        if config.ENABLE_OCR:
            ocr_engine = ImageOCREngine(
                use_gpu=config.USE_GPU,
                engine=config.OCR_ENGINE,
                model_dir=config.OCR_MODEL_DIR,
                offline=config.OCR_OFFLINE_MODE,
                retry_count=config.OCR_RETRY_COUNT,
            )

        # --- 元数据 ---
        metadata, page_count, is_scanned = extractor.extract_metadata(file_path)
        doc_id = extractor._generate_doc_id(file_path)
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)

        # 检查断点续传
        existing_chunks = cache.load_document_chunks(doc_id)
        processed_chunks = {c.chunk_index for c in existing_chunks}

        if is_scanned:
            # === 扫描版 PDF ===
            logger.info(f"[Worker] 扫描版: {filename} ({page_count} 页)")

            all_page_hashes = extractor.extract_all_page_hashes(
                file_path, sample_step=2
            )

            chunks = []
            for chunk_result in extractor.extract_chunks(
                file_path, config.CHUNK_PAGE_SIZE, 0
            ):
                chunk_result.image_hashes = all_page_hashes
                cache.store_chunk(chunk_result)
                chunks.append(chunk_result)

            if chunks:
                feature = text_processor.aggregate_chunks(
                    doc_id=doc_id, filename=filename, file_size=file_size,
                    chunks=chunks, metadata=metadata,
                    is_scanned=True, page_count=page_count,
                )
                feature.image_hashes = list(set(
                    feature.image_hashes + all_page_hashes
                ))
                cache.store_document(feature)

                if ocr_engine is not None:
                    _ocr_pages(
                        file_path, doc_id, page_count, cache,
                        config, ocr_engine, force=True,
                        ocr_workers=config.OCR_WORKERS,
                    )
                    _aggregate_ocr_paragraphs(
                        doc_id, page_count, cache, extractor, text_processor,
                    )
            else:
                # 纯扫描版，无文本
                feature = BidFeature(
                    doc_id=doc_id, filename=filename,
                    file_size=file_size, text_content="",
                    text_length=0, text_simhash="",
                    paragraphs=[], paragraph_hashes=[],
                    metadata=metadata, quotes=[],
                    quote_signature=QuoteSignature(),
                    image_hashes=all_page_hashes,
                    is_scanned=True, page_count=page_count,
                    doc_minhash=None, chunk_count=0,
                )
                cache.store_document(feature)

                if ocr_engine is not None:
                    _ocr_pages(
                        file_path, doc_id, page_count, cache,
                        config, ocr_engine, force=True,
                        ocr_workers=config.OCR_WORKERS,
                    )
                    _aggregate_ocr_paragraphs(
                        doc_id, page_count, cache, extractor, text_processor,
                    )

        else:
            # === 文本版 PDF ===
            start_page = max(
                (max(processed_chunks) + 1) * config.CHUNK_PAGE_SIZE
                if processed_chunks else 0, 0
            )

            if start_page >= page_count and page_count > 0:
                logger.info(f"[Worker] 已完全提取: {filename}")
                cache.close()
                return {"doc_id": doc_id, "filename": filename, "success": True}

            chunks = []
            chunk_size = config.CHUNK_PAGE_SIZE

            for chunk_result in extractor.extract_chunks(
                file_path, chunk_size, start_page
            ):
                cache.store_chunk(chunk_result)
                chunks.append(chunk_result)

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

                if ocr_engine is not None:
                    _ocr_pages(
                        file_path, doc_id, page_count, cache,
                        config, ocr_engine, force=False,
                        ocr_workers=config.OCR_WORKERS,
                    )

        cache.conn.commit()
        logger.info(f"[Worker] 完成: {filename}")
        return {"doc_id": doc_id, "filename": filename, "success": True}

    except Exception as e:
        logger.error(f"[Worker] 失败 ({file_path}): {e}", exc_info=True)
        return {
            "doc_id": doc_id,
            "filename": os.path.basename(file_path),
            "success": False,
            "error": str(e),
        }
    finally:
        if cache:
            cache.close()


# ================================================================
# Phase 1.5 Worker: 单文档嵌入编码 (ProcessPoolExecutor)
# ================================================================

def embed_single_worker(args: tuple) -> dict:
    """Phase 1.5 worker: 编码一个文档的所有段落嵌入

    Args:
        args: (doc_id, config_dict, db_dir)

    Returns:
        {"doc_id": str, "success": bool, "paragraphs_encoded": int, "error": str}
    """
    doc_id, config_dict, db_dir = args

    config = DetectionConfig(**config_dict)
    cache = None
    try:
        from extraction.feature_cache import DocumentCache
        from embedding.embedding_engine import EmbeddingEngine

        cache = DocumentCache(db_dir, config)

        engine = EmbeddingEngine(config)
        if not engine.is_available:
            return {"doc_id": doc_id, "success": False,
                    "paragraphs_encoded": 0, "error": "SBERT unavailable"}

        doc_feat = cache.load_document(doc_id)
        if not doc_feat or not doc_feat.doc_minhash:
            return {"doc_id": doc_id, "success": False,
                    "paragraphs_encoded": 0, "error": "No MinHash"}

        paragraphs = cache.load_all_paragraphs_text(doc_id)
        if paragraphs:
            count = engine.encode_document(doc_id, paragraphs, cache)
            cache.conn.commit()
            return {"doc_id": doc_id, "success": True,
                    "paragraphs_encoded": count, "error": ""}
        else:
            return {"doc_id": doc_id, "success": False,
                    "paragraphs_encoded": 0, "error": "No paragraphs"}

    except Exception as e:
        logger.error(f"[Embed Worker] 失败 ({doc_id}): {e}", exc_info=True)
        return {"doc_id": doc_id, "success": False,
                "paragraphs_encoded": 0, "error": str(e)}
    finally:
        if cache:
            cache.close()


# ================================================================
# Phase 3 Worker: 单对分析 (ThreadPoolExecutor)
# ================================================================

def analyze_pair_worker(args: tuple) -> dict:
    """Phase 3 worker: 分析一个候选文档对

    Args:
        args: (doc_a_id, doc_b_id, config_dict, db_dir, semantic_matcher)

    Returns:
        {"pair_id": str, "success": bool, "match_count": int, "error": str}
    """
    doc_a_id, doc_b_id, config_dict, db_dir = args[:4]
    shared_matcher = args[4] if len(args) > 4 else None
    pair_id = "::".join(sorted([doc_a_id, doc_b_id]))

    config = DetectionConfig(**config_dict)
    cache = None
    try:
        from extraction.feature_cache import DocumentCache
        from matching.paragraph_matcher import ParagraphMatcher
        from scoring import RiskScoringEngine

        cache = DocumentCache(db_dir, config)

        doc_a = cache.load_document(doc_a_id)
        doc_b = cache.load_document(doc_b_id)
        if not doc_a or not doc_b:
            cache.mark_pair_processed(doc_a_id, doc_b_id)
            cache.conn.commit()
            return {"pair_id": pair_id, "success": False,
                    "match_count": 0, "error": "Doc not found"}

        matcher = ParagraphMatcher(config)
        if shared_matcher is not None:
            # 复用主线程已加载的 SBERT 模型，避免每个 worker 重复加载
            matcher.semantic_matcher = shared_matcher
        scorer = RiskScoringEngine(config)

        # 检查文字可用性
        if not doc_a.doc_minhash or not doc_b.doc_minhash:
            if doc_a.is_scanned or doc_b.is_scanned:
                # 纯图片比对路径
                image_evidence = _build_image_evidence(doc_a, doc_b, cache)
                metadata_evidence = _build_metadata_evidence(doc_a, doc_b)

                evidence = EvidenceChain(
                    text_evidence=TextEvidence(),
                    metadata_evidence=metadata_evidence,
                    image_evidence=image_evidence,
                )

                result = PairwiseResult(
                    pair_id=pair_id,
                    doc_a_id=doc_a_id, doc_b_id=doc_b_id,
                    similarity_scores={
                        'text_local': 0.0,
                        'metadata_match': len(metadata_evidence.matched_fields),
                        'image_common': image_evidence.common_image_count,
                    },
                    evidence=evidence,
                )
                result = scorer._score_pair(result)
                cache.store_pairwise_result(result)
                cache.mark_pair_processed(doc_a_id, doc_b_id)
                cache.conn.commit()
                return {"pair_id": pair_id, "success": True,
                        "match_count": 0, "error": ""}
            else:
                cache.mark_pair_processed(doc_a_id, doc_b_id)
                cache.conn.commit()
                return {"pair_id": pair_id, "success": True,
                        "match_count": 0, "error": ""}

        # 正常文本分析路径
        paragraph_matches = matcher.match(doc_a, doc_b, cache)

        # 构建证据
        text_evidence = _build_text_evidence_basic(
            doc_a, doc_b, paragraph_matches, config
        )
        metadata_evidence = _build_metadata_evidence(doc_a, doc_b)
        image_evidence = _build_image_evidence(doc_a, doc_b, cache)

        evidence = EvidenceChain(
            text_evidence=text_evidence,
            metadata_evidence=metadata_evidence,
            image_evidence=image_evidence,
        )

        result = PairwiseResult(
            pair_id=pair_id,
            doc_a_id=doc_a_id, doc_b_id=doc_b_id,
            similarity_scores={
                'text_local': text_evidence.local_similarity,
                'metadata_match': len(metadata_evidence.matched_fields),
                'image_common': image_evidence.common_image_count,
            },
            evidence=evidence,
        )
        result = scorer._score_pair(result)
        cache.store_pairwise_result(result)
        cache.mark_pair_processed(doc_a_id, doc_b_id)
        cache.conn.commit()

        return {"pair_id": pair_id, "success": True,
                "match_count": len(paragraph_matches), "error": ""}

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


def _build_text_evidence_basic(
    doc_a: BidFeature, doc_b: BidFeature,
    paragraph_matches: List[Dict], config: DetectionConfig,
) -> TextEvidence:
    """构建文本证据（基本版，不含差异高亮 — 委托 evidence_builder 复用）

    差异高亮依赖 difflib.SequenceMatcher 且计算量大，
    在 worker 中跳过，由 report 阶段按需计算。
    """
    return build_text_evidence(
        doc_a, doc_b, paragraph_matches, config, compute_highlight=False
    )


def _detect_clone_blocks(
    paragraph_matches: List[Dict], config: DetectionConfig
) -> List[Dict]:
    """检测连续克隆块（委托 evidence_builder 复用）"""
    from pipeline.evidence_builder import _detect_clone_blocks as _dcb
    return _dcb(paragraph_matches, config)
