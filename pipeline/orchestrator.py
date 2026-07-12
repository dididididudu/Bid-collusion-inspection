"""
6 阶段流式管道编排器 (v2)

管理完整的检测流程:
  Phase 0: SCAN    - 扫描目录，收集 PDF 元数据
  Phase 1: EXTRACT - 多进程分块文本提取（PyMuPDF）
  Phase 1.5: EMBED - 全局 SBERT 嵌入编码（一次性）
  Phase 2: SELECT  - 候选对筛选（LSH + 元数据 + 文档向量预筛）
  Phase 3: ANALYZE - 逐对精细分析（查表点积，不调模型）
  Phase 4: CLUSTER - 相似内容聚类与报告编排
  Phase 5: REPORT  - 生成报告
"""

import os
import glob
import logging
import time
from datetime import datetime
from typing import List, Dict
from concurrent.futures import (
    ProcessPoolExecutor, ThreadPoolExecutor, as_completed
)

from config import DetectionConfig
from data_structures import (
    BidFeature, GlobalReport, PairwiseResult, EvidenceChain,
    TextEvidence, MetadataEvidence, ImageEvidence, ContactEvidence
)
from extraction.feature_cache import DocumentCache
from extraction.pdf_extractor import PyMuPDFExtractor
from extraction.text_processor import ChunkedTextProcessor
from matching.selector import CandidatePairSelector
from matching.paragraph_matcher import ParagraphMatcher
from pipeline.checkpoint import CheckpointManager
from pipeline.streaming_context import StreamingContext
from embedding.embedding_engine import EmbeddingEngine
from image_analysis.image_ocr import ImageOCREngine, OCRResult
from image_analysis.image_matcher import ImageMatcher
from scoring import RiskScoringEngine
from report import ReportGenerator
from pipeline.evidence_builder import (
    build_metadata_evidence, build_text_evidence,
    build_contact_evidence, _get_image_dimension_tag,
)
from pipeline.ocr_helpers import ocr_pages, aggregate_ocr_paragraphs
from extraction.contact_extractor import extract_contacts_from_sqlite

logger = logging.getLogger(__name__)


class BidDetectionOrchestrator:
    """5 阶段流式管道编排器"""

    OCR_CHUNK_INDEX: int = 1000000

    def __init__(self, config: DetectionConfig, progress_callback=None):
        self.config = config
        self._progress_callback = progress_callback

        self.cache = DocumentCache(config.CACHE_DIR, config)
        self.checkpoint = CheckpointManager(config.CHECKPOINT_DIR, config)
        self.streaming = StreamingContext(self.cache, config.MAX_CHUNKS_IN_MEMORY)

        self.extractor = PyMuPDFExtractor(config)
        self.text_processor = ChunkedTextProcessor(config)
        self.selector = CandidatePairSelector(config)
        self.paragraph_matcher = ParagraphMatcher(config)
        self.scoring_engine = RiskScoringEngine(config)
        self.report_generator = ReportGenerator(config)

        self.embedding_engine = EmbeddingEngine(config)

        self.ocr_engine = ImageOCREngine(
            use_gpu=config.USE_GPU,
            engine=config.OCR_ENGINE,
            model_dir=config.OCR_MODEL_DIR,
            offline=config.OCR_OFFLINE_MODE,
            retry_count=config.OCR_RETRY_COUNT,
        )
        self.image_matcher = ImageMatcher()

        from pipeline.gpu_manager import GPUManager
        self.gpu_manager = GPUManager(config)

        self._all_features_cache = None
        self._phase3_para_cache = {}
        self._phase3_embedding_cache = {}
        self._timings = {}

        logger.info("流式管道编排器已初始化")

    def _get_picklable_config(self) -> dict:
        return {
            k: v for k, v in self.config.__dict__.items()
            if not k.startswith('_')
        }

    def detect(self, input_dir: str, output_dir: str) -> GlobalReport:
        process_start = datetime.now()

        logger.info("=" * 60)
        logger.info("流式管道检测开始")
        logger.info("=" * 60)

        # 打印关键配置摘要
        cfg = self.config
        dims_on = [k for k, v in cfg.ENABLED_DIMENSIONS.items() if v]
        logger.info(f"配置: workers(P1={cfg.PHASE1_WORKERS}, P3={cfg.PHASE3_WORKERS}, "
                    f"PDF_CHUNK={getattr(cfg, 'PDF_CHUNK_WORKERS', 1)}, "
                    f"OCR_COLLECT={getattr(cfg, 'OCR_COLLECT_WORKERS', 1)}, "
                    f"OCR={cfg.OCR_WORKERS}), "
                    f"GPU={cfg.SBERT_DEVICE}, OCR={cfg.OCR_ENGINE}, "
                    f"阈值(LSH={cfg.MINHASH_LSH_THRESHOLD}, Jaccard={cfg.MINHASH_JACCARD_THRESHOLD}), "
                    f"启用的维度: {dims_on}")

        if self.config.ENABLE_OCR:
            ok, msg = self.ocr_engine.health_check()
            if ok:
                logger.info(f"OCR 引擎健康检查通过: {msg}")
            else:
                logger.warning(f"OCR 引擎健康检查未通过: {msg}")

        if self.config.DISABLE_CACHE:
            logger.info("缓存已禁用，清除旧缓存文件...")
            self.checkpoint.clear()
            self.cache.clear_cache()
            for suffix in ['', '-wal', '-shm']:
                f = self.cache.db_path + suffix
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except OSError:
                        pass

        input_hash = self._compute_input_hash(input_dir)
        state = self.checkpoint.load_or_new()

        if not self.config.DISABLE_CACHE:
            if state.input_hash and state.input_hash != input_hash:
                logger.warning(
                    f"检测到输入文件夹内容变化！旧哈希: {state.input_hash}, "
                    f"新哈希: {input_hash}。将重新开始处理。"
                )
                self.checkpoint.clear()
                self.cache.clear_cache()
                state = self.checkpoint.load_or_new()

        state.input_hash = input_hash

        try:
            file_paths = self._scan_pdf_files(input_dir)
            if not file_paths:
                logger.error("未找到 PDF 文件")
                return self._empty_report()

            total_size_mb = sum(os.path.getsize(fp) for fp in file_paths) / (1024 * 1024)
            logger.info(f"扫描到 {len(file_paths)} 个 PDF 文件, "
                        f"总计 {total_size_mb:.1f} MB, "
                        f"最大 {max(os.path.getsize(fp) for fp in file_paths) / (1024 * 1024):.1f} MB")

            if state.phase < 1:
                logger.info("Phase 0: 扫描文档...")
                self._phase0_metadata(file_paths)
                state.phase = 1
                self.checkpoint.save(state)
                logger.info(f"Phase 0 完成: {len(file_paths)} 个 PDF 文件")

            self.gpu_manager.start()

            if state.phase < 2:
                logger.info("Phase 1: 提取特征...")
                unprocessed = [
                    fp for fp in file_paths
                    if fp not in state.processed_files
                ]

                if not unprocessed:
                    logger.info("所有文件已处理，跳过 Phase 1")
                else:
                    extract_start = datetime.now()
                    processed_count = 0

                    num_workers = min(
                        max(1, self.config.PHASE1_WORKERS),
                        len(unprocessed),
                        os.cpu_count() or 4,
                    )
                    if len(unprocessed) <= 20:
                        num_workers = min(num_workers, 4)
                    else:
                        num_workers = min(num_workers, 8)

                    if num_workers <= 1:
                        for file_path in unprocessed:
                            t0_file = datetime.now()
                            try:
                                self._phase1_extract_single(file_path)
                                state.processed_files.add(file_path)
                                processed_count += 1
                                elapsed = (datetime.now() - t0_file).total_seconds()
                                logger.info(f"Phase 1 [{processed_count}/{len(unprocessed)}] "
                                            f"完成: {os.path.basename(file_path)} ({elapsed:.1f}s)")
                                if processed_count % 10 == 0:
                                    logger.info(f"Phase 1 进度: {processed_count}/{len(unprocessed)}")
                                    self.checkpoint.save(state)
                            except Exception as e:
                                logger.error(f"提取失败 ({file_path}): {e}", exc_info=True)
                                continue
                    else:
                        logger.info(f"Phase 1 并行模式: {num_workers} workers, {len(unprocessed)} 个文件")
                        from pipeline.parallel_workers import extract_single_worker
                        config_dict = self._get_picklable_config()
                        db_dir = self.config.CACHE_DIR
                        gpu_client = self.gpu_manager.client

                        with ProcessPoolExecutor(max_workers=num_workers) as executor:
                            future_to_file = {
                                executor.submit(
                                    extract_single_worker,
                                    (fp, config_dict, db_dir, gpu_client)
                                ): fp
                                for fp in unprocessed
                            }
                            for future in as_completed(future_to_file):
                                file_path = future_to_file[future]
                                try:
                                    result = future.result()
                                    if result['success']:
                                        state.processed_files.add(file_path)
                                        processed_count += 1
                                    else:
                                        logger.error(f"提取失败 ({file_path}): {result.get('error', 'unknown')}")
                                except Exception as e:
                                    logger.error(f"Worker 异常 ({file_path}): {e}", exc_info=True)
                                if processed_count % self.config.CHECKPOINT_INTERVAL == 0:
                                    self.checkpoint.save(state)

                state.phase = 2
                self.checkpoint.save(state)
                extract_time = (datetime.now() - extract_start).total_seconds()
                logger.info(f"Phase 1 完成: {processed_count}/{len(unprocessed)} 个文档, 耗时 {extract_time:.2f}s")

            # ── Phase 1.2: TOC 解析（技术标/商务标分界）──
            if state.phase < 3 and self.config.ENABLED_DIMENSIONS.get('content_similarity', True):
                self._phase_toc_parse(state)

            # ── Phase 1.5: SBERT 嵌入 ──
            if state.phase < 3 and self.config.ENABLED_DIMENSIONS.get('content_similarity', True):
                self._phase_embed(state)

            if (self.embedding_engine.is_available
                    and self.embedding_engine.model is not None
                    and self.paragraph_matcher is not None):
                self.paragraph_matcher._ensure_semantic_matcher()
                self.paragraph_matcher.semantic_matcher.set_model(
                    self.embedding_engine.model
                )
                logger.debug("Phase 1.5 SBERT 模型已注入 Phase 3 ParagraphMatcher")

                # 也注入 ImageMatcher，使 OCR 文字比对使用 SBERT（而非 Jaccard）
                if self.image_matcher is not None:
                    self.image_matcher.semantic_matcher = (
                        self.paragraph_matcher.semantic_matcher
                    )
                    logger.debug("SBERT 模型已注入 ImageMatcher（OCR 文字比对）")

            if state.phase < 4:
                logger.info("Phase 2: 候选对选择...")
                select_start = datetime.now()
                features = self.cache.load_all_documents()
                self._all_features_cache = features
                candidates = self.selector.select(features, cache=self.cache)
                self.cache.store_candidate_pairs(candidates)
                state.total_pairs = len(candidates)
                # 候选对来源分布
                method_counts = {}
                for _, _, method, _ in candidates:
                    method_counts[method] = method_counts.get(method, 0) + 1
                method_summary = ", ".join(f"{m}={c}" for m, c in sorted(method_counts.items()))
                state.phase = 3
                self.checkpoint.save(state)
                select_time = (datetime.now() - select_start).total_seconds()
                logger.info(f"Phase 2 完成: {len(candidates)} 对候选 "
                            f"({method_summary}), 耗时 {select_time:.2f}s")

            if state.phase < 5:
                logger.info("Phase 3: 精细分析...")
                completed_ids = self.checkpoint.load_phase3_progress()
                state.completed_pair_ids = completed_ids
                all_pairs = self.cache.get_unprocessed_pairs()
                pending_pairs = [
                    p for p in all_pairs
                    if "::".join(sorted(p)) not in completed_ids
                ]

                if not pending_pairs:
                    logger.info("所有候选对已分析，跳过 Phase 3")
                else:
                    analyze_start = datetime.now()
                    total_pending = len(pending_pairs)
                    num_workers = min(
                        max(1, self.config.PHASE3_WORKERS),
                        total_pending,
                        (os.cpu_count() or 4) * 2,
                    )
                    if total_pending <= 20:
                        num_workers = min(num_workers, 4)
                    else:
                        num_workers = min(num_workers, 8)

                    if num_workers <= 1:
                        if getattr(self.config, 'PHASE3_PRELOAD_EMBEDDINGS', True):
                            all_doc_ids = set()
                            for doc_a_id, doc_b_id in pending_pairs:
                                all_doc_ids.add(doc_a_id)
                                all_doc_ids.add(doc_b_id)
                            for did in all_doc_ids:
                                try:
                                    self._phase3_embedding_cache[did] = self.cache.load_all_paragraph_embeddings(did)
                                except Exception as e:
                                    logger.warning(f"预加载嵌入失败 ({did[:12]}...): {e}")
                        for idx, (doc_a_id, doc_b_id) in enumerate(pending_pairs, 1):
                            try:
                                self._phase3_analyze_single(doc_a_id, doc_b_id)
                                pair_id = "::".join(sorted([doc_a_id, doc_b_id]))
                                completed_ids.add(pair_id)
                                state.completed_pairs = len(completed_ids)
                                if self._progress_callback:
                                    self._progress_callback({'update_progress': True, 'current': idx, 'total': total_pending})
                                if idx % 50 == 0 or idx == total_pending:
                                    logger.info(f"Phase 3 进度: {idx}/{total_pending} ({state.completed_pairs}/{state.total_pairs})")
                                if idx % self.config.CHECKPOINT_INTERVAL == 0:
                                    self.checkpoint.save_phase3_progress(completed_ids)
                                    self.checkpoint.save(state)
                                    self.cache.conn.commit()
                            except Exception as e:
                                logger.error(f"分析失败 ({doc_a_id} vs {doc_b_id}): {e}", exc_info=True)
                                self.cache.mark_pair_processed(doc_a_id, doc_b_id)
                                continue
                        self.cache.conn.commit()
                    else:
                        logger.info(f"Phase 3 并行模式: {num_workers} workers, {total_pending} 对候选")
                        from pipeline.parallel_workers import analyze_pair_worker
                        config_dict = self._get_picklable_config()
                        db_dir = self.config.CACHE_DIR
                        self.paragraph_matcher._ensure_semantic_matcher()
                        shared_matcher = self.paragraph_matcher.semantic_matcher

                        all_doc_ids = set()
                        for doc_a_id, doc_b_id in pending_pairs:
                            all_doc_ids.add(doc_a_id)
                            all_doc_ids.add(doc_b_id)
                        preload_start = datetime.now()
                        all_para_full = {}
                        all_para_embeddings = {}
                        for did in all_doc_ids:
                            try:
                                all_para_full[did] = self.cache.load_all_paragraphs_full(did)
                                if getattr(self.config, 'PHASE3_PRELOAD_EMBEDDINGS', True):
                                    all_para_embeddings[did] = self.cache.load_all_paragraph_embeddings(did)
                            except Exception as e:
                                logger.warning(f"预加载段落失败 ({did[:12]}...): {e}")
                        preload_time = (datetime.now() - preload_start).total_seconds()
                        logger.info(
                            f"Phase 3: 预加载 {len(all_para_full)}/{len(all_doc_ids)} 个文档段落, "
                            f"{len(all_para_embeddings)} 个文档向量, 耗时 {preload_time:.2f}s"
                        )

                        use_process_pool = getattr(self.config, 'PHASE3_USE_PROCESS_POOL', True)
                        executor_cls = ProcessPoolExecutor if use_process_pool else ThreadPoolExecutor
                        shared_matcher_arg = None if use_process_pool else shared_matcher
                        para_full_arg = None if use_process_pool else all_para_full
                        para_embeddings_arg = None if use_process_pool else all_para_embeddings
                        pool_name = "多进程" if use_process_pool else "多线程"
                        logger.info(f"Phase 3 执行器: {pool_name}, workers={num_workers}")

                        with executor_cls(max_workers=num_workers) as executor:
                            future_to_pair = {}
                            for doc_a_id, doc_b_id in pending_pairs:
                                future = executor.submit(
                                    analyze_pair_worker,
                                    (doc_a_id, doc_b_id, config_dict, db_dir, shared_matcher_arg, para_full_arg, para_embeddings_arg)
                                )
                                future_to_pair[future] = (doc_a_id, doc_b_id)

                            done_count = 0
                            for future in as_completed(future_to_pair):
                                doc_a_id, doc_b_id = future_to_pair[future]
                                pair_id = "::".join(sorted([doc_a_id, doc_b_id]))
                                done_count += 1
                                try:
                                    result = future.result()
                                    if result.get('success', False):
                                        completed_ids.add(pair_id)
                                        state.completed_pairs = len(completed_ids)
                                        if self._progress_callback:
                                            self._progress_callback({
                                                'pair_id': pair_id, 'doc_a_id': doc_a_id, 'doc_b_id': doc_b_id,
                                                'filename_a': result.get('filename_a', doc_a_id),
                                                'filename_b': result.get('filename_b', doc_b_id),
                                                'text_similarity': result.get('text_similarity', 0),
                                                'match_count': result.get('match_count', 0),
                                                'clone_count': result.get('clone_block_count', 0),
                                            })
                                except Exception as e:
                                    logger.error(f"分析 worker 异常 ({doc_a_id} vs {doc_b_id}): {e}", exc_info=True)
                                if done_count % 50 == 0 or done_count == total_pending:
                                    logger.info(f"Phase 3 进度: {done_count}/{total_pending} ({len(completed_ids)}/{state.total_pairs})")
                                if done_count % self.config.CHECKPOINT_INTERVAL == 0:
                                    self.checkpoint.save_phase3_progress(completed_ids)
                                    self.checkpoint.save(state)

                    analyze_time = (datetime.now() - analyze_start).total_seconds()
                    self._timings['phase3_analyze'] = analyze_time
                    logger.info(f"Phase 3 完成: {len(completed_ids)} 对, 耗时 {analyze_time:.2f}s")

                state.phase = 5
                state.completed_pairs = len(completed_ids)
                self.checkpoint.save(state)
                self._phase3_para_cache.clear()
                self._phase3_embedding_cache.clear()

            report = None
            if state.phase < 6:
                logger.info("Phase 4: 相似内容聚类...")
                score_start = datetime.now()
                pairwise_results = self.cache.load_all_results()
                features = self._all_features_cache or self.cache.load_all_documents()
                report = self.scoring_engine.generate_report(pairwise_results, features)
                state.phase = 6
                self.checkpoint.save(state)
                score_time = (datetime.now() - score_start).total_seconds()
                self._timings['phase4_score'] = score_time
                logger.info(f"Phase 4 完成: {report.suspicious_pairs} 对可疑, 耗时 {score_time:.2f}s")

            if state.phase >= 6 and report is None:
                pairwise_results = self.cache.load_all_results()
                features = self._all_features_cache or self.cache.load_all_documents()
                report = self.scoring_engine.generate_report(pairwise_results, features)

            if report is not None:
                logger.info("Phase 5: 报告生成...")
                report_start = datetime.now()
                self.report_generator.generate(report, output_dir)
                report_time = (datetime.now() - report_start).total_seconds()
                self._timings['phase5_report'] = report_time
                logger.info(f"Phase 5 完成: 报告已输出到 {output_dir}, 耗时 {report_time:.2f}s")
            else:
                report = self._empty_report()
                logger.warning("未生成任何报告数据")

            total_time = (datetime.now() - process_start).total_seconds()
            if getattr(self.config, 'PERF_LOG_ENABLED', True):
                timing_summary = ", ".join(
                    f"{name}={seconds:.2f}s"
                    for name, seconds in self._timings.items()
                )
                if timing_summary:
                    logger.info(f"性能埋点: {timing_summary}")
            logger.info("=" * 60)
            logger.info(f"检测完成! 总耗时: {total_time:.2f}s")
            logger.info("=" * 60)
            return report

        except Exception as e:
            logger.error(f"管道执行失败: {e}", exc_info=True)
            self.checkpoint.save(state)
            raise
        finally:
            self.gpu_manager.shutdown()
            self.streaming.clear()
            self.cache.close()

    @staticmethod
    def _scan_pdf_files(input_dir: str) -> List[str]:
        pattern = os.path.join(input_dir, "*.pdf")
        files = glob.glob(pattern)
        logger.info(f"扫描到 {len(files)} 个 PDF 文件")
        return files

    def _phase0_metadata(self, file_paths: List[str]):
        for file_path in file_paths:
            try:
                metadata, page_count, is_scanned = (
                    self.extractor.extract_metadata(file_path)
                )
                doc_id = self.extractor._generate_doc_id(file_path)
                filename = os.path.basename(file_path)
                file_size = os.path.getsize(file_path)

                # 不覆盖已提取的文档特征（Phase 0 仅存储元数据占位）
                existing = self.cache.load_document(doc_id)
                if existing and existing.doc_minhash:
                    logger.debug(f"Phase 0: 跳过已提取文档 {filename}")
                    continue

                feature = BidFeature(
                    doc_id=doc_id,
                    filename=filename,
                    file_size=file_size,
                    text_content="",
                    text_length=0,
                    text_simhash="",
                    paragraphs=[],
                    paragraph_hashes=[],
                    metadata=metadata,
                    quotes=[],
                    image_hashes=[],
                    extracted_at=datetime.now().isoformat(),
                    is_scanned=False,
                    page_count=page_count,
                    doc_minhash=None,
                    chunk_count=0,
                )
                self.cache.store_document(feature)
                self.cache.store_metadata_fingerprint(
                    doc_id=doc_id,
                    author=metadata.author or '',
                    creator=metadata.creator or '',
                    producer=metadata.producer or '',
                    software_fingerprint=metadata.software_fingerprint or '',
                    time_bucket=metadata.time_bucket or '',
                )
                logger.debug(f"Phase 0: {filename} ({page_count} 页)")
            except Exception as e:
                logger.error(f"Phase 0 失败 ({file_path}): {e}")

    def _phase1_extract_single(self, file_path: str):
        logger.info(f"Phase 1: 提取 {os.path.basename(file_path)}...")

        metadata, page_count, is_scanned = self.extractor.extract_metadata(file_path)
        doc_id = self.extractor._generate_doc_id(file_path)
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)

        existing_chunks = self.cache.load_document_chunks(doc_id)
        processed_chunks = {c.chunk_index for c in existing_chunks}

        # === 文本版 PDF ===
        start_page = max(
            (max(processed_chunks) + 1) * self.config.CHUNK_PAGE_SIZE
            if processed_chunks else 0,
            0
        )

        if start_page >= page_count and page_count > 0:
            # 检查文档特征是否已存储（含 doc_minhash）
            existing_doc = self.cache.load_document(doc_id)
            if existing_doc and existing_doc.doc_minhash:
                logger.info(f"文档已完全提取: {filename}")
                if self.config.ENABLE_OCR:
                    self._phase1_ocr_pages(file_path, doc_id, page_count)
                try:
                    fp = extract_contacts_from_sqlite(doc_id, self.cache)
                    self.cache.store_contact_fingerprint(doc_id, fp.to_json())
                except Exception:
                    pass
                return
            else:
                # chunks 已缓存但文档特征缺失（doc_minhash 为空），需重新提取
                logger.warning(f"文档 {filename} 的 chunks 已缓存但特征缺失，重新提取")
                self.cache.delete_document_chunks(doc_id)
                start_page = 0

        chunks = []
        chunk_size = self.config.CHUNK_PAGE_SIZE

        for chunk_result in self.extractor.extract_chunks(
            file_path, chunk_size, start_page
        ):
            self.cache.store_chunk(chunk_result)
            chunks.append(chunk_result)
            img_count = len(chunk_result.image_hashes)
            logger.debug(
                f"  块 {chunk_result.chunk_index}: "
                f"页 {chunk_result.start_page}-{chunk_result.end_page}, "
                f"{len(chunk_result.paragraphs)} 段, {img_count} 图片哈希"
            )

        if chunks:
            feature = self.text_processor.aggregate_chunks(
                doc_id=doc_id,
                filename=filename,
                file_size=file_size,
                chunks=chunks,
                metadata=metadata,
                is_scanned=False,
                page_count=page_count,
            )
            all_img_hashes = set()
            for c in chunks:
                all_img_hashes.update(c.image_hashes)
            feature.image_hashes = list(all_img_hashes)
            self.cache.store_document(feature)

            ocr_count = self._phase1_ocr_pages(
                file_path, doc_id, page_count
            )

            try:
                fp = extract_contacts_from_sqlite(doc_id, self.cache)
                self.cache.store_contact_fingerprint(doc_id, fp.to_json())
            except Exception:
                pass

            logger.info(
                f"Phase 1 完成: {filename} "
                f"({page_count} 页, {len(chunks)} 块, "
                f"{feature.text_length} 字符, {len(feature.image_hashes)} 图片哈希)"
            )

    def _phase1_ocr_pages(
        self, file_path: str, doc_id: str, page_count: int,
    ) -> int:
        """OCR 提取嵌入图片文字 — 统一委托给 ocr_helpers.ocr_pages

        始终使用 ocr_pages()（含去重、并行、批量入库），
        不再区分 GPU Manager 启用与否走不同代码路径。
        """
        if not self.config.ENABLE_OCR:
            return 0

        return ocr_pages(
            file_path, doc_id, page_count,
            self.cache, self.config, self.ocr_engine,
            ocr_workers=self.config.OCR_WORKERS,
            gpu_manager=self.gpu_manager.client,
        )

    def _phase_toc_parse(self, state) -> None:
        """Phase 1.2: TOC 解析 — 识别技术标/商务标分界

        对每个文档运行 TOCParser，将 page_classifications 存入缓存。
        """
        from pipeline.toc_parser import TOCParser

        features = self.cache.load_all_documents()
        if not features:
            logger.info("Phase 1.2: 无文档可分析")
            return

        logger.info(f"Phase 1.2: TOC 解析 ({len(features)} 个文档)...")
        parse_start = datetime.now()
        parsed_count = 0

        for feat in features:
            if feat.page_count <= 0:
                continue
            try:
                paragraphs_full = self.cache.load_all_paragraphs_full(feat.doc_id)
                if not paragraphs_full:
                    continue

                paras = []
                page_nums = []
                for idx in sorted(paragraphs_full.keys()):
                    info = paragraphs_full[idx]
                    paras.append(info.get('text', ''))
                    page_nums.append(info.get('page_num', -1))

                parser = TOCParser(
                    paragraphs=paras,
                    paragraph_page_nums=page_nums,
                    total_pages=feat.page_count,
                )
                result = parser.parse()

                classifications = result.get('page_classifications') or {}
                meaningful_count = sum(
                    1 for label in classifications.values()
                    if label in ("technical", "commercial")
                )

                if classifications:
                    feat.page_classifications = result['page_classifications']
                    self.cache.store_document(feat)
                    if meaningful_count:
                        parsed_count += 1
                        logger.info(
                            f"  [{feat.filename}] method={result.get('method','?')}, "
                            f"confidence={result.get('confidence', 0):.2f}, "
                            f"tech_start={result.get('tech_start_page', -1)}, "
                            f"commercial_start={result.get('com_start_page', -1)}, "
                            f"classified={meaningful_count}/{len(classifications)} pages"
                        )
                    else:
                        logger.warning(
                            f"  [{feat.filename}] 未识别到技术/商务边界，"
                            f"method={result.get('method','?')}, pages={len(classifications)}"
                        )
                else:
                    logger.debug(f"  [{feat.filename}] 未检测到技术标分界")

            except Exception as e:
                logger.error(f"TOC 解析失败 ({feat.filename}): {e}", exc_info=True)
                continue

        elapsed = (datetime.now() - parse_start).total_seconds()
        logger.info(f"Phase 1.2 完成: {parsed_count}/{len(features)} 个文档已分类, 耗时 {elapsed:.2f}s")

    def _phase_embed(self, state) -> None:
        logger.info("Phase 1.5: 全局 SBERT 嵌入编码...")
        embed_start = datetime.now()

        engine = self.embedding_engine
        if not engine.is_available:
            logger.warning("SBERT 不可用，跳过 Phase 1.5 嵌入编码")
            state.phase = 3
            self.checkpoint.save(state)
            return

        features = self.cache.load_all_documents()
        if not features:
            logger.warning("Phase 1.5: 未找到已提取的文档，跳过嵌入编码")
            state.phase = 3
            self.checkpoint.save(state)
            return

        # ── PyTorch 线程数优化：CPU 模式下 8 线程已达峰值 ──
        try:
            import torch
            optimal_threads = min(8, os.cpu_count() or 4)
            if torch.get_num_threads() != optimal_threads:
                torch.set_num_threads(optimal_threads)
                logger.debug(f"Phase 1.5: PyTorch 线程数设为 {optimal_threads}")
        except Exception:
            pass

        logger.info(f"Phase 1.5: 准备编码 {len(features)} 个文档的段落嵌入")

        dimension = self.config.ANALYSIS_DIMENSION
        if dimension in ("technical", "commercial"):
            logger.info(f"Phase 1.5: 仅编码 {dimension} 页面段落")

        import numpy as np
        import hashlib as _hashlib

        batch_size = max(32, self.config.SBERT_BATCH_SIZE)
        accumulate_size = 512

        # ── 阶段A: 加载所有段落 + 检查已有嵌入缓存 ──
        # 结构: {doc_id: {para_idx: text}}  待编码段落
        #        {doc_id: {para_idx: embedding}}  已有缓存嵌入
        pending_by_doc = {}
        cached_by_doc = {}
        total_paragraphs = 0
        cached_count = 0

        for doc_feat in features:
            if not doc_feat.doc_minhash:
                continue
            try:
                if dimension in ("technical", "commercial"):
                    para_full = self.cache.load_all_paragraphs_full(doc_feat.doc_id)
                    pages = getattr(doc_feat, 'page_classifications', {}) or {}
                    has_target_pages = any(label == dimension for label in pages.values())
                    if pages and has_target_pages:
                        paragraphs = {}
                        for idx, info in sorted(para_full.items()):
                            page_num = info.get('page_num', -1)
                            if pages.get(page_num, 'unknown') == dimension:
                                paragraphs[idx] = info.get('text', '')
                    else:
                        logger.error(
                            f"Phase 1.5: {doc_feat.filename} 未识别到 {dimension} 页面，"
                            "严格模式下不混用全文"
                        )
                        paragraphs = {}
                else:
                    paragraphs = {
                        idx: text
                        for idx, text in enumerate(
                            self.cache.load_all_paragraphs_text(doc_feat.doc_id)
                        )
                    }

                if not paragraphs:
                    continue

                # 检查已有嵌入
                existing = self.cache.load_all_paragraph_embeddings(doc_feat.doc_id)
                pending = {}
                for para_idx, text in paragraphs.items():
                    total_paragraphs += 1
                    if para_idx in existing:
                        cached_by_doc.setdefault(doc_feat.doc_id, {})[para_idx] = existing[para_idx]
                        cached_count += 1
                    else:
                        pending[para_idx] = text

                if pending:
                    pending_by_doc[doc_feat.doc_id] = pending

            except Exception as e:
                logger.error(f"段落加载失败 ({doc_feat.filename}): {e}", exc_info=True)
                continue

        if cached_count > 0:
            logger.info(f"Phase 1.5: 嵌入缓存命中 {cached_count}/{total_paragraphs} 段落 "
                        f"({cached_count/total_paragraphs*100:.1f}%)")

        if not pending_by_doc:
            if total_paragraphs == 0:
                if dimension in ("technical", "commercial"):
                    logger.warning(
                        f"Phase 1.5: {dimension} 维度过滤后没有可编码段落，"
                        "请检查 TOC/页码分类或文本抽取结果"
                    )
                else:
                    logger.warning("Phase 1.5: 没有可编码段落，请检查文本抽取结果")
                state.phase = 3
                self.checkpoint.save(state)
                embed_time = (datetime.now() - embed_start).total_seconds()
                self._timings['phase15_embed'] = embed_time
                logger.info(
                    f"Phase 1.5 完成: 0 个段落 (无可编码文本), 耗时 {embed_time:.2f}s"
                )
                return

            logger.info("Phase 1.5: 所有段落嵌入已缓存，跳过编码")
            # 仍需确保文档级嵌入存在
            self._ensure_document_embeddings(cached_by_doc)
            state.phase = 3
            self.checkpoint.save(state)
            embed_time = (datetime.now() - embed_start).total_seconds()
            self._timings['phase15_embed'] = embed_time
            logger.info(f"Phase 1.5 完成: {total_paragraphs} 个段落 (全部缓存命中), 耗时 {embed_time:.2f}s")
            return

        # ── 阶段B: 段落文本去重（相同文本只编码一次）──
        # text_hash -> (text, [(doc_id, para_idx), ...])
        unique_texts = {}
        for doc_id, paras in pending_by_doc.items():
            for para_idx, text in paras.items():
                if not text:
                    continue
                h = _hashlib.md5(text.encode('utf-8')).hexdigest()
                if h not in unique_texts:
                    unique_texts[h] = (text, [])
                unique_texts[h][1].append((doc_id, para_idx))

        unique_count = len(unique_texts)
        pending_count = sum(len(p) for p in pending_by_doc.values())
        dedup_count = pending_count - unique_count
        if dedup_count > 0:
            logger.info(f"Phase 1.5: 段落去重 {pending_count} -> {unique_count} "
                        f"(减少 {dedup_count}, {dedup_count/pending_count*100:.1f}%)")

        # ── 阶段C: 批量编码唯一文本 ──
        all_texts = [v[0] for v in unique_texts.values()]
        all_hashes = list(unique_texts.keys())
        new_embeddings = {}  # hash -> embedding
        encoded_count = 0
        model_name = 'paraphrase-multilingual-MiniLM-L12-v2'

        text_cache_hits = {}
        if getattr(self.config, 'ENABLE_TEXT_EMBEDDING_CACHE', True):
            cache_lookup_start = datetime.now()
            text_cache_hits = self.cache.load_text_embedding_cache(all_hashes, model_name)
            self._timings['phase15_text_cache_lookup'] = (
                datetime.now() - cache_lookup_start
            ).total_seconds()
            if text_cache_hits:
                logger.info(
                    f"Phase 1.5: text_hash 嵌入缓存命中 "
                    f"{len(text_cache_hits)}/{len(all_hashes)} "
                    f"({len(text_cache_hits)/max(len(all_hashes), 1)*100:.1f}%)"
                )
                new_embeddings.update(text_cache_hits)

        encode_pairs = [
            (h, text) for h, text in zip(all_hashes, all_texts)
            if h not in text_cache_hits
        ]
        encode_hashes = [h for h, _text in encode_pairs]
        encode_texts = [text for _h, text in encode_pairs]
        cache_store_items = []

        encode_start = datetime.now()
        for batch_start in range(0, len(encode_texts), accumulate_size):
            batch_texts = encode_texts[batch_start:batch_start + accumulate_size]
            batch_hashes = encode_hashes[batch_start:batch_start + accumulate_size]
            try:
                logger.debug(f"Phase 1.5: 批量编码 {len(batch_texts)} 个唯一段落...")
                embeddings = engine.model.encode(
                    batch_texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=getattr(self.config, 'NORMALIZE_EMBEDDINGS', True),
                )
                for i, h in enumerate(batch_hashes):
                    new_embeddings[h] = embeddings[i]
                    cache_store_items.append((h, embeddings[i]))
                encoded_count += len(batch_texts)
                logger.debug(f"Phase 1.5: 批量完成，累计 {encoded_count} 个唯一段落")
            except Exception as e:
                logger.error(f"Phase 1.5: 批量编码失败: {e}", exc_info=True)
        self._timings['phase15_sbert_encode'] = (
            datetime.now() - encode_start
        ).total_seconds()

        if cache_store_items and getattr(self.config, 'ENABLE_TEXT_EMBEDDING_CACHE', True):
            cache_store_start = datetime.now()
            self.cache.store_text_embedding_cache(cache_store_items, model_name)
            self._timings['phase15_text_cache_store'] = (
                datetime.now() - cache_store_start
            ).total_seconds()

        # ── 阶段D: 存储嵌入 + 计算文档级嵌入 ──
        # 合并缓存嵌入和新编码嵌入
        all_embeddings_by_doc = {doc_id: dict(cached) for doc_id, cached in cached_by_doc.items()}

        store_batch = []
        for h, (text, locations) in unique_texts.items():
            if h not in new_embeddings:
                continue
            emb = new_embeddings[h]
            for doc_id, para_idx in locations:
                store_batch.append((doc_id, para_idx, emb))
                all_embeddings_by_doc.setdefault(doc_id, {})[para_idx] = emb

        if store_batch:
            self.cache.store_paragraph_embeddings_batch(store_batch)
            logger.debug(f"Phase 1.5: 存储 {len(store_batch)} 个段落嵌入")

        # 计算并存储文档级嵌入（均值池化）
        self._ensure_document_embeddings(all_embeddings_by_doc)

        state.phase = 3
        self.checkpoint.save(state)
        embed_time = (datetime.now() - embed_start).total_seconds()
        self._timings['phase15_embed'] = embed_time
        logger.info(f"Phase 1.5 完成: {total_paragraphs} 个段落 "
                    f"(缓存 {cached_count}, 新编码 {len(store_batch)}, 唯一 {unique_count}), "
                    f"耗时 {embed_time:.2f}s")

    def _ensure_document_embeddings(self, embeddings_by_doc: dict) -> None:
        """计算并存储文档级嵌入（如果尚不存在）"""
        import numpy as np
        existing_doc_embs = self.cache.load_all_document_embeddings()
        for doc_id, para_embs in embeddings_by_doc.items():
            if not para_embs:
                continue
            if doc_id in existing_doc_embs:
                continue
            emb_array = np.array(list(para_embs.values()))
            doc_embedding = np.mean(emb_array, axis=0).astype(np.float32)
            self.cache.store_document_embedding(doc_id, doc_embedding)

    def _get_para_full(self, doc_id: str) -> Dict[int, dict]:
        if doc_id not in self._phase3_para_cache:
            self._phase3_para_cache[doc_id] = self.cache.load_all_paragraphs_full(doc_id)
        return self._phase3_para_cache[doc_id]

    def _filter_para_full_by_dimension(
        self, para_full: Dict[int, dict], doc
    ) -> Dict[int, dict]:
        """过滤段落字典，仅保留指定维度的页面"""
        dimension = self.config.ANALYSIS_DIMENSION
        if dimension not in ("technical", "commercial"):
            return para_full
        pages = getattr(doc, 'page_classifications', {}) or {}
        if not pages:
            logger.error(f"{doc.filename} 缺少页分类，{dimension} 维度严格过滤为空")
            return {}
        if not any(label == dimension for label in pages.values()):
            logger.error(f"{doc.filename} 未识别到 {dimension} 页面，严格过滤为空")
            return {}
        return {
            idx: info for idx, info in para_full.items()
            if pages.get(info.get('page_num', -1), 'unknown') == dimension
        }

    def _filter_hashes_by_dimension(self, hashes, doc) -> list:
        """过滤图片哈希列表，仅保留指定维度的页面

        哈希格式: "page_N:..."
        """
        dimension = self.config.ANALYSIS_DIMENSION
        if dimension not in ("technical", "commercial"):
            return hashes
        pages = getattr(doc, 'page_classifications', {}) or {}
        if not pages:
            logger.error(f"{doc.filename} 缺少页分类，{dimension} 图片维度严格过滤为空")
            return []
        if not any(label == dimension for label in pages.values()):
            logger.error(f"{doc.filename} 未识别到 {dimension} 页面，图片维度严格过滤为空")
            return []
        filtered = []
        for h in (hashes or []):
            if h.startswith('page_'):
                try:
                    page_num = int(h.split(':')[0].split('_')[1])
                    if pages.get(page_num, 'unknown') == dimension:
                        filtered.append(h)
                except (ValueError, IndexError):
                    filtered.append(h)
            else:
                filtered.append(h)
        return filtered

    def _phase3_analyze_single(self, doc_a_id: str, doc_b_id: str):
        doc_a = self.cache.load_document(doc_a_id)
        doc_b = self.cache.load_document(doc_b_id)

        if not doc_a or not doc_b:
            logger.error(f"文档未找到: {doc_a_id} / {doc_b_id}")
            return

        if not doc_a.doc_minhash or not doc_b.doc_minhash:
            if doc_a.image_hashes or doc_b.image_hashes:
                logger.info(
                    f"无文字文档对（仅图片比对）: "
                    f"{doc_a.filename} vs {doc_b.filename}, "
                    f"图片 {len(doc_a.image_hashes)}/{len(doc_b.image_hashes)}"
                )
                self._phase3_image_only(doc_a, doc_b, doc_a_id, doc_b_id)
                return
            else:
                logger.warning(f"文档缺少数据: {doc_a.filename} / {doc_b.filename}, 跳过")
                self.cache.mark_pair_processed(doc_a_id, doc_b_id)
                return

        pair_id = "::".join(sorted([doc_a_id, doc_b_id]))

        self.streaming.activate_document(doc_a_id)
        self.streaming.activate_document(doc_b_id)

        try:
            dims = self.config.ENABLED_DIMENSIONS
            paragraph_matches = []
            if dims.get('content_similarity', True):
                # 按维度过滤段落数据（仅保留技术标或商务标页面）
                para_full_a = self._filter_para_full_by_dimension(
                    self._get_para_full(doc_a_id), doc_a)
                para_full_b = self._filter_para_full_by_dimension(
                    self._get_para_full(doc_b_id), doc_b)
                if not para_full_a or not para_full_b:
                    logger.info(f"维度过滤后无段落可匹配 ({doc_a.filename} vs {doc_b.filename})")
                else:
                    paragraph_matches = self.paragraph_matcher.match(
                        doc_a, doc_b, self.cache,
                        para_full_a=para_full_a,
                        para_full_b=para_full_b,
                        para_embeddings_a=self._phase3_embedding_cache.get(doc_a_id),
                        para_embeddings_b=self._phase3_embedding_cache.get(doc_b_id),
                    )

            # 按维度过滤图片哈希（仅保留指定页面的图片）
            dimension = self.config.ANALYSIS_DIMENSION
            if dimension in ("technical", "commercial"):
                orig_ha = doc_a.image_hashes
                orig_hb = doc_b.image_hashes
                doc_a.image_hashes = self._filter_hashes_by_dimension(orig_ha, doc_a)
                doc_b.image_hashes = self._filter_hashes_by_dimension(orig_hb, doc_b)

            evidence = self._build_evidence(doc_a, doc_b, paragraph_matches)

            # 恢复原始图片哈希（不影响缓存）
            if dimension in ("technical", "commercial"):
                doc_a.image_hashes = orig_ha
                doc_b.image_hashes = orig_hb

            result = PairwiseResult(
                pair_id=pair_id,
                doc_a_id=doc_a_id,
                doc_b_id=doc_b_id,
                similarity_scores={
                    'text_local': evidence.text_evidence.local_similarity,
                    'metadata_match': len(evidence.metadata_evidence.matched_fields),
                    'image_common': evidence.image_evidence.common_image_count,
                },
                evidence=evidence,
            )

            self.cache.store_pairwise_result(result)
            self.cache.mark_pair_processed(doc_a_id, doc_b_id)

            if self._progress_callback:
                self._progress_callback({
                    'pair_id': pair_id,
                    'doc_a_id': doc_a_id,
                    'doc_b_id': doc_b_id,
                    'filename_a': doc_a.filename,
                    'filename_b': doc_b.filename,
                    'text_similarity': evidence.text_evidence.local_similarity,
                    'match_count': len(paragraph_matches),
                    'clone_count': len(evidence.text_evidence.continuous_clone_blocks),
                })
        finally:
            self.streaming.release_document(doc_a_id)
            self.streaming.release_document(doc_b_id)

    def _build_evidence(self, doc_a, doc_b, paragraph_matches):
        dims = self.config.ENABLED_DIMENSIONS
        cs = dims.get('content_similarity', True)

        text_evidence = TextEvidence()
        if cs:
            text_evidence = self._build_text_evidence(doc_a, doc_b, paragraph_matches)

        metadata_evidence = MetadataEvidence()
        if dims.get('file_id', True) or dims.get('author', True) or dims.get('editor', True):
            metadata_evidence = self._build_metadata_evidence(doc_a, doc_b)

        image_evidence = ImageEvidence()
        if cs and getattr(self.config, 'ENABLE_IMAGE_ANALYSIS', True):
            image_evidence = self._build_image_evidence(doc_a, doc_b)

        contact_evidence = ContactEvidence()
        if any(dims.get(k, True) for k in ['contact', 'company_name', 'credit_code', 'member_id']):
            contact_evidence = build_contact_evidence(doc_a.doc_id, doc_b.doc_id, self.cache)

        return EvidenceChain(
            text_evidence=text_evidence,
            metadata_evidence=metadata_evidence,
            image_evidence=image_evidence,
            contact_evidence=contact_evidence,
        )

    @staticmethod
    def _build_metadata_evidence(doc_a, doc_b):
        return build_metadata_evidence(doc_a, doc_b)

    def _build_text_evidence(self, doc_a, doc_b, paragraph_matches):
        return build_text_evidence(doc_a, doc_b, paragraph_matches, self.config, compute_highlight=False)

    def _build_image_evidence(self, doc_a, doc_b):
        evidence = ImageEvidence()
        boilerplate_hashes = set(self.config.IMAGE_BOILERPLATE_HASHES) if self.config.IMAGE_BOILERPLATE_HASHES else None

        hashes_a = doc_a.image_hashes
        hashes_b = doc_b.image_hashes
        common_exact = list(set(hashes_a) & set(hashes_b))
        evidence.common_image_count = len(common_exact)
        evidence.common_image_hashes = common_exact

        if self.config.ENABLE_OCR:
            ocr_a = self.cache.load_image_ocr_results(doc_a.doc_id)
            ocr_b = self.cache.load_image_ocr_results(doc_b.doc_id)
        else:
            ocr_a = []
            ocr_b = []

        if ocr_a:
            evidence.ocr_results_a = ocr_a
        if ocr_b:
            evidence.ocr_results_b = ocr_b

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

        match_result = self.image_matcher.analyze(
            hashes_a=hashes_a, hashes_b=hashes_b,
            ocr_results_a=ocr_objects_a if ocr_objects_a else None,
            ocr_results_b=ocr_objects_b if ocr_objects_b else None,
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

        for v in match_result.image_verdicts:
            thumb_a_b64 = self._thumbnail_to_base64(v.sig_a.thumbnail)
            thumb_b_b64 = self._thumbnail_to_base64(v.sig_b.thumbnail)
            ocr_text_a = self._find_ocr_text_by_hash(ocr_objects_a, v.sig_a.phash or v.sig_a.dhash)
            ocr_text_b = self._find_ocr_text_by_hash(ocr_objects_b, v.sig_b.phash or v.sig_b.dhash)
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
                # 维度标签
                'page_a': v.sig_a.page_num,
                'page_b': v.sig_b.page_num,
                '_dimension_tag': _get_image_dimension_tag(
                    v.sig_a.page_num, v.sig_b.page_num, doc_a, doc_b
                ),
            })
        evidence.matched_text_pairs = match_result.text_matches
        evidence.ps_detail_list = match_result.ps_details
        return evidence

    @staticmethod
    def _thumbnail_to_base64(thumb_bytes: bytes) -> str:
        """将缩略图字节转为 base64 data URI"""
        if not thumb_bytes:
            return ''
        import base64
        encoded = base64.b64encode(thumb_bytes).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded}"

    @staticmethod
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

    def _phase3_image_only(self, doc_a, doc_b, doc_a_id, doc_b_id):
        pair_id = "::".join(sorted([doc_a_id, doc_b_id]))
        dims = self.config.ENABLED_DIMENSIONS

        # 按维度过滤图片哈希
        dimension = self.config.ANALYSIS_DIMENSION
        if dimension in ("technical", "commercial"):
            orig_ha = doc_a.image_hashes
            orig_hb = doc_b.image_hashes
            doc_a.image_hashes = self._filter_hashes_by_dimension(orig_ha, doc_a)
            doc_b.image_hashes = self._filter_hashes_by_dimension(orig_hb, doc_b)

        image_evidence = ImageEvidence()
        if dims.get('content_similarity', True) and getattr(self.config, 'ENABLE_IMAGE_ANALYSIS', True):
            image_evidence = self._build_image_evidence(doc_a, doc_b)

        # 恢复原始图片哈希
        if dimension in ("technical", "commercial"):
            doc_a.image_hashes = orig_ha
            doc_b.image_hashes = orig_hb

        metadata_evidence = MetadataEvidence()
        if dims.get('file_id', True) or dims.get('author', True) or dims.get('editor', True):
            metadata_evidence = self._build_metadata_evidence(doc_a, doc_b)

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

        self.cache.store_pairwise_result(result)
        self.cache.mark_pair_processed(doc_a_id, doc_b_id)

    @staticmethod
    def _empty_report() -> GlobalReport:
        return GlobalReport(
            report_id="empty", high_risk_pairs=0,
            generated_at=datetime.now().isoformat(),
            total_files=0, total_pairs=0, candidate_pairs=0,
            suspicious_pairs=0,
        )

    def _compute_input_hash(self, input_dir: str) -> str:
        import hashlib
        import json

        file_info = []
        for root, dirs, files in os.walk(input_dir):
            for filename in sorted(files):
                if filename.lower().endswith('.pdf'):
                    file_path = os.path.join(root, filename)
                    try:
                        file_stat = os.stat(file_path)
                        file_info.append({
                            'path': file_path,
                            'size': file_stat.st_size,
                            'mtime': file_stat.st_mtime,
                        })
                    except (OSError, IOError):
                        continue

        file_info_str = json.dumps(file_info, sort_keys=True)
        return hashlib.md5(file_info_str.encode()).hexdigest()
