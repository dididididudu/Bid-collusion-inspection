"""
6 阶段流式管道编排器 (v2)

管理完整的检测流程:
  Phase 0: SCAN    - 扫描目录，收集 PDF 元数据
  Phase 1: EXTRACT - 多进程分块文本提取（PyMuPDF）
  Phase 1.5: EMBED - 全局 SBERT 嵌入编码（一次性）
  Phase 2: SELECT  - 候选对筛选（LSH + 元数据 + 文档向量预筛）
  Phase 3: ANALYZE - 逐对精细分析（查表点积，不调模型）
  Phase 4: SCORE   - 风险评分与聚类
  Phase 5: REPORT  - 生成报告

每一阶段都支持断点续传。
"""

import os
import glob
import logging
from datetime import datetime
from typing import List, Dict
from concurrent.futures import (
    ProcessPoolExecutor, ThreadPoolExecutor, as_completed
)

from config import DetectionConfig
from data_structures import (
    BidFeature, GlobalReport, PairwiseResult, EvidenceChain,
    TextEvidence, MetadataEvidence, ImageEvidence
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
    build_metadata_evidence, build_image_evidence, build_text_evidence,
)
from pipeline.ocr_helpers import ocr_pages, aggregate_ocr_paragraphs

logger = logging.getLogger(__name__)


class BidDetectionOrchestrator:
    """5 阶段流式管道编排器"""

    # OCR 聚合段落的虚拟块索引（避免与正常文本块 0,1,2... 冲突）
    OCR_CHUNK_INDEX: int = 1000000

    def __init__(self, config: DetectionConfig):
        self.config = config

        # 初始化缓存和检查点
        self.cache = DocumentCache(config.CACHE_DIR, config)
        self.checkpoint = CheckpointManager(config.CHECKPOINT_DIR, config)
        self.streaming = StreamingContext(self.cache, config.MAX_CHUNKS_IN_MEMORY)

        # 初始化各阶段引擎
        self.extractor = PyMuPDFExtractor(config)
        self.text_processor = ChunkedTextProcessor(config)
        self.selector = CandidatePairSelector(config)
        self.paragraph_matcher = ParagraphMatcher(config)
        self.scoring_engine = RiskScoringEngine(config)
        self.report_generator = ReportGenerator(config)

        # SBERT 嵌入引擎（跨阶段复用：Phase 1.5 编码 + Phase 3 匹配）
        self.embedding_engine = EmbeddingEngine(config)

        # 图片分析引擎（OCR + 四层检测）
        self.ocr_engine = ImageOCREngine(
            use_gpu=config.USE_GPU,
            engine=config.OCR_ENGINE,
            model_dir=config.OCR_MODEL_DIR,
            offline=config.OCR_OFFLINE_MODE,
            retry_count=config.OCR_RETRY_COUNT,
        )
        self.image_matcher = ImageMatcher()

        # GPU 资源管理器（多进程场景统一 GPU 访问，避免 OOM）
        from pipeline.gpu_manager import GPUManager
        self.gpu_manager = GPUManager(config)

        # 跨阶段缓存：避免重复从 SQLite 加载文档特征
        self._all_features_cache = None

        logger.info("流式管道编排器已初始化")

    def _get_picklable_config(self) -> dict:
        """生成可 pickle 序列化的配置字典（供 worker 进程传输）"""
        return {
            k: v for k, v in self.config.__dict__.items()
            if not k.startswith('_')
        }

    def detect(self, input_dir: str, output_dir: str) -> GlobalReport:
        """执行完整的 5 阶段检测流程

        Args:
            input_dir: 输入 PDF 目录
            output_dir: 报告输出目录

        Returns:
            GlobalReport 检测报告
        """
        process_start = datetime.now()

        logger.info("=" * 60)
        logger.info("流式管道检测开始")
        logger.info("=" * 60)

        # 启动时 OCR 引擎健康检查
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
            # 同时删除物理数据库文件，避免残留锁文件干扰
            for suffix in ['', '-wal', '-shm']:
                f = self.cache.db_path + suffix
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except OSError:
                        pass

        # 计算输入文件夹内容哈希（检测文件变化）
        input_hash = self._compute_input_hash(input_dir)

        # 加载检查点
        state = self.checkpoint.load_or_new()

        if not self.config.DISABLE_CACHE:
            # 检测输入文件夹变化，如果变化则重置检查点
            if state.input_hash and state.input_hash != input_hash:
                logger.warning(
                    f"检测到输入文件夹内容变化！旧哈希: {state.input_hash}, "
                    f"新哈希: {input_hash}。将重新开始处理。"
                )
                self.checkpoint.clear()
                self.cache.clear_cache()
                state = self.checkpoint.load_or_new()

        # 更新输入哈希
        state.input_hash = input_hash

        try:
            # === Phase 0: SCAN ===
            # 重新扫描目录获取当前文件列表
            file_paths = self._scan_pdf_files(input_dir)
            if not file_paths:
                logger.error("未找到 PDF 文件")
                return self._empty_report()

            if state.phase < 1:
                logger.info("Phase 0: 扫描文档...")
                self._phase0_metadata(file_paths)
                state.phase = 1
                self.checkpoint.save(state)
                logger.info(
                    f"Phase 0 完成: {len(file_paths)} 个 PDF 文件"
                )

            # 启动 GPU Manager（Phase 1 OCR 需要）
            self.gpu_manager.start()

            # === Phase 1: EXTRACT ===
            if state.phase < 2:
                logger.info("Phase 1: 提取特征...")
                # 找出尚未提取的文件
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
                        self.config.PHASE1_WORKERS,
                        len(unprocessed),
                        os.cpu_count() or 4,
                    )

                    if num_workers <= 1:
                        # === 串行路径 ===
                        for file_path in unprocessed:
                            try:
                                self._phase1_extract_single(file_path)
                                state.processed_files.add(file_path)
                                processed_count += 1

                                if processed_count % 10 == 0:
                                    logger.info(
                                        f"Phase 1 进度: {processed_count}/{len(unprocessed)}"
                                    )
                                    self.checkpoint.save(state)

                            except Exception as e:
                                logger.error(
                                    f"提取失败 ({file_path}): {e}", exc_info=True
                                )
                                continue
                    else:
                        # === 并行路径 (ProcessPoolExecutor) ===
                        logger.info(
                            f"Phase 1 并行模式: {num_workers} workers, "
                            f"{len(unprocessed)} 个文件"
                        )
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
                                        logger.error(
                                            f"提取失败 ({file_path}): "
                                            f"{result.get('error', 'unknown')}"
                                        )
                                except Exception as e:
                                    logger.error(
                                        f"Worker 异常 ({file_path}): {e}",
                                        exc_info=True,
                                    )

                                if processed_count % self.config.CHECKPOINT_INTERVAL == 0:
                                    self.checkpoint.save(state)

                state.phase = 2
                self.checkpoint.save(state)

                extract_time = (datetime.now() - extract_start).total_seconds()
                logger.info(
                    f"Phase 1 完成: {processed_count}/{len(unprocessed)} 个文档, "
                    f"耗时 {extract_time:.2f}s"
                )

            # === Phase 1.5: EMBED (全局 SBERT 嵌入编码) ===
            if state.phase < 3:
                self._phase_embed(state)

            # Phase 1.5 完成后，将已加载的 SBERT 模型注入 Phase 3 的匹配器
            if (self.embedding_engine.is_available
                    and self.embedding_engine.model is not None
                    and self.paragraph_matcher is not None):
                self.paragraph_matcher._ensure_semantic_matcher()
                self.paragraph_matcher.semantic_matcher.set_model(
                    self.embedding_engine.model
                )
                logger.debug("Phase 1.5 SBERT 模型已注入 Phase 3 ParagraphMatcher")

            # === Phase 2: SELECT ===
            if state.phase < 4:
                logger.info("Phase 2: 候选对选择...")
                select_start = datetime.now()

                features = self.cache.load_all_documents()
                self._all_features_cache = features  # 缓存供后续阶段复用
                candidates = self.selector.select(features, cache=self.cache)

                # 存入缓存
                self.cache.store_candidate_pairs(candidates)

                state.total_pairs = len(candidates)
                state.phase = 3
                self.checkpoint.save(state)

                select_time = (datetime.now() - select_start).total_seconds()
                logger.info(
                    f"Phase 2 完成: {len(candidates)} 对候选, "
                    f"耗时 {select_time:.2f}s"
                )

            # === Phase 3: ANALYZE ===
            if state.phase < 5:
                logger.info("Phase 3: 精细分析...")

                # 加载已完成的配对（恢复模式）
                completed_ids = self.checkpoint.load_phase3_progress()
                state.completed_pair_ids = completed_ids

                # 获取所有未处理的候选对
                all_pairs = self.cache.get_unprocessed_pairs()
                # 过滤掉已完成的
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
                        self.config.PHASE3_WORKERS,
                        total_pending,
                        (os.cpu_count() or 4) * 2,  # 线程可超过 CPU 数
                    )

                    if num_workers <= 1:
                        # === 串行路径 ===
                        for idx, (doc_a_id, doc_b_id) in enumerate(pending_pairs, 1):
                            try:
                                self._phase3_analyze_single(doc_a_id, doc_b_id)

                                pair_id = "::".join(sorted([doc_a_id, doc_b_id]))
                                completed_ids.add(pair_id)
                                state.completed_pairs = len(completed_ids)

                                if idx % 50 == 0 or idx == total_pending:
                                    logger.info(
                                        f"Phase 3 进度: {idx}/{total_pending} "
                                        f"({state.completed_pairs}/{state.total_pairs})"
                                    )

                                if idx % self.config.CHECKPOINT_INTERVAL == 0:
                                    self.checkpoint.save_phase3_progress(completed_ids)
                                    self.checkpoint.save(state)
                                    self.cache.conn.commit()

                            except Exception as e:
                                logger.error(
                                    f"分析失败 ({doc_a_id} vs {doc_b_id}): {e}",
                                    exc_info=True
                                )
                                self.cache.mark_pair_processed(doc_a_id, doc_b_id)
                                continue

                        self.cache.conn.commit()
                    else:
                        # === 并行路径 (ThreadPoolExecutor) ===
                        logger.info(
                            f"Phase 3 并行模式: {num_workers} workers, "
                            f"{total_pending} 对候选"
                        )
                        from pipeline.parallel_workers import analyze_pair_worker

                        config_dict = self._get_picklable_config()
                        db_dir = self.config.CACHE_DIR

                        # 预先加载 SBERT 模型一次，所有线程共享
                        self.paragraph_matcher._ensure_semantic_matcher()
                        shared_matcher = self.paragraph_matcher.semantic_matcher

                        with ThreadPoolExecutor(max_workers=num_workers) as executor:
                            future_to_pair = {}
                            for doc_a_id, doc_b_id in pending_pairs:
                                future = executor.submit(
                                    analyze_pair_worker,
                                    (doc_a_id, doc_b_id, config_dict, db_dir,
                                     shared_matcher)
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
                                except Exception as e:
                                    logger.error(
                                        f"分析 worker 异常 "
                                        f"({doc_a_id} vs {doc_b_id}): {e}",
                                        exc_info=True,
                                    )

                                if done_count % 50 == 0 or done_count == total_pending:
                                    logger.info(
                                        f"Phase 3 进度: {done_count}/{total_pending} "
                                        f"({len(completed_ids)}/{state.total_pairs})"
                                    )

                                if done_count % self.config.CHECKPOINT_INTERVAL == 0:
                                    self.checkpoint.save_phase3_progress(completed_ids)
                                    self.checkpoint.save(state)

                    analyze_time = (
                        datetime.now() - analyze_start
                    ).total_seconds()
                    logger.info(
                        f"Phase 3 完成: {len(completed_ids)} 对, "
                        f"耗时 {analyze_time:.2f}s"
                    )

                state.phase = 5
                state.completed_pairs = len(completed_ids)
                self.checkpoint.save(state)

            # === Phase 4: SCORE ===
            report = None
            if state.phase < 6:
                logger.info("Phase 4: 风险评分...")
                score_start = datetime.now()

                # 从 SQLite 加载结果（复用 Phase 2 缓存的文档特征）
                pairwise_results = self.cache.load_all_results()
                features = self._all_features_cache or self.cache.load_all_documents()

                # 生成报告
                report = self.scoring_engine.generate_report(
                    pairwise_results, features
                )

                state.phase = 6
                self.checkpoint.save(state)

                score_time = (datetime.now() - score_start).total_seconds()
                logger.info(
                    f"Phase 4 完成: {report.suspicious_pairs} 对可疑, "
                    f"{report.high_risk_pairs} 对高风险, "
                    f"耗时 {score_time:.2f}s"
                )

            # === Phase 5: REPORT ===
            if state.phase >= 6 and report is None:
                # 恢复模式：从缓存重新生成报告
                pairwise_results = self.cache.load_all_results()
                features = self._all_features_cache or self.cache.load_all_documents()
                report = self.scoring_engine.generate_report(
                    pairwise_results, features
                )

            if report is not None:
                logger.info("Phase 5: 报告生成...")
                report_start = datetime.now()
                self.report_generator.generate(report, output_dir)
                report_time = (datetime.now() - report_start).total_seconds()
                logger.info(
                    f"Phase 5 完成: 报告已输出到 {output_dir}, "
                    f"耗时 {report_time:.2f}s"
                )
            else:
                report = self._empty_report()
                logger.warning("未生成任何报告数据")

            # 完成
            total_time = (datetime.now() - process_start).total_seconds()
            logger.info("=" * 60)
            logger.info(f"检测完成! 总耗时: {total_time:.2f}s")
            logger.info("=" * 60)

            return report

        except Exception as e:
            logger.error(f"管道执行失败: {e}", exc_info=True)
            # 保存当前状态以供恢复
            self.checkpoint.save(state)
            raise

        finally:
            self.gpu_manager.shutdown()
            self.streaming.clear()
            self.cache.close()

    # ============================================================
    # Phase 0: 扫描和元数据收集
    # ============================================================

    @staticmethod
    def _scan_pdf_files(input_dir: str) -> List[str]:
        """扫描目录获取 PDF 文件列表"""
        pattern = os.path.join(input_dir, "*.pdf")
        files = glob.glob(pattern)
        logger.info(f"扫描到 {len(files)} 个 PDF 文件")
        return files

    def _phase0_metadata(self, file_paths: List[str]):
        """Phase 0: 收集所有 PDF 的元数据"""
        for file_path in file_paths:
            try:
                metadata, page_count, is_scanned = (
                    self.extractor.extract_metadata(file_path)
                )
                doc_id = self.extractor._generate_doc_id(file_path)
                filename = os.path.basename(file_path)
                file_size = os.path.getsize(file_path)

                # 创建轻量级特征并存储
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
                    is_scanned=is_scanned,
                    page_count=page_count,
                    doc_minhash=None,
                    chunk_count=0,
                )
                self.cache.store_document(feature)
                # 存储元数据指纹（供 Phase 2 筛选使用）
                self.cache.store_metadata_fingerprint(
                    doc_id=doc_id,
                    author=metadata.author or '',
                    creator=metadata.creator or '',
                    producer=metadata.producer or '',
                    software_fingerprint=metadata.software_fingerprint or '',
                    time_bucket=metadata.time_bucket or '',
                )
                logger.debug(
                    f"Phase 0: {filename} ({page_count} 页, "
                    f"{'扫描版' if is_scanned else '文本版'})"
                )

            except Exception as e:
                logger.error(f"Phase 0 失败 ({file_path}): {e}")

    # ============================================================
    # Phase 1: 单文档提取
    # ============================================================

    def _phase1_extract_single(self, file_path: str):
        """Phase 1: 提取单个文档的特征

        处理策略:
        - 文本版 PDF: 提取文本 + 嵌入图片 + 页级图片采样
        - 扫描版 PDF: 提取页级图片哈希（作为主要特征）+ 少量文本（如有）
        """
        logger.info(f"Phase 1: 提取 {os.path.basename(file_path)}...")

        # 提取元数据
        metadata, page_count, is_scanned = self.extractor.extract_metadata(file_path)
        doc_id = self.extractor._generate_doc_id(file_path)
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)

        # 检查是否已有部分处理（断点续传）
        existing_chunks = self.cache.load_document_chunks(doc_id)
        processed_chunks = {c.chunk_index for c in existing_chunks}

        if is_scanned:
            # === 扫描版 PDF：主路径是图片比对 ===
            logger.info(f"检测到扫描版 PDF: {filename} ({page_count} 页)")

            # 提取页级图片哈希（主要特征）
            all_page_hashes = self.extractor.extract_all_page_hashes(
                file_path, sample_step=2
            )
            logger.info(
                f"  页面哈希提取完成: {len(all_page_hashes)} 个哈希 "
                f"({page_count} 页, 每 2 页采样 1 次)"
            )

            # 为扫描版创建一个"虚拟块"来存储图片哈希
            # 同时尝试提取少量文本（扫描版可能也有部分文本层）
            chunks = []
            for chunk_result in self.extractor.extract_chunks(
                file_path, self.config.CHUNK_PAGE_SIZE, 0
            ):
                chunk_result.image_hashes = all_page_hashes
                self.cache.store_chunk(chunk_result)
                chunks.append(chunk_result)

            if chunks:
                feature = self.text_processor.aggregate_chunks(
                    doc_id=doc_id,
                    filename=filename,
                    file_size=file_size,
                    chunks=chunks,
                    metadata=metadata,
                    is_scanned=True,
                    page_count=page_count,
                )
                # 确保页级图片哈希被记录
                feature.image_hashes = list(set(
                    feature.image_hashes + all_page_hashes
                ))
                self.cache.store_document(feature)
                # 扫描版 PDF：OCR 是必须的（提取文字用于后续文本比对）
                ocr_count = self._phase1_ocr_pages(
                    file_path, doc_id, page_count, force=True
                )
                # OCR 文字聚合为段落 → 注入文本匹配管线
                ocr_para_count = self._aggregate_ocr_to_paragraphs(
                    doc_id, page_count
                )
                logger.info(
                    f"Phase 1 完成 (扫描版): {filename} "
                    f"({page_count} 页, {len(all_page_hashes)} 页面哈希, "
                    f"OCR: {ocr_count} 页, {ocr_para_count} 段)"
                )
            else:
                # 完全没有文本，只存图片哈希
                from data_structures import QuoteSignature
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
                    quote_signature=QuoteSignature(),
                    image_hashes=all_page_hashes,
                    is_scanned=True,
                    page_count=page_count,
                    doc_minhash=None,
                    chunk_count=0,
                )
                self.cache.store_document(feature)
                # 纯扫描版 PDF：OCR 是唯一的文字来源
                ocr_count = self._phase1_ocr_pages(
                    file_path, doc_id, page_count, force=True
                )
                ocr_para_count = self._aggregate_ocr_to_paragraphs(
                    doc_id, page_count
                )
                logger.info(
                    f"Phase 1 完成 (纯扫描版): {filename} "
                    f"({page_count} 页, {len(all_page_hashes)} 页面哈希, "
                    f"OCR: {ocr_count} 页, {ocr_para_count} 段)"
                )
            return

        # === 文本版 PDF：文本为主，图片为辅 ===
        start_page = max(
            (max(processed_chunks) + 1) * self.config.CHUNK_PAGE_SIZE
            if processed_chunks else 0,
            0
        )

        if start_page >= page_count and page_count > 0:
            logger.info(f"文档已完全提取: {filename}")
            return

        # 流式提取文本块（含嵌入图片和页级图片采样）
        chunks = []
        chunk_size = self.config.CHUNK_PAGE_SIZE

        for chunk_result in self.extractor.extract_chunks(
            file_path, chunk_size, start_page
        ):
            # 存储块到 SQLite
            self.cache.store_chunk(chunk_result)
            chunks.append(chunk_result)
            img_count = len(chunk_result.image_hashes)
            logger.debug(
                f"  块 {chunk_result.chunk_index}: "
                f"页 {chunk_result.start_page}-{chunk_result.end_page}, "
                f"{len(chunk_result.paragraphs)} 段, "
                f"{img_count} 图片哈希"
            )

        # 聚合所有块的特征
        if chunks:
            # 聚合文档特征
            feature = self.text_processor.aggregate_chunks(
                doc_id=doc_id,
                filename=filename,
                file_size=file_size,
                chunks=chunks,
                metadata=metadata,
                is_scanned=False,
                page_count=page_count,
            )

            # 确保图片哈希被合并（去重）
            all_img_hashes = set()
            for c in chunks:
                all_img_hashes.update(c.image_hashes)
            feature.image_hashes = list(all_img_hashes)

            # 更新文档特征
            self.cache.store_document(feature)

            # 自动 OCR：对页面图片提取文字（如有图片且 OCR 启用）
            ocr_count = self._phase1_ocr_pages(
                file_path, doc_id, page_count
            )

            logger.info(
                f"Phase 1 完成: {filename} "
                f"({page_count} 页, {len(chunks)} 块, "
                f"{feature.text_length} 字符, "
                f"{len(feature.image_hashes)} 图片哈希)"
            )

    # ============================================================
    # Phase 1 OCR: 自动页面图片文字提取
    # ============================================================

    def _phase1_ocr_pages(
        self, file_path: str, doc_id: str, page_count: int,
        force: bool = False,
    ) -> int:
        """对 PDF 中的图片运行 OCR 提取文字

        两种模式:
        - 扫描版 (force=True): 渲染整页为图片 → OCR（页面本身就是图片）
        - 文本版 (force=False): 提取嵌入图片 → 逐个 OCR（只识别图片中的文字）

        Returns:
            成功 OCR 的图片数
        """
        if not self.config.ENABLE_OCR:
            return 0

        # GPU Manager 路径：委托给 ocr_helpers.ocr_pages（统一实现）
        if self.gpu_manager.enabled:
            return ocr_pages(
                file_path, doc_id, page_count,
                self.cache, self.config, self.ocr_engine,
                force=force, ocr_workers=self.config.OCR_WORKERS,
                gpu_manager=self.gpu_manager.client,
            )

        if not self.ocr_engine.is_available:
            logger.debug("OCR 引擎不可用，跳过图片文字提取")
            return 0

        # 如果已有 OCR 结果，跳过（避免重复）
        if not force:
            existing_ocr = self.cache.load_image_ocr_results(doc_id)
            if existing_ocr:
                logger.debug(
                    f"OCR: {os.path.basename(file_path)} 已有 "
                    f"{len(existing_ocr)} 条结果，跳过"
                )
                return len(existing_ocr)

        import fitz
        from PIL import Image
        import io
        import imagehash
        import numpy as np

        logger.info(
            f"OCR: {'扫描版全页' if force else '嵌入图片'}模式 "
            f"{os.path.basename(file_path)} ({page_count} 页)..."
        )

        ocr_count = 0
        sample_step = self.config.OCR_SAMPLE_STEP
        min_conf = self.config.OCR_MIN_CONFIDENCE

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
                        # === 扫描版：渲染整页 → OCR ===
                        pix = page.get_pixmap(dpi=150)
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        phash = str(imagehash.phash(img))
                        img_array = np.array(img)

                        ocr_result = self.ocr_engine.extract(img_array)
                        ocr_result.image_hash = phash

                        if ocr_result.confidence >= min_conf and ocr_result.text.strip():
                            self.cache.store_image_ocr_result(
                                doc_id=doc_id, page_num=page_num,
                                image_hash=phash,
                                ocr_text=ocr_result.text,
                                ocr_words=ocr_result.words,
                                bboxes=ocr_result.bboxes,
                                confidence=ocr_result.confidence,
                            )
                            ocr_count += 1
                    else:
                        # === 文本版：渲染页面 → 裁剪图片区域 → OCR ===
                        # 使用 get_image_info() 获取图片在页面上的位置，
                        # 然后从页面渲染图中裁剪出来。比 extract_image()
                        # 更可靠（extract_image 可能丢失颜色/透明度信息）。
                        image_info_list = page.get_image_info()
                        if not image_info_list:
                            continue

                        # 过滤太小的装饰图，收集有效图片区域
                        valid_images = []
                        min_size = getattr(self.config, 'IMAGE_MIN_SIZE', 50)
                        for info in image_info_list:
                            w = info.get('width', 0)
                            h = info.get('height', 0)
                            if w >= min_size and h >= min_size:
                                valid_images.append(info)

                        if not valid_images:
                            continue

                        # 渲染整页一次（200 DPI），然后裁剪各图片区域
                        OCR_DPI = 200
                        scale = OCR_DPI / 72.0
                        pix = page.get_pixmap(dpi=OCR_DPI)
                        full_img = Image.open(io.BytesIO(
                            pix.tobytes("png")
                        ))

                        for info in valid_images:
                            try:
                                bbox = info.get('bbox', (0, 0, 0, 0))
                                x0, y0, x1, y1 = bbox

                                # 页面坐标 → 像素坐标
                                px0 = int(x0 * scale)
                                py0 = int(y0 * scale)
                                px1 = int(x1 * scale)
                                py1 = int(y1 * scale)

                                # 裁剪
                                crop = full_img.crop((px0, py0, px1, py1))
                                if crop.size[0] < 10 or crop.size[1] < 10:
                                    continue

                                phash = str(imagehash.phash(crop))
                                img_array = np.array(crop)

                                ocr_result = self.ocr_engine.extract(img_array)
                                ocr_result.image_hash = phash

                                if ocr_result.confidence < min_conf:
                                    continue
                                if not ocr_result.text.strip():
                                    continue

                                self.cache.store_image_ocr_result(
                                    doc_id=doc_id, page_num=page_num,
                                    image_hash=phash,
                                    ocr_text=ocr_result.text,
                                    ocr_words=ocr_result.words,
                                    bboxes=ocr_result.bboxes,
                                    confidence=ocr_result.confidence,
                                )
                                ocr_count += 1

                            except Exception as e:
                                logger.debug(
                                    f"OCR: 第 {page_num} 页裁剪区失败: {e}"
                                )
                                continue

                except Exception as e:
                    logger.debug(
                        f"OCR: 第 {page_num} 页失败 "
                        f"({os.path.basename(file_path)}): {e}"
                    )
                    continue

        finally:
            doc.close()

        if ocr_count > 0:
            logger.info(
                f"OCR: {os.path.basename(file_path)} — "
                f"{ocr_count} 张图片成功提取文字"
            )

        return ocr_count

    # ============================================================
    # Phase 1 OCR 聚合: OCR 文字 → 段落 → MinHash → 文本匹配管线
    # ============================================================

    def _aggregate_ocr_to_paragraphs(
        self, doc_id: str, page_count: int
    ) -> int:
        """将 OCR 结果聚合为段落并注入文本匹配管线

        OCR 文字提取后，需要转换为段落结构才能参与下游的
        LSH 候选筛选、SBERT 嵌入编码、段落匹配等流程。

        流程:
        1. 从 SQLite 加载 OCR 结果
        2. 拼接所有页面 OCR 文字
        3. 调用 extractor._split_paragraphs() 分段
        4. 计算每段的 MinHash
        5. 创建虚拟 ChunkResult 存入 SQLite（段落自动写入）
        6. 更新 BidFeature: text_length, simhash, doc_minhash

        Returns:
            创建的段落数量（0 = 无可用 OCR 文字）
        """
        import jieba
        from data_structures import ChunkResult

        # 1. 加载 OCR 结果
        ocr_results = self.cache.load_image_ocr_results(doc_id)
        if not ocr_results:
            logger.debug(f"OCR 聚合: {doc_id} 无 OCR 结果，跳过")
            return 0

        # 2. 按页码排序，拼接所有 OCR 文字
        ocr_sorted = sorted(ocr_results, key=lambda r: r.get('page_num', 0))
        all_text = "\n".join(
            r['ocr_text'] for r in ocr_sorted
            if r.get('ocr_text', '').strip()
        )
        if not all_text.strip():
            logger.debug(f"OCR 聚合: {doc_id} OCR 文字为空，跳过")
            return 0

        logger.info(
            f"OCR 聚合: {doc_id} — {len(all_text)} 字符, "
            f"{len(ocr_results)} 页 OCR 结果"
        )

        # 3. 分段（复用 extractor 的 _split_paragraphs）
        paragraphs = self.extractor._split_paragraphs(all_text)
        if not paragraphs:
            logger.debug(f"OCR 聚合: {doc_id} 无法分段，跳过")
            return 0

        # 4. 分词 + 预计算词哈希缓存（复用 extractor 的 hash functions）
        stopwords = self.extractor.stopwords
        all_tokens = [
            w for w in jieba.cut(all_text)
            if w not in stopwords and len(w) > 1
        ]

        # 预计算所有唯一词的哈希值（避免每段重复计算）
        word_hash_cache = {}
        unique_words = set(all_tokens)
        for w in unique_words:
            word_hash_cache[w] = [
                hf(w) for hf in self.extractor._minhash_funcs
            ]

        # 5. 计算每段的 MinHash
        paragraph_hashes = []
        for para in paragraphs:
            para_words = [
                w for w in jieba.cut(para)
                if w not in stopwords and len(w) > 1
            ]
            para_hash = self.extractor._compute_minhash_cached(
                para_words, word_hash_cache
            )
            paragraph_hashes.append(para_hash)

        # 6. 计算文档级 SimHash
        simhash = (
            self.extractor._compute_simhash_from_tokens(all_tokens)
            if all_tokens else "0" * 16
        )

        # 7. 提取报价
        quotes = self.extractor._extract_quotes(all_text)

        # 8. 创建虚拟 ChunkResult（使用 OCR_CHUNK_INDEX 避免与文本块冲突）
        chunk_result = ChunkResult(
            doc_id=doc_id,
            chunk_index=self.OCR_CHUNK_INDEX,
            start_page=0,
            end_page=page_count - 1 if page_count > 0 else 0,
            text=all_text,
            paragraphs=paragraphs,
            paragraph_hashes=paragraph_hashes,
            simhash=simhash,
            quotes=quotes,
            image_hashes=[],
        )

        # 存储到 SQLite（段落自动写入 paragraphs 表）
        self.cache.store_chunk(chunk_result)

        # 9. 更新 BidFeature（填充 text_length, simhash, doc_minhash）
        feature = self.cache.load_document(doc_id)
        if feature:
            doc_minhash = self.text_processor._aggregate_minhash(paragraph_hashes)
            feature.text_length = len(all_text)
            feature.text_simhash = simhash
            feature.doc_minhash = doc_minhash
            feature.chunk_count = 1
            self.cache.store_document(feature)
            logger.info(
                f"OCR 聚合完成: {doc_id} → {len(paragraphs)} 段, "
                f"MinHash 已填充"
            )
        else:
            logger.warning(f"OCR 聚合: {doc_id} 文档特征未找到，无法更新")

        return len(paragraphs)

    # ============================================================
    # Phase 1.5 / Phase 3: 嵌入编码 + 单对分析
    # ============================================================

    def _phase_embed(self, state) -> None:
        """Phase 1.5: 全局 SBERT 嵌入编码

        一次性编码所有文档的所有段落嵌入并持久化到 SQLite。
        后续 Phase 3 分析只需查表加载嵌入 + 点积，不再调用 SBERT 模型。
        """
        logger.info("Phase 1.5: 全局 SBERT 嵌入编码...")
        embed_start = datetime.now()

        engine = self.embedding_engine
        if not engine.is_available:
            logger.warning("SBERT 不可用，跳过 Phase 1.5 嵌入编码")
            state.phase = 3
            self.checkpoint.save(state)
            return

        # 获取所有已提取的文档
        features = self.cache.load_all_documents()
        if not features:
            logger.warning("Phase 1.5: 未找到已提取的文档，跳过嵌入编码")
            state.phase = 3
            self.checkpoint.save(state)
            return

        logger.info(f"Phase 1.5: 准备编码 {len(features)} 个文档的段落嵌入")
        total_paragraphs = 0

        # 筛选需要编码的文档（有 MinHash = 有文本）
        embed_doc_ids = [
            f.doc_id for f in features if f.doc_minhash
        ]

        if not embed_doc_ids:
            logger.info("Phase 1.5: 没有需要编码的文档")
        else:
            # 统一使用主进程编码，避免每个 worker 重复加载 SBERT 模型（~480MB/次）
            # 对 GPU 场景，单进程编码可充分利用 CUDA 批量推理，无 GIL 问题；
            # 对 CPU 场景，PyTorch 矩阵运算自动释放 GIL，串行与并行速度接近。
            logger.info(
                f"Phase 1.5 单进程模式: 编码 {len(embed_doc_ids)} 个文档, "
                f"模型已加载 (设备: {engine.device})"
            )
            for doc_feat in features:
                if not doc_feat.doc_minhash:
                    continue
                try:
                    paragraphs = self.cache.load_all_paragraphs_text(
                        doc_feat.doc_id
                    )
                    if paragraphs:
                        count = engine.encode_document(
                            doc_feat.doc_id, paragraphs, self.cache
                        )
                        total_paragraphs += count
                    else:
                        logger.debug(
                            f"Phase 1.5: {doc_feat.filename} 无段落文本，跳过"
                        )
                except Exception as e:
                    logger.error(
                        f"嵌入编码失败 ({doc_feat.filename}): {e}",
                        exc_info=True,
                    )
                    continue

        state.phase = 3
        self.checkpoint.save(state)

        embed_time = (datetime.now() - embed_start).total_seconds()
        logger.info(
            f"Phase 1.5 完成: {total_paragraphs} 个段落已编码, "
            f"耗时 {embed_time:.2f}s"
        )

    def _phase3_analyze_single(self, doc_a_id: str, doc_b_id: str):
        """Phase 3: 分析单个文档对（流式加载文本）"""
        # 加载轻量特征
        doc_a = self.cache.load_document(doc_a_id)
        doc_b = self.cache.load_document(doc_b_id)

        if not doc_a or not doc_b:
            logger.error(f"文档未找到: {doc_a_id} / {doc_b_id}")
            return

        # 检查是否有可用文字（MinHash 为空 = 无文字可用）
        if not doc_a.doc_minhash or not doc_b.doc_minhash:
            if doc_a.is_scanned or doc_b.is_scanned:
                # 纯扫描版文档（OCR 无结果）：仅进行图片比对
                logger.info(
                    f"纯扫描文档对（无 OCR 文字）: "
                    f"{doc_a.filename} vs {doc_b.filename}, 仅图片比对"
                )
                self._phase3_image_only(doc_a, doc_b, doc_a_id, doc_b_id)
                return
            else:
                # 非扫描版文档缺少 MinHash = 数据异常
                logger.warning(
                    f"文档缺少 MinHash 签名: "
                    f"{doc_a.filename} / {doc_b.filename}, 跳过"
                )
                self.cache.mark_pair_processed(doc_a_id, doc_b_id)
                return
        # 有 MinHash 的文档（含 OCR 成功的扫描版）进入正常文本分析流程

        pair_id = "::".join(sorted([doc_a_id, doc_b_id]))

        # 激活文档的流式上下文
        self.streaming.activate_document(doc_a_id)
        self.streaming.activate_document(doc_b_id)

        try:
            # 执行两阶段段落匹配
            paragraph_matches = self.paragraph_matcher.match(
                doc_a, doc_b, self.cache
            )

            # 构建证据链
            evidence = self._build_evidence(doc_a, doc_b, paragraph_matches)

            # 构建分析结果
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

            # 风险评分
            result = self.scoring_engine._score_pair(result)

            # 存储结果
            self.cache.store_pairwise_result(result)
            self.cache.mark_pair_processed(doc_a_id, doc_b_id)

        finally:
            # 释放文档缓存
            self.streaming.release_document(doc_a_id)
            self.streaming.release_document(doc_b_id)

    def _build_evidence(
        self,
        doc_a: BidFeature,
        doc_b: BidFeature,
        paragraph_matches: List[Dict],
    ) -> EvidenceChain:
        """构建证据链"""
        # 文本证据
        text_evidence = self._build_text_evidence(
            doc_a, doc_b, paragraph_matches
        )

        # 元数据证据
        metadata_evidence = self._build_metadata_evidence(doc_a, doc_b)

        # 图片证据（增强版：四层检测）
        image_evidence = self._build_image_evidence(doc_a, doc_b)

        return EvidenceChain(
            text_evidence=text_evidence,
            metadata_evidence=metadata_evidence,
            image_evidence=image_evidence,
        )

    @staticmethod
    def _build_metadata_evidence(
        doc_a: BidFeature, doc_b: BidFeature
    ) -> MetadataEvidence:
        """构建元数据证据（委托 evidence_builder 复用）"""
        return build_metadata_evidence(doc_a, doc_b)

    def _phase3_image_only(
        self, doc_a: BidFeature, doc_b: BidFeature,
        doc_a_id: str, doc_b_id: str
    ):
        """Phase 3 纯图片比对路径（无文字可用的扫描版文档对）

        当两个文档都没有 OCR 文字可用时，仅进行图片哈希比对
        和元数据匹配，文本证据留空。
        """
        pair_id = "::".join(sorted([doc_a_id, doc_b_id]))

        # 图片证据（四层检测）
        image_evidence = self._build_image_evidence(doc_a, doc_b)

        # 元数据证据
        metadata_evidence = self._build_metadata_evidence(doc_a, doc_b)

        evidence = EvidenceChain(
            text_evidence=TextEvidence(),
            metadata_evidence=metadata_evidence,
            image_evidence=image_evidence,
        )

        result = PairwiseResult(
            pair_id=pair_id,
            doc_a_id=doc_a_id,
            doc_b_id=doc_b_id,
            similarity_scores={
                'text_local': 0.0,
                'metadata_match': len(metadata_evidence.matched_fields),
                'image_common': image_evidence.common_image_count,
            },
            evidence=evidence,
        )

        # 风险评分
        result = self.scoring_engine._score_pair(result)
        self.cache.store_pairwise_result(result)
        self.cache.mark_pair_processed(doc_a_id, doc_b_id)

    def _build_image_evidence(
        self, doc_a: BidFeature, doc_b: BidFeature
    ) -> ImageEvidence:
        """构建增强图片证据 — 四层检测（哈希 + OCR + 错字 + 文字相同）"""
        evidence = ImageEvidence()

        # 保留原有的精确哈希匹配
        hashes_a = doc_a.image_hashes
        hashes_b = doc_b.image_hashes
        common_exact = list(set(hashes_a) & set(hashes_b))
        evidence.common_image_count = len(common_exact)
        evidence.common_image_hashes = common_exact

        # 加载 OCR 结果
        ocr_a = self.cache.load_image_ocr_results(doc_a.doc_id)
        ocr_b = self.cache.load_image_ocr_results(doc_b.doc_id)

        if ocr_a:
            evidence.ocr_results_a = ocr_a
        if ocr_b:
            evidence.ocr_results_b = ocr_b

        # 使用 ImageMatcher 执行四层检测
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

        match_result = self.image_matcher.analyze(
            hashes_a=hashes_a,
            hashes_b=hashes_b,
            ocr_results_a=ocr_objects_a if ocr_objects_a else None,
            ocr_results_b=ocr_objects_b if ocr_objects_b else None,
        )

        # 填充增强字段
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

        # 填充逐对图片详情（供报告逐对展示检测层标签）
        for v in match_result.image_verdicts:
            evidence.matched_image_pairs.append({
                'phash_dist': v.phash_dist,
                'dhash_dist': v.dhash_dist,
                'orb_match_ratio': round(v.orb_match_ratio, 3),
                'histogram_correlation': round(v.histogram_correlation, 3),
                'confidence': round(v.confidence, 3),
                'reasons': v.reasons,
                'l1_pass': v.l1_pass,
                'l2_pass': v.l2_pass,
                'l3_pass': v.l3_pass,
                'source_a': v.sig_a.source_id,
                'source_b': v.sig_b.source_id,
            })
        evidence.matched_text_pairs = match_result.text_matches
        evidence.ps_detail_list = match_result.ps_details

        return evidence

    def _build_text_evidence(
        self,
        doc_a: BidFeature,
        doc_b: BidFeature,
        paragraph_matches: List[Dict],
    ) -> TextEvidence:
        """构建文本证据（委托 evidence_builder 复用）"""
        return build_text_evidence(
            doc_a, doc_b, paragraph_matches, self.config, compute_highlight=True
        )

    def _detect_clone_blocks(self, paragraph_matches: List[Dict]) -> List[Dict]:
        """检测连续克隆块（委托 evidence_builder 复用）"""
        from pipeline.evidence_builder import _detect_clone_blocks as _dcb
        return _dcb(paragraph_matches, self.config)

    @staticmethod
    def _empty_report() -> GlobalReport:
        """生成空报告"""
        return GlobalReport(
            report_id="empty",
            generated_at=datetime.now().isoformat(),
            total_files=0,
            total_pairs=0,
            candidate_pairs=0,
            suspicious_pairs=0,
            high_risk_pairs=0,
        )

    def _compute_input_hash(self, input_dir: str) -> str:
        """计算输入文件夹内容的哈希值

        通过扫描文件夹中所有PDF文件的文件名、大小和修改时间，
        生成一个哈希值，用于检测文件内容是否发生变化。

        Args:
            input_dir: 输入文件夹路径

        Returns:
            32位MD5哈希字符串
        """
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
