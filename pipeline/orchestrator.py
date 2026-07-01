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
from typing import List, Dict, Optional

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
from image_analysis.image_matcher import ImageMatcher, ImageMatchResult
from scoring import RiskScoringEngine
from report import ReportGenerator

logger = logging.getLogger(__name__)


class BidDetectionOrchestrator:
    """5 阶段流式管道编排器"""

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

        # 图片分析引擎（OCR + 四层检测）
        self.ocr_engine = ImageOCREngine(
            use_gpu=config.USE_GPU,
        )
        self.image_matcher = ImageMatcher()

        # 跨阶段缓存：避免重复从 SQLite 加载文档特征
        self._all_features_cache = None

        logger.info("流式管道编排器已初始化")

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

                extract_start = datetime.now()
                processed_count = 0

                for file_path in unprocessed:
                    try:
                        self._phase1_extract_single(file_path)
                        state.processed_files.add(file_path)
                        processed_count += 1

                        if processed_count % 10 == 0:
                            logger.info(
                                f"Phase 1 进度: {processed_count}/{len(unprocessed)}"
                            )
                            # 阶段性保存检查点
                            self.checkpoint.save(state)

                    except Exception as e:
                        logger.error(f"提取失败 ({file_path}): {e}", exc_info=True)
                        # 不中断，继续处理其他文件
                        continue

                state.phase = 2
                self.checkpoint.save(state)

                extract_time = (datetime.now() - extract_start).total_seconds()
                logger.info(
                    f"Phase 1 完成: {processed_count} 个文档, "
                    f"耗时 {extract_time:.2f}s"
                )

            # === Phase 1.5: EMBED (全局 SBERT 嵌入编码) ===
            if state.phase < 3:
                self._phase_embed(state)

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

                    for idx, (doc_a_id, doc_b_id) in enumerate(pending_pairs, 1):
                        try:
                            self._phase3_analyze_single(doc_a_id, doc_b_id)

                            pair_id = "::".join(sorted([doc_a_id, doc_b_id]))
                            completed_ids.add(pair_id)
                            state.completed_pairs = len(completed_ids)

                            # 进度日志
                            if idx % 10 == 0 or idx == total_pending:
                                logger.info(
                                    f"Phase 3 进度: {idx}/{total_pending} "
                                    f"({state.completed_pairs}/{state.total_pairs})"
                                )

                            # 增量检查点 + 批量提交 SQLite 写入
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

                    # 最终批量提交
                    self.cache.conn.commit()

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
                logger.info(
                    f"Phase 1 完成 (扫描版): {filename} "
                    f"({page_count} 页, {len(all_page_hashes)} 页面哈希)"
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
                logger.info(
                    f"Phase 1 完成 (纯扫描版): {filename} "
                    f"({page_count} 页, {len(all_page_hashes)} 页面哈希)"
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
            logger.info(
                f"Phase 1 完成: {filename} "
                f"({page_count} 页, {len(chunks)} 块, "
                f"{feature.text_length} 字符, "
                f"{len(feature.image_hashes)} 图片哈希)"
            )

    # ============================================================
    # Phase 3: 单对分析
    # ============================================================

    def _phase_embed(self, state) -> None:
        """Phase 1.5: 全局 SBERT 嵌入编码

        一次性编码所有文档的所有段落嵌入并持久化到 SQLite。
        后续 Phase 3 分析只需查表加载嵌入 + 点积，不再调用 SBERT 模型。
        """
        logger.info("Phase 1.5: 全局 SBERT 嵌入编码...")
        embed_start = datetime.now()

        engine = EmbeddingEngine(self.config)
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

        for doc_feat in features:
            if doc_feat.is_scanned:
                continue
            try:
                paragraphs = self.cache.load_all_paragraphs_text(doc_feat.doc_id)
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
                    f"嵌入编码失败 ({doc_feat.filename}): {e}", exc_info=True
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

        if doc_a.is_scanned or doc_b.is_scanned:
            logger.info(
                f"扫描版文档对: {doc_a.filename} vs {doc_b.filename}, "
                f"跳过文本分析（图片比对功能后续版本实现）"
            )
            self.cache.mark_pair_processed(doc_a_id, doc_b_id)
            return

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
        metadata_evidence = MetadataEvidence()
        fields_to_check = ['author', 'creator', 'producer', 'software_fingerprint']
        for field in fields_to_check:
            val_a = getattr(doc_a.metadata, field, '').lower().strip()
            val_b = getattr(doc_b.metadata, field, '').lower().strip()
            if val_a and val_b and val_a == val_b:
                metadata_evidence.matched_fields.append(field)
                metadata_evidence.matched_values[field] = val_a
        if doc_a.metadata.time_bucket and doc_b.metadata.time_bucket:
            metadata_evidence.same_time_bucket = (
                doc_a.metadata.time_bucket == doc_b.metadata.time_bucket
            )

        # 图片证据（增强版：四层检测）
        image_evidence = self._build_image_evidence(doc_a, doc_b)

        return EvidenceChain(
            text_evidence=text_evidence,
            metadata_evidence=metadata_evidence,
            image_evidence=image_evidence,
        )

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
            ) for r in ocr_a
        ]
        ocr_objects_b = [
            OCRResult(
                text=r['ocr_text'],
                words=r['ocr_words'],
                bboxes=r['bboxes'],
                confidence=r['confidence'],
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

        return evidence

    def _build_text_evidence(
        self,
        doc_a: BidFeature,
        doc_b: BidFeature,
        paragraph_matches: List[Dict],
    ) -> TextEvidence:
        """构建文本证据（自包含实现，不依赖 analyzer 内部）"""
        import math

        evidence = TextEvidence()
        evidence.paragraph_matches = paragraph_matches

        if paragraph_matches:
            # === 计算混合评分（加权和，各分量均在 [0,1]，总分 ≤1.0） ===
            similarities = [m['similarity'] for m in paragraph_matches]
            max_sim = max(similarities) if similarities else 0.0

            top_k = min(self.config.SCORE_TOP_K, len(similarities))
            top_k_similarities = sorted(similarities, reverse=True)[:top_k]
            top_k_sim = sum(top_k_similarities) / top_k if top_k_similarities else 0.0

            weighted_sum = sum(s * s for s in similarities)
            weighted_mean = weighted_sum / sum(similarities) if sum(similarities) > 0 else 0.0
            mean_sim = sum(similarities) / len(similarities) if similarities else 0.0

            # 质量分数：加权和，权重归一化确保 ≤ 1.0
            # 使用 max/mean/weighted_mean 三个维度
            quality_score = (
                0.50 * max_sim +       # 最强匹配信号
                0.35 * top_k_sim +     # Top-K 平均
                0.15 * weighted_mean   # 加权平均（高相似度权重大）
            )
            # quality_score ∈ [0, 1] (weights sum to 1.0)

            # 覆盖率分数：使用较平缓的指数衰减
            covered_a = len(set(m['paragraph_a_index'] for m in paragraph_matches))
            covered_b = len(set(m['paragraph_b_index'] for m in paragraph_matches))
            # 估算总段落数（保守估计）
            estimated_total = max(1, covered_a + covered_b)
            coverage_ratio = (covered_a + covered_b) / (estimated_total * 2) if estimated_total > 0 else 0
            coverage_score = 1.0 - math.exp(-4 * coverage_ratio) if coverage_ratio > 0 else 0.0

            # 一致性分数：连续匹配加分
            sorted_by_a = sorted(paragraph_matches, key=lambda x: x['paragraph_a_index'])
            consecutive = sum(
                1 for k in range(1, len(sorted_by_a))
                if (sorted_by_a[k]['paragraph_a_index'] - sorted_by_a[k-1]['paragraph_a_index'] == 1 and
                    sorted_by_a[k]['paragraph_b_index'] - sorted_by_a[k-1]['paragraph_b_index'] == 1)
            )
            if consecutive >= 3:
                consistency_score = min(1.0, 0.5 + consecutive * 0.01)
            else:
                consistency_score = 0.5  # 无连续匹配时的基线

            # 加权和：质量 60% + 覆盖率 25% + 一致性 15%
            evidence.local_similarity = min(
                1.0,
                0.60 * quality_score + 0.25 * coverage_score + 0.15 * consistency_score
            )

            # === 检测连续克隆块 ===
            clone_blocks = self._detect_clone_blocks(paragraph_matches)
            evidence.continuous_clone_blocks = clone_blocks

            # 更新克隆标记
            clone_index = {}
            for block in clone_blocks:
                for pair in block['pairs']:
                    key = (pair['a_index'], pair['b_index'])
                    clone_index[key] = {
                        'is_clone': True,
                        'group_id': block['group_id']
                    }
            for match in paragraph_matches:
                key = (match['paragraph_a_index'], match['paragraph_b_index'])
                if key in clone_index:
                    match['is_continuous_clone'] = True
                    match['continuous_clone_group_id'] = clone_index[key]['group_id']

            evidence.detection_summary = {
                'sbert_match_count': len(paragraph_matches),
                'continuous_clone_block_count': len(clone_blocks),
            }

            evidence.common_paragraphs = [
                m.get('paragraph_a', '')[:200]
                for m in paragraph_matches[:5]
            ]

            # === 计算文本差异高亮（确保所有相似内容完整报告） ===
            for match in paragraph_matches:
                text_a = match.get('paragraph_a', '')
                text_b = match.get('paragraph_b', '')
                if text_a and text_b:
                    highlighted_a, highlighted_b, common_parts = self._compute_text_diff(text_a, text_b)
                    match['highlighted_text_a'] = highlighted_a
                    match['highlighted_text_b'] = highlighted_b
                    match['common_parts'] = common_parts

        return evidence

    def _detect_clone_blocks(self, paragraph_matches: List[Dict]) -> List[Dict]:
        """检测连续克隆块"""
        min_length = self.config.CLONE_BLOCK_MIN_LENGTH
        max_gap = self.config.CLONE_BLOCK_MAX_GAP

        if len(paragraph_matches) < min_length:
            return []

        matches_sorted = sorted(
            paragraph_matches,
            key=lambda x: (x['paragraph_a_index'], x['paragraph_b_index'])
        )

        blocks = []
        current_block = []
        group_id = 0

        for match in matches_sorted:
            if not current_block:
                current_block.append(match)
            else:
                last = current_block[-1]
                a_gap = match['paragraph_a_index'] - last['paragraph_a_index'] - 1
                b_gap = match['paragraph_b_index'] - last['paragraph_b_index'] - 1

                if a_gap <= max_gap and b_gap <= max_gap:
                    current_block.append(match)
                else:
                    if len(current_block) >= min_length:
                        blocks.append({
                            'group_id': f'clone_block_{group_id}',
                            'pairs': [
                                {'a_index': m['paragraph_a_index'],
                                 'b_index': m['paragraph_b_index']}
                                for m in current_block
                            ],
                            'similarity': sum(m['similarity'] for m in current_block) / len(current_block),
                            'length': len(current_block),
                        })
                        group_id += 1
                    current_block = [match]

        if len(current_block) >= min_length:
            blocks.append({
                'group_id': f'clone_block_{group_id}',
                'pairs': [
                    {'a_index': m['paragraph_a_index'],
                     'b_index': m['paragraph_b_index']}
                    for m in current_block
                ],
                'similarity': sum(m['similarity'] for m in current_block) / len(current_block),
                'length': len(current_block),
            })

        return blocks

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

    def _compute_text_diff(self, text_a: str, text_b: str) -> tuple:
        """计算两个文本的差异并生成高亮标记

        使用 difflib.SequenceMatcher 找最长公共子串，
        然后用【】标记相同部分。

        Returns:
            (highlighted_a, highlighted_b, common_parts):
            - highlighted_a: 文本A的高亮版本（用【】标记相同部分）
            - highlighted_b: 文本B的高亮版本（用【】标记相同部分）
            - common_parts: 共同文本片段列表
        """
        from difflib import SequenceMatcher as SMatch

        clean_a = text_a.strip()
        clean_b = text_b.strip()

        sm = SMatch(None, clean_a, clean_b)
        matching_blocks = sm.get_matching_blocks()

        significant_blocks = [b for b in matching_blocks if b.size >= 10]

        common_parts = []
        for block in significant_blocks:
            if block.size >= 10:
                common_text = clean_a[block.a:block.a + block.size]
                common_parts.append(common_text.strip())

        common_parts.sort(key=len, reverse=True)
        common_parts = common_parts[:20]

        highlighted_a = self._highlight_text_with_blocks(clean_a, significant_blocks, 'a')
        highlighted_b = self._highlight_text_with_blocks(clean_b, significant_blocks, 'b')

        # 报告需要完整文本，设置足够大的截断限制（50000字符基本等于不截断）
        if len(highlighted_a) > 50000:
            highlighted_a = highlighted_a[:50000] + "\n... [文本过长，已截断]"
        if len(highlighted_b) > 50000:
            highlighted_b = highlighted_b[:50000] + "\n... [文本过长，已截断]"

        return highlighted_a, highlighted_b, common_parts

    def _highlight_text_with_blocks(self, text: str, blocks: list, which: str) -> str:
        """用匹配块标记文本中的相同部分"""
        if not blocks:
            return text

        if which == 'a':
            blocks = sorted(blocks, key=lambda b: b.a)
        else:
            blocks = sorted(blocks, key=lambda b: b.b)

        result_parts = []
        last_end = 0

        for block in blocks:
            if block.size < 10:
                continue

            if which == 'a':
                start = block.a
                end = block.a + block.size
            else:
                start = block.b
                end = block.b + block.size

            if start > last_end:
                result_parts.append(text[last_end:start])

            matched_text = text[start:end]
            if len(matched_text.strip()) > 0:
                result_parts.append(f"【{matched_text}】")

            last_end = end

        if last_end < len(text):
            result_parts.append(text[last_end:])

        result = ''.join(result_parts)

        import re
        result = re.sub(r'】\s*【', '', result)

        return result

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
