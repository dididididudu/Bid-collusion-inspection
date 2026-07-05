"""
PyMuPDF (fitz) PDF 提取器 — 高速 PDF 文本与图片解析

相比 pdfplumber，PyMuPDF 在提取上有 5-10x 的速度优势:
- 使用 C 编写的底层渲染引擎
- 逐页流式生成，避免一次性加载完整 PDF 到内存
- 支持断点续传（从指定页码开始提取）
- 页级图片渲染 + 感知哈希（支持扫描版 PDF 的图片比对）
- 嵌入图片提取（文本 PDF 中的 logo/印章/图表）

回退策略:
- 如果 PyMuPDF 不可用，自动回退到 pdfplumber
- 如果单个页面提取失败，跳过该页继续处理
"""

import os
import re
import io
import logging
import hashlib
from typing import List, Generator, Tuple, Optional
from collections import Counter

import jieba
import numpy as np
from PIL import Image
import imagehash

try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

from data_structures import MetadataFeature, ChunkResult
from config import DetectionConfig
from extraction.base import BasePDFExtractor

logger = logging.getLogger(__name__)


class PyMuPDFExtractor(BasePDFExtractor):
    """基于 PyMuPDF (fitz) 的高速 PDF 提取器"""

    # 类级编译正则表达式（避免每次调用时重新编译）
    _RE_CRLF = re.compile(r'\r\n')
    _RE_MULTI_NEWLINE = re.compile(r'\n{4,}')
    _RE_VERSION_STRIP = re.compile(r'\d+\.\d+[\d.]*')
    _RE_PDF_DATE = re.compile(r"D:(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})")

    def __init__(self, config: DetectionConfig):
        if not FITZ_AVAILABLE:
            raise ImportError(
                "PyMuPDF (fitz) 未安装，请运行: pip install PyMuPDF"
            )
        self.config = config
        self.stopwords = self._load_stopwords()
        # 预创建 MinHash 哈希函数（64 个）
        self._minhash_funcs = self._precreate_minhash_functions()
        logger.info("PyMuPDF 提取器已初始化")

    def _load_stopwords(self) -> set:
        """加载停用词表"""
        if self.config.STOPWORDS_PATH and os.path.exists(self.config.STOPWORDS_PATH):
            with open(self.config.STOPWORDS_PATH, 'r', encoding='utf-8') as f:
                return set(line.strip() for line in f)
        return {'的', '了', '和', '是', '在', '有', '与', '为', '等', '及'}

    def _precreate_minhash_functions(self) -> list:
        """预创建 MinHash 哈希函数

        使用 hashlib.md5 确保跨进程确定性（Python hash() 因 PYTHONHASHSEED 随机化）。
        """
        import hashlib
        num_hash = self.config.MINHASH_NUM_HASHES_PARAGRAPH

        def make_hash_func(seed):
            def hash_func(s):
                h = hashlib.md5(f'{s}:{seed}'.encode('utf-8')).hexdigest()
                return int(h, 16) % (2 ** 32)
            return hash_func

        return [make_hash_func(i) for i in range(num_hash)]

    # ============================================================
    # 公共接口
    # ============================================================

    def get_page_count(self, file_path: str) -> int:
        """快速获取 PDF 页数"""
        doc = fitz.open(file_path)
        page_count = doc.page_count
        doc.close()
        return page_count

    def extract_metadata(self, file_path: str) -> Tuple[MetadataFeature, int, bool]:
        """Phase 0: 提取 PDF 元数据（不解析文本）"""
        doc = fitz.open(file_path)
        try:
            meta = doc.metadata or {}

            author = meta.get('author', '').strip()
            creator = meta.get('creator', '').strip()
            producer = meta.get('producer', '').strip()

            # 解析创建/修改时间
            created_time = self._parse_pdf_date(meta.get('creationDate', ''))
            modified_time = self._parse_pdf_date(meta.get('modDate', ''))

            # 生成软件指纹
            software_fp = self._generate_software_fingerprint(creator, producer)

            # 时间分桶
            time_bucket = ""
            if created_time:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(created_time)
                    time_bucket = dt.strftime(self.config.TIME_BUCKET_FORMAT)
                except Exception:
                    pass

            metadata = MetadataFeature(
                author=author,
                creator=creator,
                producer=producer,
                created_time=created_time,
                modified_time=modified_time,
                software_fingerprint=software_fp,
                time_bucket=time_bucket,
            )

            page_count = doc.page_count

            # 判断是否为扫描版（前 3 页文本量很少则判定为扫描版）
            is_scanned = False
            if page_count > 0:
                sample_text = ""
                for i in range(min(3, page_count)):
                    try:
                        sample_text += doc[i].get_text("text")
                    except Exception:
                        pass
                is_scanned = len(sample_text.strip()) < 100

            return metadata, page_count, is_scanned

        finally:
            doc.close()

    def extract_chunks(
        self,
        file_path: str,
        chunk_size: int = 50,
        start_page: int = 0,
    ) -> Generator[ChunkResult, None, None]:
        """Phase 1: 按块流式提取文本内容

        每次生成一个 ChunkResult，调用方一次性处理完后释放。
        支持从 start_page 开始（用于断点续传）。
        """
        doc_id = self._generate_doc_id(file_path)
        doc = fitz.open(file_path)
        page_count = doc.page_count

        try:
            # 按 chunk_size 分批处理页面
            for chunk_start in range(start_page, page_count, chunk_size):
                chunk_end = min(chunk_start + chunk_size, page_count)
                chunk_index = chunk_start // chunk_size

                logger.debug(
                    f"提取块 {chunk_index}: 页 {chunk_start}-{chunk_end-1} "
                    f"({file_path})"
                )

                # 提取该块的文本
                all_text_parts = []
                for page_num in range(chunk_start, chunk_end):
                    try:
                        page = doc[page_num]
                        text = page.get_text("text")
                        if text:
                            all_text_parts.append(text)
                    except Exception as e:
                        logger.warning(
                            f"页面 {page_num} 提取失败 ({file_path}): {e}"
                        )
                        continue

                chunk_text = "\n".join(all_text_parts)

                # 分词和段落分割
                paragraphs = self._split_paragraphs(chunk_text)

                # 一次性对整块文本分词，simhash 和 minhash 共享分词结果
                chunk_tokens = []
                if chunk_text:
                    chunk_tokens = [
                        w for w in jieba.cut(chunk_text)
                        if w not in self.stopwords and len(w) > 1
                    ]

                # 计算该块的 SimHash（使用预分词结果）
                simhash = self._compute_simhash_from_tokens(chunk_tokens) if chunk_tokens else ""

                # 预计算所有唯一词的 MinHash 值（跨段落共享）
                word_hash_cache = {}
                if chunk_tokens:
                    unique_words = set(chunk_tokens)
                    for w in unique_words:
                        word_hash_cache[w] = [
                            hash_func(w) for hash_func in self._minhash_funcs
                        ]

                # 计算段落的 MinHash（使用缓存词哈希）
                paragraph_hashes = []
                for para in paragraphs:
                    para_words = [
                        w for w in jieba.cut(para)
                        if w not in self.stopwords and len(w) > 1
                    ]
                    para_hash = self._compute_minhash_cached(para_words, word_hash_cache)
                    paragraph_hashes.append(para_hash)

                # 提取报价
                quotes = self._extract_quotes(chunk_text)

                # 提取图片哈希（仅在 ENABLE_PAGE_IMAGE_HASHES 启用时）
                # 纯文本 PDF 可禁用以大幅加速（跳过高成本的全页渲染和绘图检测）
                if getattr(self.config, 'ENABLE_PAGE_IMAGE_HASHES', True):
                    embedded_hashes = self._extract_embedded_images(
                        doc, chunk_start, chunk_end
                    )
                    page_image_hashes = self._extract_page_image_hashes(
                        doc, chunk_start, chunk_end
                    )
                    all_image_hashes = list(set(embedded_hashes + page_image_hashes))
                else:
                    all_image_hashes = []

                yield ChunkResult(
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    start_page=chunk_start,
                    end_page=chunk_end - 1,
                    text=chunk_text,
                    paragraphs=paragraphs,
                    paragraph_hashes=paragraph_hashes,
                    simhash=simhash,
                    quotes=quotes,
                    image_hashes=all_image_hashes,
                )

        finally:
            doc.close()

    # ============================================================
    # 图片提取方法
    # ============================================================

    def _merge_overlapping_bboxes(
        self, bboxes: List[Tuple[float, float, float, float]], gap: float = 10
    ) -> List[Tuple[float, float, float, float]]:
        """合并相邻/重叠的 bbox 矩形

        Word→PDF 会将一张大图切分为多个相邻的图块（tile），
        合并后还原为完整的大图区域，统一渲染和哈希。

        Args:
            bboxes: [(x0, y0, x1, y1), ...] 坐标列表
            gap: 合并容差（像素），间距小于此值的矩形被合并

        Returns:
            合并后的 bbox 列表
        """
        if not bboxes:
            return []

        merged = [list(b) for b in bboxes]
        changed = True
        while changed:
            changed = False
            new_list = []
            used = [False] * len(merged)
            for i in range(len(merged)):
                if used[i]:
                    continue
                cur = merged[i][:]
                for j in range(i + 1, len(merged)):
                    if used[j]:
                        continue
                    o = merged[j]
                    # 检查是否在 gap 范围内相交
                    if (cur[0] - gap <= o[2] and cur[2] + gap >= o[0] and
                        cur[1] - gap <= o[3] and cur[3] + gap >= o[1]):
                        cur[0] = min(cur[0], o[0])
                        cur[1] = min(cur[1], o[1])
                        cur[2] = max(cur[2], o[2])
                        cur[3] = max(cur[3], o[3])
                        used[j] = True
                        changed = True
                new_list.append(tuple(cur))
                used[i] = True
            merged = new_list
        return merged

    def _render_region(self, page, bbox, page_w, page_h, min_size, zoom=200/72.0):
        """渲染页面指定区域并返回 pHash，失败返回 None"""
        try:
            x0, y0, x1, y1 = bbox
            x0 = max(0, x0 - 2)
            y0 = max(0, y0 - 2)
            x1 = min(page_w, x1 + 2)
            y1 = min(page_h, y1 + 2)
            if x1 - x0 < min_size or y1 - y0 < min_size:
                return None
            import fitz
            mat = fitz.Matrix(zoom, zoom)
            clip = fitz.Rect(x0, y0, x1, y1)
            pix = page.get_pixmap(matrix=mat, clip=clip)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            if img.size[0] > 1000 or img.size[1] > 1000:
                img.thumbnail((512, 512), Image.LANCZOS)
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            return imagehash.phash(img)
        except Exception as e:
            logger.debug(f"区域渲染失败 ({bbox}): {e}")
            return None

    def _extract_embedded_images(
        self, doc, start_page: int, end_page: int
    ) -> List[str]:
        """提取嵌入图片的感知哈希（处理 Word→PDF 图块合并）

        三路策略：
        1. cluster_drawings() — 检测矢量绘图集群（Word→PDF 切碎的大图）
        2. get_image_info() + bbox 合并 — 检测嵌入图片集群
        3. extract_image() — 孤立的嵌入图片（兜底）

        Returns:
            感知哈希字符串列表
        """
        import fitz
        hashes = []
        min_size = getattr(self.config, 'IMAGE_MIN_SIZE', 50)
        zoom = 200 / 72.0

        for page_num in range(start_page, min(end_page, doc.page_count)):
            try:
                page = doc[page_num]
                page_w = page.rect.width
                page_h = page.rect.height
                rendered_regions = []  # 已渲染的区域，避免重复

                # === 策略1: cluster_drawings 检测矢量绘图集群 ===
                try:
                    clusters = page.cluster_drawings(
                        x_tolerance=10, y_tolerance=10, final_filter=False
                    )
                    for cl in clusters:
                        bbox = (cl.x0, cl.y0, cl.x1, cl.y1)
                        w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
                        if w < 50 or h < 50:
                            continue
                        ph = self._render_region(page, bbox, page_w, page_h, min_size, zoom)
                        if ph is not None:
                            hashes.append(f"p{ph}")
                            rendered_regions.append(bbox)
                except Exception:
                    pass

                # === 策略2: get_image_info + bbox 合并 ===
                image_info = page.get_image_info()
                all_bboxes = []
                for info in image_info:
                    bbox = info.get('bbox')
                    if bbox and len(bbox) == 4:
                        x0, y0, x1, y1 = bbox
                        if (x1-x0) >= min_size and (y1-y0) >= min_size \
                           and x0 > -10 and y0 > -10 \
                           and x1 < page_w + 10 and y1 < page_h + 10:
                            all_bboxes.append(bbox)

                if all_bboxes:
                    # 过滤掉已被 cluster_drawings 覆盖的区域
                    remaining = []
                    for b in all_bboxes:
                        covered = False
                        for rb in rendered_regions:
                            ix = max(b[0], rb[0]); iy = max(b[1], rb[1])
                            ix2 = min(b[2], rb[2]); iy2 = min(b[3], rb[3])
                            if ix < ix2 and iy < iy2:
                                overlap = (ix2-ix)*(iy2-iy)
                                area_b = (b[2]-b[0])*(b[3]-b[1])
                                if area_b > 0 and overlap/area_b > 0.5:
                                    covered = True
                                    break
                        if not covered:
                            remaining.append(b)

                    merged = self._merge_overlapping_bboxes(remaining, gap=8)
                    for bbox in merged:
                        ph = self._render_region(page, bbox, page_w, page_h, min_size, zoom)
                        if ph is not None:
                            hashes.append(f"p{ph}")
                            rendered_regions.append(bbox)

                # === 策略3: 兜底 — 孤立图片直接用 extract_image ===
                image_list = page.get_images(full=True)
                if image_list:
                    for img_info in image_list:
                        try:
                            xref = img_info[0]
                            # 获取这个图片的 bbox
                            img_name = img_info[7] if len(img_info) > 7 else None
                            img_bbox = None
                            if img_name:
                                try:
                                    bobj = page.get_image_bbox(img_name)
                                    if bobj:
                                        img_bbox = (bobj.x0, bobj.y0, bobj.x1, bobj.y1)
                                except Exception:
                                    pass
                            # 检查是否已被覆盖
                            if img_bbox:
                                covered = False
                                for rb in rendered_regions:
                                    ix = max(img_bbox[0], rb[0]); iy = max(img_bbox[1], rb[1])
                                    ix2 = min(img_bbox[2], rb[2]); iy2 = min(img_bbox[3], rb[3])
                                    if ix < ix2 and iy < iy2:
                                        overlap = (ix2-ix)*(iy2-iy)
                                        area = (img_bbox[2]-img_bbox[0])*(img_bbox[3]-img_bbox[1])
                                        if area > 0 and overlap/area > 0.5:
                                            covered = True
                                            break
                                if covered:
                                    continue

                            # 过滤和提取
                            w, h = img_info[2], img_info[3]
                            if w < min_size or h < min_size:
                                continue
                            base = doc.extract_image(xref)
                            if not base or not base.get("image"):
                                continue
                            img_bytes = base["image"]
                            if len(img_bytes) < 1024:
                                continue
                            img = Image.open(io.BytesIO(img_bytes))
                            if img.size[0] > 1000 or img.size[1] > 1000:
                                img.thumbnail((512, 512), Image.LANCZOS)
                            if img.mode not in ('RGB', 'L'):
                                img = img.convert('RGB')
                            ph = imagehash.phash(img)
                            hashes.append(f"p{ph}")
                        except Exception:
                            continue

            except Exception as e:
                logger.debug(f"页面 {page_num} 图片提取失败: {e}")
                continue

        return hashes

    def _extract_page_image_hashes(
        self, doc, start_page: int, end_page: int
    ) -> List[str]:
        """提取页级图片的感知哈希（用于扫描版 PDF 的页面对比）

        将每个页面渲染为图像，然后计算感知哈希。
        使用采样策略：每 2 页取 1 页（对扫描版 PDF 足够检测相似度）。
        渲染分辨率：200 DPI 缩放到 256x256 缩略图（快速哈希）。

        Args:
            doc: fitz.Document 对象
            start_page: 起始页码（0-based）
            end_page: 结束页码（0-based，不含）

        Returns:
            感知哈希字符串列表（格式: "page_{n}:{hash}"）
        """
        hashes = []
        # 采样策略：每 2 页取 1 页，减少计算量
        sample_step = 2

        for page_num in range(start_page, min(end_page, doc.page_count), sample_step):
            try:
                page = doc[page_num]
                # 渲染页面为图像（200 DPI，缩放到 256 像素宽）
                pix = page.get_pixmap(dpi=150)
                if pix is None:
                    continue

                # 转换为 PIL Image
                img_data = pix.tobytes("rgb") if pix.n >= 3 else pix.tobytes("gray")
                mode = "RGB" if pix.n >= 3 else "L"
                img = Image.frombytes(mode, (pix.width, pix.height), img_data)

                # 缩放到统一尺寸（256x256 左右，快速哈希）
                img.thumbnail((256, 256), Image.LANCZOS)

                # 计算多种哈希以提高鲁棒性
                phash = imagehash.phash(img)
                dhash = imagehash.dhash(img)

                # 组合哈希：同时使用 pHash 和 dHash
                hashes.append(f"page_{page_num}:p{phash}")
                hashes.append(f"page_{page_num}:d{dhash}")

            except Exception as e:
                logger.debug(f"页面 {page_num} 渲染失败: {e}")
                continue

        return hashes

    def extract_all_page_hashes(
        self, file_path: str, sample_step: int = 2
    ) -> List[str]:
        """提取整个文档所有页面的图片哈希（用于扫描版 PDF）

        独立方法，在 Phase 1 中专门调用。返回完整的页面哈希列表。
        用于后续文档间页级相似度比对。

        Args:
            file_path: PDF 文件路径
            sample_step: 采样步长（每隔 N 页取 1 页）

        Returns:
            ["page_0:p{hash1}", "page_0:d{hash1}", "page_2:p{hash2}", ...]
        """
        doc = fitz.open(file_path)
        try:
            all_hashes = []
            for page_num in range(0, doc.page_count, sample_step):
                try:
                    page = doc[page_num]
                    pix = page.get_pixmap(dpi=120)
                    if pix is None:
                        continue

                    img_data = pix.tobytes("rgb") if pix.n >= 3 else pix.tobytes("gray")
                    mode = "RGB" if pix.n >= 3 else "L"
                    img = Image.frombytes(mode, (pix.width, pix.height), img_data)
                    img.thumbnail((256, 256), Image.LANCZOS)

                    phash = imagehash.phash(img)
                    dhash = imagehash.dhash(img)
                    all_hashes.append(f"page_{page_num}:p{phash}")
                    all_hashes.append(f"page_{page_num}:d{dhash}")

                except Exception as e:
                    logger.debug(f"页面 {page_num} 渲染失败: {e}")
                    continue

            return all_hashes
        finally:
            doc.close()

    # ============================================================
    # 文本处理 (复用原有逻辑，独立于 extractor.py)
    # ============================================================

    def _split_paragraphs(self, text: str) -> List[str]:
        """句子级切分 - 用句子边界替代段落边界，实现位置无关匹配

        改进：以。！？；：等为分隔符，最小长度15字符，最大500字符。
        相同内容无论处于PDF何处，都会被切分为相似的句子片段。
        """
        if not text:
            return []

        text = self._RE_CRLF.sub('\n', text)
        text = self._RE_MULTI_NEWLINE.sub('\n\n\n', text)

        # 第1步：PDF断行合并
        lines = text.split('\n')
        merged_lines = []
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                merged_lines.append('')
                continue
            if merged_lines and merged_lines[-1] and merged_lines[-1] != '':
                prev = merged_lines[-1]
                if (re.search(r'[一-鿿　-〿＀-￯]', prev[-1]) and
                        re.search(r'^[一-鿿]', line_stripped)):
                    merged_lines[-1] = prev + line_stripped
                    continue
                if (prev[-1] not in '。！？；：》」』)"]' and
                        len(prev) > 10 and len(line_stripped) > 5 and
                        not self._is_title(line_stripped) and
                        not self._is_list_item(line_stripped)):
                    merged_lines[-1] = prev + line_stripped
                    continue
            merged_lines.append(line_stripped)

        merged_text = '\n'.join(merged_lines)

        # 第2步：双换行粗切
        raw_chunks = re.split(r'\n\s*\n', merged_text)
        raw_chunks = [p.strip() for p in raw_chunks if len(p.strip()) > 15]

        # 第3步：对每个粗切块做句子级细切
        final_sentences = []
        for chunk in raw_chunks:
            if len(chunk) > 500:
                subs = self._sentence_boundary_fine(chunk)
                final_sentences.extend(subs)
            elif len(chunk) > 15:
                final_sentences.append(chunk)

        # 第4步：如果句子太少，强制全文句子级切分
        if len(final_sentences) <= 3:
            final_sentences = self._sentence_boundary_fine(text)

        if not final_sentences and len(text.strip()) > 15:
            final_sentences = [text.strip()]

        # 第5步：合并连续短段（解决 Word→PDF 换行切碎问题）
        # 连续长度 < 50 字符的段落合并为一段，避免因换行位置不同导致匹配失败
        MERGE_THRESHOLD = 50
        merged = []
        buffer = ""
        for p in final_sentences:
            if len(p) < MERGE_THRESHOLD:
                buffer += p
            else:
                if buffer:
                    merged.append(buffer)
                    buffer = ""
                merged.append(p)
        if buffer:
            merged.append(buffer)
        # 确保合并后的段落至少15字符
        final_sentences = [p for p in merged if len(p) >= 15]

        return final_sentences

    def _sentence_boundary_fine(self, text: str) -> List[str]:
        """细粒度句子切分：按。！？；：等标点拆分，15-500 字符"""
        sentences = re.split(r'([。！？；：\n])', text)
        full_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            s = sentences[i].strip()
            if i + 1 < len(sentences):
                s += sentences[i + 1]
            if len(s) >= 15:
                full_sentences.append(s)

        if len(sentences) % 2 == 1 and sentences[-1].strip():
            remaining = sentences[-1].strip()
            if len(remaining) >= 15:
                full_sentences.append(remaining)

        if not full_sentences:
            return [text] if len(text) >= 15 else []

        # 合并过短句，拆分过长句
        result = []
        buffer = ""
        for sent in full_sentences:
            if len(buffer) + len(sent) < 60 and len(buffer) < 200:
                buffer += sent
            else:
                if len(buffer) >= 15:
                    if len(buffer) > 500:
                        for chunk_start in range(0, len(buffer), 400):
                            chunk = buffer[chunk_start:chunk_start + 500]
                            if len(chunk.strip()) >= 15:
                                result.append(chunk.strip())
                    else:
                        result.append(buffer.strip())
                buffer = sent

        if len(buffer.strip()) >= 15:
            result.append(buffer.strip())

        return result if result else [text.strip()]

    def _is_title(self, line: str) -> bool:
        """判断是否为标题"""
        if len(line) > 100:
            return False
        patterns = [
            r'^[一二三四五六七八九十]+[、．.．]',
            r'^第[一二三四五六七八九十]+[章节部分条]',
            r'^\d+[\.．]\s*',
            r'^\d+[\.．]\d+[\.．]?\s*',
            r'^[（(]\d+[)）]\s*',
            r'^[ABCDEFGHIJKLMNOPQRSTUVWXYZ][、．.]\s*',
            r'^[【\[\(]',
        ]
        return any(re.match(p, line) for p in patterns)

    def _is_list_item(self, line: str) -> bool:
        """判断是否为列表项"""
        patterns = [
            r'^[\-—–*•●○□△]+\s*',
            r'^\d+[\.．、)]+\s*',
            r'^[（(]\d+[)）]\s*',
            r'^[①②③④⑤⑥⑦⑧⑨⑩]+',
            r'^[a-z][、．)]+\s*',
        ]
        return any(re.match(p, line) for p in patterns)

    def _compute_simhash_from_tokens(self, word_list: list) -> str:
        """从预分词的 token 列表计算 64 位 SimHash（避免重复 jieba 分词）

        使用 hashlib.md5 确保跨进程确定性。
        """
        import hashlib
        if not word_list:
            return "0" * 16

        word_freq = Counter(word_list)
        v = [0] * 64
        for word, freq in word_freq.items():
            word_hash = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16)
            for i in range(64):
                if (word_hash >> i) & 1:
                    v[i] += freq
                else:
                    v[i] -= freq

        simhash_int = 0
        for i in range(64):
            if v[i] > 0:
                simhash_int |= (1 << i)

        return format(simhash_int, '016x')

    def _compute_minhash_cached(self, words: list, word_hash_cache: dict) -> str:
        """使用预计算的词哈希缓存计算 MinHash 签名（避免重复 hash_func 调用）

        Args:
            words: 过滤后的词列表
            word_hash_cache: {word: [hash1(word), hash2(word), ...]} 预计算缓存

        Returns:
            逗号分隔的 MinHash 签名字符串
        """
        if not words:
            return ""

        # 去重
        unique_words = set(words)

        # 对每个哈希函数取最小值（使用缓存值）
        num_hashes = len(self._minhash_funcs)
        values = [float('inf')] * num_hashes

        for w in unique_words:
            cached = word_hash_cache.get(w)
            if cached is None:
                # 缓存未命中（jieba 上下文相关分词可能导致）：实时计算
                cached = [hash_func(w) for hash_func in self._minhash_funcs]
                word_hash_cache[w] = cached  # 回填缓存
            for i in range(num_hashes):
                if cached[i] < values[i]:
                    values[i] = cached[i]

        return ','.join(map(str, values))

    def _extract_quotes(self, text: str) -> List[float]:
        """提取报价金额"""
        quotes = []
        patterns = [
            r'[¥￥]\s*([\d,]+\.?\d*)\s*万元',
            r'[¥￥]\s*([\d,]+\.?\d*)\s*元',
            r'人民币\s*([\d,]+\.?\d*)\s*万元',
            r'人民币\s*([\d,]+\.?\d*)\s*元',
            r'RMB\s*([\d,]+\.?\d*)\s*万元',
            r'RMB\s*([\d,]+\.?\d*)\s*元',
            r'([\d,]+\.?\d*)\s*万元',
            r'([\d,]+\.?\d*)\s*元'
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                try:
                    amount_str = match.group(1).replace(',', '')
                    amount = float(amount_str)
                    if '万' in match.group(0):
                        amount *= 10000
                    quotes.append(amount)
                except Exception:
                    continue
        return sorted(list(set(quotes)))

    def _parse_pdf_date(self, date_str: str) -> str:
        """解析 PDF 日期格式为 ISO 8601"""
        if not date_str:
            return ""
        match = self._RE_PDF_DATE.match(date_str)
        if match:
            try:
                from datetime import datetime
                year, month, day, hour, minute, second = match.groups()
                dt = datetime(int(year), int(month), int(day),
                              int(hour), int(minute), int(second))
                return dt.isoformat()
            except Exception:
                pass
        return ""

    def _generate_software_fingerprint(self, creator: str, producer: str) -> str:
        """生成软件指纹（去除版本号）"""
        creator_clean = self._RE_VERSION_STRIP.sub('', creator).strip()
        producer_clean = self._RE_VERSION_STRIP.sub('', producer).strip()
        return f"{creator_clean} + {producer_clean}".lower()

    @staticmethod
    def _generate_doc_id(file_path: str) -> str:
        """生成文档唯一 ID"""
        return hashlib.md5(file_path.encode('utf-8')).hexdigest()[:16]
