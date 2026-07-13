"""
OCR 辅助函数 — 供 orchestrator 和 parallel_workers 共用

提取 PDF 页面/嵌入图片的文字，聚合为段落注入文本匹配管线。
"""

import os
import io
import logging
import threading
from typing import List, Dict, Optional, Tuple

import numpy as np
from PIL import Image
import imagehash

from config import DetectionConfig
from data_structures import ChunkResult

logger = logging.getLogger(__name__)

# OCR 聚合段落的虚拟块索引
OCR_CHUNK_INDEX: int = 1000000

# 线程局部存储 — 每个线程独立的 OCR 引擎
_ocr_tls = threading.local()


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


def _fix_pil_mode(img: Image.Image) -> Image.Image:
    """统一转为 RGB，避免 OCR/Hash 对 CMYK、RGBA、P 模式处理不一致。"""
    if img.mode == 'RGB':
        return img
    if img.mode == 'RGBA':
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert('RGB')


def _bbox_area(bbox: Tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _bbox_overlap_ratio(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix0 >= ix1 or iy0 >= iy1:
        return 0.0
    area = _bbox_area(a)
    return ((ix1 - ix0) * (iy1 - iy0)) / area if area > 0 else 0.0


def _merge_bboxes(
    bboxes: List[Tuple[float, float, float, float]],
    gap: float = 12,
) -> List[Tuple[float, float, float, float]]:
    """合并相邻 bbox，用于把碎片化矢量图合成为一个 OCR 区域。"""
    merged = list(bboxes)
    changed = True
    while changed:
        changed = False
        result = []
        used = [False] * len(merged)
        for i, box in enumerate(merged):
            if used[i]:
                continue
            cur = list(box)
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                other = merged[j]
                near_x = cur[0] - gap <= other[2] and cur[2] + gap >= other[0]
                near_y = cur[1] - gap <= other[3] and cur[3] + gap >= other[1]
                if near_x and near_y:
                    cur[0] = min(cur[0], other[0])
                    cur[1] = min(cur[1], other[1])
                    cur[2] = max(cur[2], other[2])
                    cur[3] = max(cur[3], other[3])
                    used[j] = True
                    changed = True
            used[i] = True
            result.append(tuple(cur))
        merged = result
    return merged


def _add_ocr_task(
    tasks: list,
    seen_hashes: set,
    pil_img: Image.Image,
    min_conf: float,
    ocr_engine_type: str,
    use_gpu: bool,
    page_num: int,
    source_prefix: str,
) -> None:
    if pil_img.width < 10 or pil_img.height < 10:
        return
    img = _fix_pil_mode(pil_img)
    phash = f"{source_prefix}{imagehash.phash(img)}"
    if phash in seen_hashes:
        return
    seen_hashes.add(phash)
    tasks.append((
        np.array(img), phash, img.width, img.height,
        min_conf, ocr_engine_type, use_gpu, page_num,
    ))


def _collect_page_ocr_tasks(
    doc,
    page,
    page_num: int,
    min_img_size: int,
    min_conf: float,
    ocr_engine_type: str,
    use_gpu: bool,
    seen_hashes: set,
    config: Optional[DetectionConfig] = None,
) -> List[tuple]:
    """收集当前页 OCR 图片任务。

    唯一策略：
    - 完整嵌入图片：直接 doc.extract_image(xref) 读取原图，不整页渲染裁剪。
    - 小尺寸 PNG 碎片：不单独 OCR。
    - 矢量/碎片化图形：cluster_drawings 合并后，只渲染合并区域再 OCR。
    """
    tasks = []
    page_w, page_h = page.rect.width, page.rect.height
    page_area = max(1.0, page_w * page_h)
    whole_image_bboxes = []
    small_fragment_bboxes = []
    small_png_fragments = 0
    skipped_vector_regions = 0
    vector_min_area_ratio = getattr(config, 'OCR_VECTOR_MIN_AREA_RATIO', 0.01)
    vector_max_area_ratio = getattr(config, 'OCR_VECTOR_MAX_AREA_RATIO', 0.35)
    vector_max_side_ratio = getattr(config, 'OCR_VECTOR_MAX_SIDE_RATIO', 0.85)

    image_list = page.get_images(full=True)
    seen_xrefs = set()
    for img_info in image_list:
        xref = img_info[0]
        if xref <= 0 or xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)
        try:
            base = doc.extract_image(xref)
            img_bytes = base.get("image") if base else None
            if not img_bytes:
                continue
            ext = (base.get("ext") or "").lower()
            width = int(base.get("width") or img_info[2] or 0)
            height = int(base.get("height") or img_info[3] or 0)
            is_small_png = (
                ext == "png"
                and (width < min_img_size * 2 or height < min_img_size * 2)
            )
            if is_small_png:
                small_png_fragments += 1
                img_name = img_info[7] if len(img_info) > 7 else None
                if img_name:
                    try:
                        bbox_obj = page.get_image_bbox(img_name)
                        if bbox_obj:
                            small_fragment_bboxes.append(
                                (bbox_obj.x0, bbox_obj.y0, bbox_obj.x1, bbox_obj.y1)
                            )
                    except Exception:
                        pass
                continue
            if width < min_img_size or height < min_img_size:
                continue

            img_name = img_info[7] if len(img_info) > 7 else None
            img_bbox = None
            if img_name:
                try:
                    bbox_obj = page.get_image_bbox(img_name)
                    if bbox_obj:
                        img_bbox = (bbox_obj.x0, bbox_obj.y0, bbox_obj.x1, bbox_obj.y1)
                        whole_image_bboxes.append(img_bbox)
                except Exception:
                    pass

            pil_img = Image.open(io.BytesIO(img_bytes))
            _add_ocr_task(
                tasks, seen_hashes, pil_img, min_conf,
                ocr_engine_type, use_gpu, page_num, "img:"
            )
        except Exception as e:
            logger.debug(f"OCR: 第 {page_num} 页嵌入图片 xref={xref} 提取失败: {e}")

    # 矢量图或碎片图：只在发现小 PNG 碎片时启用 cluster_drawings。
    # 普通表格线/矢量装饰也会被 cluster_drawings 捕获；如果无小图片碎片信号，
    # 不渲染这些区域，避免把大量版式线条当图片 OCR。
    if small_fragment_bboxes:
        try:
            raw_clusters = page.cluster_drawings(
                x_tolerance=10, y_tolerance=10, final_filter=False
            )
        except Exception as e:
            logger.debug(f"OCR: 第 {page_num} 页 cluster_drawings 失败: {e}")
            raw_clusters = []
    else:
        raw_clusters = []

    cluster_bboxes = list(small_fragment_bboxes)
    min_cluster_w = max(80, min_img_size * 2)
    min_cluster_h = max(50, min_img_size)
    for cl in raw_clusters:
        try:
            bbox = (cl.x0, cl.y0, cl.x1, cl.y1)
        except Exception:
            continue
        x0, y0, x1, y1 = bbox
        w, h = x1 - x0, y1 - y0
        if w < min_cluster_w or h < min_cluster_h:
            continue
        if x1 < 0 or y1 < 0 or x0 > page_w or y0 > page_h:
            continue
        if any(_bbox_overlap_ratio(bbox, ib) > 0.5 for ib in whole_image_bboxes):
            continue
        area_ratio = _bbox_area(bbox) / page_area
        side_like_page = (
            w / max(page_w, 1) >= vector_max_side_ratio
            or h / max(page_h, 1) >= vector_max_side_ratio
        )
        if (
            area_ratio < vector_min_area_ratio
            or area_ratio > vector_max_area_ratio
            or side_like_page
        ):
            skipped_vector_regions += 1
            continue
        cluster_bboxes.append(bbox)

    for bbox in _merge_bboxes(cluster_bboxes, gap=12):
        try:
            x0, y0, x1, y1 = bbox
            x0 = max(0, x0 - 3)
            y0 = max(0, y0 - 3)
            x1 = min(page_w, x1 + 3)
            y1 = min(page_h, y1 + 3)
            if x1 - x0 < min_cluster_w or y1 - y0 < min_cluster_h:
                continue
            merged_bbox = (x0, y0, x1, y1)
            mw, mh = x1 - x0, y1 - y0
            merged_area_ratio = _bbox_area(merged_bbox) / page_area
            merged_side_like_page = (
                mw / max(page_w, 1) >= vector_max_side_ratio
                or mh / max(page_h, 1) >= vector_max_side_ratio
            )
            if (
                merged_area_ratio < vector_min_area_ratio
                or merged_area_ratio > vector_max_area_ratio
                or merged_side_like_page
            ):
                continue
            import fitz
            pix = page.get_pixmap(
                matrix=fitz.Matrix(200 / 72.0, 200 / 72.0),
                clip=fitz.Rect(x0, y0, x1, y1),
            )
            pil_img = Image.open(io.BytesIO(pix.tobytes("png")))
            _add_ocr_task(
                tasks, seen_hashes, pil_img, min_conf,
                ocr_engine_type, use_gpu, page_num, "vec:"
            )
        except Exception as e:
            logger.debug(f"OCR: 第 {page_num} 页矢量合并区域渲染失败: {e}")

    if tasks:
        logger.debug(
            f"OCR: 第 {page_num} 页收集 {len(tasks)} 张图 "
            f"(跳过小PNG碎片 {small_png_fragments} 个, "
            f"跳过矢量区域 {skipped_vector_regions} 个, "
            f"clusters={len(cluster_bboxes)})"
        )
    return tasks


def _ocr_crop_worker(args: tuple) -> Optional[tuple]:
    """单个裁剪区域的 OCR 工作函数（线程安全，无共享状态写入）

    Args:
        args: (img_array, phash, image_width, image_height, min_conf,
               ocr_engine_type, use_gpu, page_num)

    Returns:
        (page_num, OCRResult) 或 None（低于置信度或空文本）
    """
    img_array, phash, img_w, img_h, min_conf = args[:5]
    ocr_engine_type = args[5] if len(args) > 5 else 'easyocr'
    use_gpu = args[6] if len(args) > 6 else False
    page_num = args[7] if len(args) > 7 else 0
    if img_array.size == 0:
        return None

    # 使用线程局部的 OCR 引擎（每个线程独立实例，避免多线程共享同一引擎的线程安全问题）
    # 首次调用时会初始化，后续复用
    if not hasattr(_ocr_tls, '_engine'):
        from image_analysis.image_ocr import ImageOCREngine
        _ocr_tls._engine = ImageOCREngine(
            use_gpu=use_gpu, engine=ocr_engine_type,
        )
    ocr = _ocr_tls._engine

    ocr_result = ocr.extract(img_array)
    ocr_result.image_hash = phash
    ocr_result.image_width = img_w
    ocr_result.image_height = img_h
    ocr_result.thumbnail = _make_thumbnail(img_array)
    ocr_result.non_text_hash = _compute_non_text_hash(
        img_array, ocr_result.bboxes
    )

    if ocr_result.confidence < min_conf or not ocr_result.text.strip():
        return None

    return (page_num, ocr_result)


def _collect_page_ocr_tasks_from_file(args: tuple) -> list:
    """并行收集单页 OCR 任务。

    每个线程独立打开 PDF，避免共享 PyMuPDF document/page 对象。
    """
    (
        file_path, page_num, min_img_size, min_conf,
        ocr_engine_type, use_gpu, config,
    ) = args
    import fitz

    doc = fitz.open(file_path)
    try:
        page = doc[page_num]
        return _collect_page_ocr_tasks(
            doc, page, page_num, min_img_size, min_conf,
            ocr_engine_type, use_gpu, set(), config,
        )
    finally:
        doc.close()


def ocr_pages(
    file_path: str,
    doc_id: str,
    page_count: int,
    cache,
    config: DetectionConfig,
    ocr_engine,
    ocr_workers: int = 1,
    gpu_manager=None,
) -> int:
    """对 PDF 中的图片运行 OCR 提取文字

    当 GPU Manager 可用时（推荐），图片通过批量 IPC 提交到统一 GPU 进程；
    否则回退到本地线程池（兼容旧行为）。

    Args:
        file_path: PDF 文件路径
        doc_id: 文档 ID
        page_count: 总页数
        cache: DocumentCache 实例
        config: DetectionConfig
        ocr_engine: ImageOCREngine 实例（GPU Manager 模式下可为 None）
        ocr_workers: OCR 并行线程数（GPU Manager 模式下忽略）
        gpu_manager: GPUManager 实例（可选）
        ocr_workers: OCR 并行线程数

    Returns:
        成功 OCR 的图片数
    """
    if not config.ENABLE_OCR:
        return 0

    # GPU Manager 路径：引擎在 Manager 进程中，本地不需要可用
    use_gpu_manager = (
        gpu_manager is not None
        and gpu_manager.enabled
    )
    if not use_gpu_manager:
        if ocr_engine is None or not ocr_engine.is_available:
            logger.debug("OCR 引擎不可用，跳过图片文字提取")
            return 0

    existing_ocr = cache.load_image_ocr_results(doc_id)
    if existing_ocr:
        if all(
            str(r.get('image_hash', '')).startswith(('img:', 'vec:'))
            for r in existing_ocr
        ):
            logger.info(f"OCR: {os.path.basename(file_path)} 已有 {len(existing_ocr)} 条结果，跳过")
            return len(existing_ocr)
        logger.info(
            f"OCR: {os.path.basename(file_path)} 检测到旧版 OCR 缓存，"
            "将按嵌入图片/合并矢量图策略重算"
        )
        try:
            with cache.transaction() as conn:
                conn.execute("DELETE FROM image_ocr_results WHERE doc_id = ?", (doc_id,))
        except Exception as e:
            logger.warning(f"OCR: 清理旧版缓存失败，将继续尝试重算: {e}")

    import fitz

    logger.info(
        f"OCR: 嵌入图片/合并矢量图模式 "
        f"{os.path.basename(file_path)} ({page_count} 页, "
        f"workers={ocr_workers})..."
    )

    min_conf = config.OCR_MIN_CONFIDENCE
    min_img_size = getattr(config, 'IMAGE_MIN_SIZE', 50)
    sample_step = config.OCR_SAMPLE_STEP

    # 先收集所有需要 OCR 的图片：
    # 完整嵌入图直接读取原图，矢量/碎片图只渲染合并后的区域。
    crop_tasks = []
    ocr_engine_type = getattr(config, 'OCR_ENGINE', 'easyocr')
    use_gpu = getattr(config, 'USE_GPU', False)
    page_nums = list(range(0, page_count, sample_step))
    collect_workers = min(
        max(1, getattr(config, 'OCR_COLLECT_WORKERS', 1)),
        len(page_nums) or 1,
        os.cpu_count() or 4,
    )

    if collect_workers > 1:
        logger.info(
            f"OCR: 页级任务收集并行 {collect_workers} workers, "
            f"{len(page_nums)} 页"
        )
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=collect_workers) as executor:
            futures = [
                executor.submit(
                    _collect_page_ocr_tasks_from_file,
                    (
                        file_path, page_num, min_img_size, min_conf,
                        ocr_engine_type, use_gpu, config,
                    ),
                )
                for page_num in page_nums
            ]
            for future in as_completed(futures):
                try:
                    crop_tasks.extend(future.result())
                except Exception as e:
                    logger.debug(f"OCR: 页级任务收集失败 ({os.path.basename(file_path)}): {e}")
    else:
        try:
            doc = fitz.open(file_path)
        except Exception as e:
            logger.error(f"OCR: 无法打开 PDF ({file_path}): {e}")
            return 0

        # 已见过的图片哈希去重，相同 pHash 只 OCR 一次。
        seen_hashes = set()
        try:
            for page_num in page_nums:
                try:
                    page = doc[page_num]
                    crop_tasks.extend(_collect_page_ocr_tasks(
                        doc, page, page_num, min_img_size, min_conf,
                        ocr_engine_type, use_gpu, seen_hashes, config,
                    ))

                except Exception as e:
                    logger.debug(f"OCR: 第 {page_num} 页失败 ({os.path.basename(file_path)}): {e}")
                    continue

        finally:
            doc.close()

    if crop_tasks:
        deduped_tasks = []
        seen_hashes = set()
        for task in crop_tasks:
            phash = task[1]
            if phash in seen_hashes:
                continue
            seen_hashes.add(phash)
            deduped_tasks.append(task)
        crop_tasks = deduped_tasks

    if not crop_tasks:
        return 0

    img_task_count = sum(1 for t in crop_tasks if str(t[1]).startswith('img:'))
    vec_task_count = sum(1 for t in crop_tasks if str(t[1]).startswith('vec:'))
    logger.info(
        f"OCR: {os.path.basename(file_path)} — {len(crop_tasks)} 张图片待识别 "
        f"(img={img_task_count}, vec={vec_task_count})"
    )

    ocr_results = []

    if use_gpu_manager:
        # ── GPU Manager 路径：批量提交到统一 GPU 进程 ──
        images = [t[0] for t in crop_tasks]
        metadata = [
            {'phash': t[1], 'img_w': t[2], 'img_h': t[3],
             'min_conf': t[4], 'page_num': t[7]}
            for t in crop_tasks
        ]
        logger.info("OCR: 提交 %d 张图片到 GPU Manager", len(images))

        batch_results, meta_list = gpu_manager.batch_ocr(images, metadata)
        for idx, (result, meta) in enumerate(zip(batch_results, meta_list)):
            if result.confidence >= meta['min_conf'] and result.text.strip():
                img_array = images[idx] if 0 <= idx < len(images) else None
                if img_array is not None:
                    result.thumbnail = _make_thumbnail(img_array)
                    result.non_text_hash = _compute_non_text_hash(img_array, result.bboxes)
                ocr_results.append((
                    result, result.image_hash or meta['phash'],
                    result.image_width or meta['img_w'],
                    result.image_height or meta['img_h'],
                    meta['page_num'],
                ))
    else:
        # ── 本地路径：多线程并行 OCR（兼容旧行为） ──
        num_workers = max(1, min(ocr_workers, len(crop_tasks), os.cpu_count() or 4))

        if num_workers <= 1:
            for task in crop_tasks:
                img_array, phash, img_w, img_h, mc, _, _, page_n = task[:8]
                result = ocr_engine.extract(img_array)
                result.image_hash = phash
                result.image_width = img_w
                result.image_height = img_h
                result.thumbnail = _make_thumbnail(img_array)
                result.non_text_hash = _compute_non_text_hash(img_array, result.bboxes)
                if result.confidence >= mc and result.text.strip():
                    ocr_results.append((result, phash, img_w, img_h, page_n))
        else:
            logger.info(f"OCR 并行模式: {num_workers} workers, {len(crop_tasks)} 张图")
            from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_map = {executor.submit(_ocr_crop_worker, task): task
                              for task in crop_tasks}
                for future in as_completed(future_map):
                    try:
                        # 单张图最多 30 秒，超时则跳过
                        ret = future.result(timeout=30)
                        if ret is not None:
                            page_n, result = ret
                            ocr_results.append((
                                result, result.image_hash,
                                result.image_width, result.image_height,
                                page_n,
                            ))
                    except TimeoutError:
                        logger.error(f"OCR 线程超时（单张图超过 30 秒）")
                    except Exception as e:
                        logger.debug(f"OCR 线程异常: {e}")

    # 批量入库（一条事务，避免逐条 commit 的开销）
    import json as _json
    stored = 0
    try:
        with cache.transaction() as conn:
            for result, phash, img_w, img_h, page_n in ocr_results:
                try:
                    conn.execute(
                        "INSERT OR REPLACE INTO image_ocr_results "
                        "(doc_id, page_num, image_hash, ocr_text, ocr_words_json, "
                        "text_bboxes_json, confidence, non_text_hash, "
                        "image_width, image_height, thumbnail) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            doc_id, page_n, phash,
                            result.text,
                            _json.dumps(result.words or [], ensure_ascii=False),
                            _json.dumps(result.bboxes or [], ensure_ascii=False),
                            result.confidence,
                            result.non_text_hash,
                            img_w, img_h,
                            result.thumbnail if result.thumbnail else None,
                        )
                    )
                    stored += 1
                except Exception as e:
                    logger.debug(f"OCR 结果入库失败: {e}")
    except Exception as e:
        logger.error(f"OCR 批量入库事务失败: {e}")

    if stored > 0:
        logger.info(f"OCR: {os.path.basename(file_path)} — {stored}/{len(crop_tasks)} 张图片成功提取文字")

    return stored


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

    # 虚拟 ChunkResult（标记 source="ocr"）
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
        source="ocr",
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
