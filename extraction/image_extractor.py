"""
两阶段智能 PDF 图片提取器

解决 Word→PDF 转换后图片被拆分为碎片（tiling）的问题。

阶段1: 直接提取有效的嵌入位图（过滤碎片/蒙版/纯色遮罩）
阶段2: 用 cluster_drawings 检测遗漏区域，高清渲染补全

用法:
    python image_extractor.py                              # 处理 input/ -> output/
    python image_extractor.py --input ./bids/ --output ./imgs/
    python image_extractor.py --dpi 200 --min-size 30      # 自定义参数
"""

import os
import sys
import logging
import argparse
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

import fitz  # PyMuPDF
from PIL import Image
import io
import math

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ================================================================
# 数据结构
# ================================================================

@dataclass
class ExtractedImage:
    """一张提取出的图片"""
    xref: int
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    pil_image: Image.Image
    source: str  # "bitmap" 或 "cluster"


# ================================================================
# 质量过滤
# ================================================================

def is_bad_image(pil_img: Image.Image, min_size: int = 40) -> Tuple[bool, str]:
    """检测图片是否为碎片/蒙版/纯色遮罩

    Returns:
        (is_bad, reason): True 表示应丢弃
    """
    w, h = pil_img.size
    if w < min_size or h < min_size:
        return True, f"尺寸过小 ({w}x{h})"

    # 转灰度分析像素分布
    if pil_img.mode != 'L':
        try:
            gray = pil_img.convert('L')
        except Exception:
            return False, ""
    else:
        gray = pil_img

    # 采样检测（降低大图计算开销）
    # 对边长 > 200 的图缩小后再检测
    if w > 200 or h > 200:
        ratio = min(200 / w, 200 / h)
        small = gray.resize((max(1, int(w * ratio)), max(1, int(h * ratio))),
                            Image.LANCZOS)
    else:
        small = gray

    pixels = list(small.getdata())
    if not pixels:
        return True, "空图片"

    mean_val = sum(pixels) / len(pixels)

    # 纯黑遮罩（蒙版通常为全黑）
    if mean_val < 15:
        return True, f"纯黑遮罩 (mean={mean_val:.1f})"

    # 纯白遮罩
    if mean_val > 245:
        return True, f"纯白遮罩 (mean={mean_val:.1f})"

    # 接近纯色（标准差小）且不是完整页面大小的图 → 可能是色块
    variance = sum((p - mean_val) ** 2 for p in pixels) / len(pixels)
    std_dev = math.sqrt(variance)
    if std_dev < 10 and (w < 300 or h < 300):
        return True, f"纯色块 (std={std_dev:.1f})"

    return False, ""


def fix_image_mode(pil_img: Image.Image) -> Image.Image:
    """修复颜色空间，统一输出 RGB"""
    mode = pil_img.mode
    if mode == 'CMYK':
        # CMYK → RGB
        return pil_img.convert('RGB')
    elif mode == 'P':
        # 调色板模式 → RGB
        return pil_img.convert('RGB')
    elif mode == 'RGBA':
        # RGBA → 白底 RGB（消除透明通道）
        background = Image.new('RGB', pil_img.size, (255, 255, 255))
        background.paste(pil_img, mask=pil_img.split()[3])
        return background
    elif mode == 'I' or mode == 'F':
        # 32位整数/浮点 → 8位 RGB
        return pil_img.convert('RGB')
    elif mode == 'L':
        # 灰度 → RGB
        return pil_img.convert('RGB')
    return pil_img


# ================================================================
# 阶段1: 提取有效嵌入位图
# ================================================================

def phase1_extract_bitmaps(
    page: fitz.Page,
    doc: fitz.Document,
    min_size: int = 40,
) -> Tuple[List[ExtractedImage], List[Tuple[float, float, float, float]]]:
    """阶段1：提取页面中所有有效嵌入位图

    通过 page.get_images() 获取所有嵌入图片的 xref + 名称(name)，
    然后用 page.get_image_bbox(name) 获取坐标。

    Returns:
        (提取的图片列表, 已处理区域 bbox 列表)
    """
    extracted = []
    handled_bboxes = []

    # 获取所有嵌入图片
    # get_images(full=True) 返回: (xref, smask, width, height, ...)
    images_on_page = page.get_images(full=True)
    if not images_on_page:
        return extracted, handled_bboxes

    page_rect = page.rect
    page_w = page_rect.width
    page_h = page_rect.height

    for img in images_on_page:
        xref = img[0]
        if xref <= 0:
            continue

        try:
            # 提取原始字节流
            base_image = doc.extract_image(xref)
            img_bytes = base_image.get("image")
            if not img_bytes:
                continue

            # PIL 打开
            pil_img = Image.open(io.BytesIO(img_bytes))

            # 质量过滤
            bad, reason = is_bad_image(pil_img, min_size)
            if bad:
                logger.debug(f"  跳过 xref={xref}: {reason}")
                continue

            # 修复颜色空间
            pil_img = fix_image_mode(pil_img)

            # 获取 bbox: 用图片名称 get_image_bbox(name)
            # 图片名称在 get_images(full=True) 返回的第8个字段 (index 7)
            img_name = img[7] if len(img) > 7 else None
            bbox = None
            if img_name:
                try:
                    bbox_obj = page.get_image_bbox(img_name)
                    if bbox_obj:
                        bbox = (bbox_obj.x0, bbox_obj.y0,
                                bbox_obj.x1, bbox_obj.y1)
                except Exception:
                    pass

            if not bbox:
                # 回退：用图像渲染尺寸和页面逻辑估算 bbox
                # 从 extract_image 获取原始尺寸
                img_w = base_image.get("width", 0)
                img_h = base_image.get("height", 0)
                if img_w and img_h:
                    logger.debug(f"  无法获取 xref={xref} 的 bbox，尝试从尺寸估算")
                    # 如果页面上只有一个此尺寸的图，用 get_image_info 匹配
                    for info in page.get_image_info():
                        info_w = info.get('width', 0)
                        info_h = info.get('height', 0)
                        if info_w == img_w and info_h == img_h:
                            info_bbox = info.get('bbox')
                            if info_bbox and len(info_bbox) == 4:
                                bbox = tuple(info_bbox)
                                logger.debug(f"  通过尺寸匹配到 bbox={bbox}")
                                break

            if not bbox:
                logger.debug(f"  无法定位 xref={xref}，跳过")
                continue

            # 忽略超出页面范围太多的图
            x0, y0, x1, y1 = bbox
            if x1 < -10 or y1 < -10 or x0 > page_w + 10 or y0 > page_h + 10:
                continue

            extracted.append(ExtractedImage(
                xref=xref, bbox=bbox, pil_image=pil_img, source="bitmap",
            ))
            handled_bboxes.append(bbox)

            logger.debug(f"  提取位图 xref={xref}, bbox=({bbox}), size={pil_img.size}")

        except Exception as e:
            logger.debug(f"  提取 xref={xref} 失败: {e}")
            continue

    logger.info(f"  阶段1: 提取 {len(extracted)} 张有效位图")
    return extracted, handled_bboxes


# ================================================================
# 阶段2: cluster_drawings 补漏
# ================================================================

def bbox_intersection_area(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    """计算两个 bbox 的交集面积"""
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x0 < x1 and y0 < y1:
        return (x1 - x0) * (y1 - y0)
    return 0.0


def bbox_area(bbox: Tuple[float, float, float, float]) -> float:
    """计算 bbox 面积"""
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def get_cluster_bbox(cluster) -> Tuple[float, float, float, float]:
    """从 cluster_drawings 返回的条目中提取 bbox"""
    # cluster_drawings 返回的是 fitz.Rect 或 (x0, y0, x1, y1) 格式的 dict
    if hasattr(cluster, 'x0'):
        return (cluster.x0, cluster.y0, cluster.x1, cluster.y1)
    elif isinstance(cluster, (list, tuple)) and len(cluster) >= 4:
        return (float(cluster[0]), float(cluster[1]),
                float(cluster[2]), float(cluster[3]))
    elif isinstance(cluster, dict):
        return (cluster.get('x0', 0), cluster.get('y0', 0),
                cluster.get('x1', 0), cluster.get('y1', 0))
    return (0, 0, 0, 0)


def cluster_area(cluster) -> float:
    """计算 cluster 面积"""
    bbox = get_cluster_bbox(cluster)
    return bbox_area(bbox)


def merge_bboxes(
    bboxes: List[Tuple[float, float, float, float]],
    gap: float = 60,
) -> List[Tuple[float, float, float, float]]:
    """在 gap 距离内的 bbox 合并为一个整体矩形

    不断迭代直到没有更多合并发生。
    """
    if not bboxes:
        return []

    merged = list(bboxes)
    changed = True
    while changed:
        changed = False
        new_list = []
        used = [False] * len(merged)
        for i in range(len(merged)):
            if used[i]:
                continue
            current = list(merged[i])
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                other = merged[j]
                # 检查是否在 gap 范围内
                # 水平方向重叠或间距 < gap，且垂直方向有交集
                h_overlap = (
                    (current[0] - gap <= other[2] and current[2] + gap >= other[0])
                )
                v_overlap = (
                    (current[1] - gap <= other[3] and current[3] + gap >= other[1])
                )
                if h_overlap and v_overlap:
                    # 合并
                    current[0] = min(current[0], other[0])
                    current[1] = min(current[1], other[1])
                    current[2] = max(current[2], other[2])
                    current[3] = max(current[3], other[3])
                    used[j] = True
                    changed = True
            new_list.append(tuple(current))
            used[i] = True
        merged = new_list

    return merged


def phase2_render_clusters(
    page: fitz.Page,
    handled_bboxes: List[Tuple[float, float, float, float]],
    dpi: int = 300,
    x_tolerance: float = 10,
    y_tolerance: float = 10,
    min_cluster_width_ratio: float = 0.4,
    min_cluster_height: float = 80,
    overlap_threshold: float = 0.5,
    merge_gap: float = 60,
    extend_px: float = 5,
) -> List[ExtractedImage]:
    """阶段2：用 cluster_drawings 检测遗漏区域并渲染

    Returns:
        渲染补全的图片列表
    """
    extracted = []
    page_rect = page.rect
    page_w = page_rect.width
    page_h = page_rect.height

    try:
        clusters = page.cluster_drawings(
            x_tolerance=x_tolerance,
            y_tolerance=y_tolerance,
            final_filter=False,
        )
    except Exception as e:
        logger.warning(f"  cluster_drawings 失败: {e}")
        return extracted

    if not clusters:
        return extracted

    logger.debug(f"  原始 clusters: {len(clusters)} 个")

    # 过滤过小集群
    valid_clusters = []
    for cl in clusters:
        bbox = get_cluster_bbox(cl)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        if w < page_w * min_cluster_width_ratio or h < min_cluster_height:
            continue

        # 检查是否已被阶段1的位图覆盖
        total_overlap = 0.0
        cl_area = bbox_area(bbox)
        if cl_area <= 0:
            continue

        for hb in handled_bboxes:
            total_overlap += bbox_intersection_area(bbox, hb)

        overlap_ratio = total_overlap / cl_area
        if overlap_ratio > overlap_threshold:
            # 已被覆盖，跳过
            continue

        valid_clusters.append(bbox)

    if not valid_clusters:
        logger.debug("  所有 cluster 均已被覆盖或过小")
        return extracted

    logger.debug(f"  候选 clusters: {len(valid_clusters)} 个")

    # 合并相邻 cluster
    merged = merge_bboxes(valid_clusters, gap=merge_gap)
    logger.debug(f"  合并后渲染区域: {len(merged)} 个")

    # 对每个区域高清渲染
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for idx, bbox in enumerate(merged):
        x0, y0, x1, y1 = bbox

        # 扩展边距
        x0 = max(0, x0 - extend_px)
        y0 = max(0, y0 - extend_px)
        x1 = min(page_w, x1 + extend_px)
        y1 = min(page_h, y1 + extend_px)

        if x1 - x0 < 10 or y1 - y0 < 10:
            continue

        clip = fitz.Rect(x0, y0, x1, y1)

        try:
            pix = page.get_pixmap(matrix=mat, clip=clip)
            img_data = pix.tobytes("png")
            pil_img = Image.open(io.BytesIO(img_data))
            pil_img = fix_image_mode(pil_img)

            extracted.append(ExtractedImage(
                xref=-1, bbox=(x0, y0, x1, y1),
                pil_image=pil_img, source="cluster",
            ))

            logger.debug(f"  渲染 cluster #{idx}: ({x0:.0f},{y0:.0f} {x1:.0f},{y1:.0f}) "
                         f"→ {pil_img.size}")

        except Exception as e:
            logger.warning(f"  渲染 cluster #{idx} 失败: {e}")
            continue

    logger.info(f"  阶段2: 渲染补全 {len(extracted)} 张图片")
    return extracted


# ================================================================
# 主处理流程
# ================================================================

def process_pdf(
    pdf_path: str,
    output_dir: str,
    dpi: int = 300,
    min_size: int = 40,
    x_tolerance: float = 10,
    y_tolerance: float = 10,
    min_cluster_width_ratio: float = 0.4,
    min_cluster_height: float = 80,
    overlap_threshold: float = 0.5,
    merge_gap: float = 60,
    extend_px: float = 5,
) -> int:
    """处理单个 PDF 文件，提取所有图片

    Returns:
        提取的图片总数
    """
    filename = os.path.basename(pdf_path)
    name, _ = os.path.splitext(filename)
    logger.info(f"处理: {filename}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"  无法打开 PDF: {e}")
        return 0

    total_images = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        logger.info(f"  第 {page_num + 1}/{len(doc)} 页")

        # 阶段1: 提取嵌入位图
        bitmaps, handled_bboxes = phase1_extract_bitmaps(
            page, doc, min_size=min_size,
        )

        # 阶段2: cluster_drawings 补漏
        clusters = phase2_render_clusters(
            page, handled_bboxes,
            dpi=dpi,
            x_tolerance=x_tolerance,
            y_tolerance=y_tolerance,
            min_cluster_width_ratio=min_cluster_width_ratio,
            min_cluster_height=min_cluster_height,
            overlap_threshold=overlap_threshold,
            merge_gap=merge_gap,
            extend_px=extend_px,
        )

        # 保存阶段1位图
        for idx, img in enumerate(bitmaps):
            x0 = int(img.bbox[0])
            out_name = f"{name}_p{page_num + 1}_img{img.xref}_{x0}.png"
            out_path = os.path.join(output_dir, out_name)
            img.pil_image.save(out_path, "PNG")
            logger.debug(f"  保存: {out_name}")

        # 保存阶段2渲染图
        for idx, img in enumerate(clusters):
            out_name = f"{name}_p{page_num + 1}_cluster{idx}_{dpi}dpi.png"
            out_path = os.path.join(output_dir, out_name)
            img.pil_image.save(out_path, "PNG")
            logger.debug(f"  保存: {out_name}")

        page_count = len(bitmaps) + len(clusters)
        total_images += page_count
        logger.info(f"  本页提取: {page_count} 张 (位图 {len(bitmaps)} + 渲染 {len(clusters)})")

    doc.close()
    logger.info(f"  {filename}: 共 {total_images} 张图片")
    return total_images


# ================================================================
# 入口
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="两阶段智能 PDF 图片提取器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python image_extractor.py
  python image_extractor.py --input ./bids/ --output ./imgs/
  python image_extractor.py --dpi 200 --min-size 30
  python image_extractor.py --debug
        """,
    )
    parser.add_argument('--input', '-i', default='input',
                        help='输入 PDF 文件夹路径 (默认: input/)')
    parser.add_argument('--output', '-o', default='output',
                        help='输出图片文件夹路径 (默认: output/)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='阶段2渲染 DPI (默认: 300)')
    parser.add_argument('--min-size', type=int, default=40,
                        help='图片最小边长，低于此值视为碎片 (默认: 40)')
    parser.add_argument('--debug', action='store_true',
                        help='输出详细调试日志')

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    input_dir = args.input
    output_dir = args.output

    if not os.path.isdir(input_dir):
        logger.error(f"输入文件夹不存在: {input_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # 收集 PDF 文件
    pdf_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith('.pdf')
    ])
    if not pdf_files:
        logger.error(f"未找到 PDF 文件: {input_dir}")
        sys.exit(1)

    logger.info(f"找到 {len(pdf_files)} 个 PDF 文件，输出至: {output_dir}")

    total_all = 0
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        count = process_pdf(
            pdf_path, output_dir,
            dpi=args.dpi,
            min_size=args.min_size,
        )
        total_all += count

    logger.info(f"处理完成! 共提取 {total_all} 张图片")


if __name__ == "__main__":
    main()
