"""
图片导出工具 — 将 PDF 中的图片提取并保存，供报告中展示

在 Phase 1 提取阶段保存原始图片，Phase 5 报告阶段嵌入 HTML。
JSON 报告只包含图片路径，不包含图片数据。
"""

import os
import io
import base64
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


def save_image_to_dir(
    image_bytes: bytes,
    output_dir: str,
    doc_id: str,
    page_num: int,
    img_index: int,
    fmt: str = "png",
) -> str:
    """保存图片到输出目录，返回相对路径

    Args:
        image_bytes: 图片字节数据
        output_dir: 输出根目录
        doc_id: 文档 ID
        page_num: 页码
        img_index: 图片序号
        fmt: 图片格式

    Returns:
        图片的相对路径（相对于 output_dir）
    """
    images_dir = os.path.join(output_dir, "images", doc_id[:12])
    os.makedirs(images_dir, exist_ok=True)

    filename = f"page{page_num}_img{img_index}.{fmt}"
    filepath = os.path.join(images_dir, filename)

    if not os.path.exists(filepath):
        with open(filepath, "wb") as f:
            f.write(image_bytes)

    # 返回相对路径
    return os.path.join("images", doc_id[:12], filename)


def save_image_from_array(
    img_array,
    output_dir: str,
    doc_id: str,
    page_num: int,
    img_index: int,
) -> str:
    """从 numpy 数组保存图片

    Returns:
        图片的相对路径
    """
    img = Image.fromarray(img_array)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return save_image_to_dir(
        buf.getvalue(), output_dir, doc_id, page_num, img_index, "png"
    )


def image_to_base64(image_path: str) -> str:
    """将图片文件转为 base64 字符串（用于 HTML 嵌入）"""
    try:
        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(image_path)[1].lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(
            ext.replace(".", ""), "image/png"
        )
        return f"data:{mime};base64,{data}"
    except FileNotFoundError:
        logger.warning(f"图片文件不存在: {image_path}")
        return ""
    except Exception as e:
        logger.warning(f"图片转换失败: {e}")
        return ""


def find_matching_images_from_pdf(
    pdf_path: str,
    doc_id: str,
    matched_hashes: List[str],
    output_dir: str,
    min_size: int = 50,
) -> Dict[str, str]:
    """从 PDF 中提取匹配的图片并保存

    遍历 PDF 中的嵌入图片，找到哈希匹配的图片并保存。

    Args:
        pdf_path: PDF 文件路径
        doc_id: 文档 ID
        matched_hashes: 需要匹配的图片哈希列表
        output_dir: 输出目录
        min_size: 最小图片尺寸

    Returns:
        {image_hash: saved_path} 字典
    """
    import fitz
    import imagehash

    result = {}
    hash_set = set(matched_hashes)

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"无法打开 PDF: {e}")
        return result

    try:
        img_counter = 0
        for page_num in range(doc.page_count):
            page = doc[page_num]
            image_list = page.get_images(full=True)

            for img_info in image_list:
                xref = img_info[0]
                width, height = img_info[2], img_info[3]

                if width < min_size or height < min_size:
                    continue

                try:
                    base_image = doc.extract_image(xref)
                    if not base_image:
                        continue

                    image_bytes = base_image.get("image")
                    if not image_bytes or len(image_bytes) < 1024:
                        continue

                    img = Image.open(io.BytesIO(image_bytes))
                    if img.mode not in ("RGB", "L"):
                        img = img.convert("RGB")

                    phash = str(imagehash.phash(img))

                    if phash in hash_set and phash not in result:
                        img_counter += 1
                        saved_path = save_image_to_dir(
                            image_bytes, output_dir, doc_id,
                            page_num, img_counter,
                            fmt=base_image.get("ext", "png"),
                        )
                        result[phash] = saved_path

                except Exception:
                    continue

    finally:
        doc.close()

    return result
