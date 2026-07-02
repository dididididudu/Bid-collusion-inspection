"""
文本差异计算工具 — 高亮标记相同部分

从 orchestrator.py 和 feature_cache.py 中提取，避免重复定义。
"""

import re
from difflib import SequenceMatcher


def compute_text_diff(text_a: str, text_b: str) -> tuple:
    """计算两个文本的差异并生成高亮标记

    使用 difflib.SequenceMatcher 找最长公共子串。

    Returns:
        (highlighted_a, highlighted_b, common_parts):
        - highlighted_a: 文本A的高亮版本（用【】标记相同部分）
        - highlighted_b: 文本B的高亮版本（用【】标记相同部分）
        - common_parts: 共同文本片段列表
    """
    clean_a = text_a.strip()
    clean_b = text_b.strip()

    sm = SequenceMatcher(None, clean_a, clean_b)
    matching_blocks = sm.get_matching_blocks()

    significant_blocks = [b for b in matching_blocks if b.size >= 10]

    common_parts = []
    for block in significant_blocks:
        if block.size >= 10:
            common_text = clean_a[block.a:block.a + block.size]
            common_parts.append(common_text.strip())

    common_parts.sort(key=len, reverse=True)
    common_parts = common_parts[:20]

    highlighted_a = _highlight_text_with_blocks(clean_a, significant_blocks, 'a')
    highlighted_b = _highlight_text_with_blocks(clean_b, significant_blocks, 'b')

    # 截断过长文本
    if len(highlighted_a) > 50000:
        highlighted_a = highlighted_a[:50000] + "\n... [文本过长，已截断]"
    if len(highlighted_b) > 50000:
        highlighted_b = highlighted_b[:50000] + "\n... [文本过长，已截断]"

    return highlighted_a, highlighted_b, common_parts


def _highlight_text_with_blocks(text: str, blocks: list, which: str) -> str:
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
            start, end = block.a, block.a + block.size
        else:
            start, end = block.b, block.b + block.size

        if start > last_end:
            result_parts.append(text[last_end:start])

        matched_text = text[start:end]
        if len(matched_text.strip()) > 0:
            result_parts.append(f"【{matched_text}】")

        last_end = end

    if last_end < len(text):
        result_parts.append(text[last_end:])

    result = ''.join(result_parts)
    # 清理连续的标记
    result = re.sub(r'】\s*【', '', result)

    return result
