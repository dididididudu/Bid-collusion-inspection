"""
工具函数模块
"""
import hashlib
from typing import List, Set


def hamming_distance(hash1: str, hash2: str) -> int:
    """
    计算两个哈希字符串的汉明距离

    Args:
        hash1: 第一个哈希字符串
        hash2: 第二个哈希字符串

    Returns:
        汉明距离
    """
    if len(hash1) != len(hash2):
        return max(len(hash1), len(hash2))

    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))


def jaccard_similarity(set1: Set, set2: Set) -> float:
    """
    计算两个集合的Jaccard相似度

    Args:
        set1: 第一个集合
        set2: 第二个集合

    Returns:
        Jaccard相似度 (0-1)
    """
    if not set1 or not set2:
        return 0.0

    intersection = set1 & set2
    union = set1 | set2

    return len(intersection) / len(union) if union else 0.0


def normalize_text(text: str) -> str:
    """
    标准化文本（去除多余空格、换行等）

    Args:
        text: 原始文本

    Returns:
        标准化后的文本
    """
    import re

    # 替换多个空白字符为单个空格
    text = re.sub(r'\s+', ' ', text)

    # 去除首尾空格
    text = text.strip()

    return text


def hash_string(text: str) -> str:
    """
    对字符串进行MD5哈希

    Args:
        text: 输入字符串

    Returns:
        MD5哈希值（十六进制）
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def format_file_size(size_bytes: int) -> str:
    """
    格式化文件大小

    Args:
        size_bytes: 文件大小（字节）

    Returns:
        格式化的文件大小字符串
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0

    return f"{size_bytes:.2f} TB"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    截断文本到指定长度

    Args:
        text: 原始文本
        max_length: 最大长度
        suffix: 截断后缀

    Returns:
        截断后的文本
    """
    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix
