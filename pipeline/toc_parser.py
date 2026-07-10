"""
TOC 解析器 — 检测技术标/商务标的分界页面

通过关键词匹配 + 段落位置分析，将 PDF 页面分类为:
  - "technical"   (技术标)
  - "commercial"  (商务标)
  - "unknown"     (无法判定)

用法:
    parser = TOCParser(paragraphs=texts, paragraph_page_nums=pages, total_pages=N)
    result = parser.parse()
    # → {"page_classifications": {0:"technical", ..., 10:"commercial"},
    #     "method": "keyword_boundary",
    #     "tech_start_page": 0,
    #     "com_start_page": 10}
"""

import re
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


# 技术标关键词（检测到即判为技术标页）
TECH_KEYWORDS = [
    "技术标", "技术部分", "技术方案", "施工组织设计",
    "施工方案", "施工组织", "技术措施", "主要施工方法",
    "工程概况", "项目概况", "总体概述", "项目管理",
    "质量保证", "安全保证", "工期保证", "施工部署",
    "施工进度", "资源配置", "主要机械", "劳动力安排",
]

# 商务标关键词（检测到即判为商务标页）
COM_KEYWORDS = [
    "商务标", "商务部分", "投标报价", "报价表", "工程量清单",
    "报价汇总", "投标总价", "单项报价", "分项报价",
    "报价文件", "商务文件", "价格文件", "报价书",
    "报价说明", "清单报价", "投标价格",
]

# 技术标章节号模式（如 "第一章" 在技术标关键词附近）
TECH_SECTION_PATTERN = re.compile(r'第[一二三四五六七八九十\d]+[章节篇]')


class TOCParser:
    """技术标/商务标分界解析器"""

    def __init__(
        self,
        paragraphs: List[str],
        paragraph_page_nums: List[int],
        total_pages: int,
    ):
        """
        Args:
            paragraphs: 文档段落文本列表
            paragraph_page_nums: 每段对应的页码（长度与 paragraphs 相同）
            total_pages: PDF 总页数
        """
        self.paragraphs = paragraphs
        self.page_nums = paragraph_page_nums
        self.total_pages = total_pages

    def parse(self) -> dict:
        """执行 TOC 解析

        Returns:
            dict: {
                "page_classifications": {page_num: "technical"|"commercial"|"unknown"},
                "method": str,           # 检测方法名称
                "tech_start_page": int,  # 技术标起始页
                "com_start_page": int,   # 商务标起始页（若无则 -1）
            }
        """
        classifications = {}
        method = "none"
        tech_start = 0
        com_start = -1

        # 方法1: 关键词边界检测
        result = self._detect_by_keyword_boundary()
        if result:
            classifications, tech_start, com_start = result
            method = "keyword_boundary"
            logger.debug(f"TOC: 关键词边界法 — com_start={com_start}")

        # 方法2: 关键词密度统计（回退）
        if not classifications:
            result = self._detect_by_keyword_density()
            if result:
                classifications, tech_start, com_start = result
                method = "keyword_density"

        # 如果仍无分类，标记全部为 unknown
        if not classifications:
            for p in range(self.total_pages):
                classifications[p] = "unknown"
            method = "none"

        # 确保所有页面都有标签
        for p in range(self.total_pages):
            if p not in classifications:
                classifications[p] = "unknown"

        return {
            "page_classifications": classifications,
            "method": method,
            "tech_start_page": tech_start,
            "com_start_page": com_start,
        }

    def _detect_by_keyword_boundary(self):
        """方法1: 关键词边界检测

        扫描段落找到第一个明确的商务标关键词出现位置，
        之前全部标记为 technical，之后全部标记为 commercial。
        """
        if not self.paragraphs:
            return None

        # 找到第一个商务标关键词出现的位置（段落级别）
        com_first_page = None

        for i, para in enumerate(self.paragraphs):
            text = para.strip().lower()
            if not text:
                continue
            for kw in COM_KEYWORDS:
                if kw.lower() in text:
                    # 检查这个词是否真的在段首附近（标题位置）
                    first_100 = text[:100]
                    if kw.lower() in first_100:
                        page_num = self.page_nums[i]
                        com_first_page = page_num
                        break
            if com_first_page is not None:
                break

        if com_first_page is None or com_first_page <= 0:
            # 没找到商务标关键词，或者就在第一页（异常），尝试其他方法
            return None

        # 取商务标前一页作为边界
        boundary_page = max(0, com_first_page - 1)
        tech_start = 0

        classifications = {}
        for p in range(self.total_pages):
            if p <= boundary_page:
                classifications[p] = "technical"
            else:
                classifications[p] = "commercial"

        logger.info(f"TOC 关键词边界: 第 {boundary_page} 页之前=技术标, 之后=商务标")
        return classifications, tech_start, com_first_page

    def _detect_by_keyword_density(self):
        """方法2: 关键词密度统计

        对每页统计 tech/com关键词命中数，按多数票决定分类。
        """
        page_scores = {}

        for i, para in enumerate(self.paragraphs):
            text = para.strip().lower()
            page_num = self.page_nums[i]
            if page_num < 0:
                continue
            if page_num not in page_scores:
                page_scores[page_num] = {"tech": 0, "com": 0}

            for kw in TECH_KEYWORDS:
                if kw.lower() in text:
                    page_scores[page_num]["tech"] += 1
            for kw in COM_KEYWORDS:
                if kw.lower() in text:
                    page_scores[page_num]["com"] += 1

        if not page_scores:
            return None

        classifications = {}
        tech_pages = []
        com_pages = []

        for p in range(self.total_pages):
            scores = page_scores.get(p, {"tech": 0, "com": 0})
            if scores["com"] > scores["tech"] and scores["com"] >= 1:
                classifications[p] = "commercial"
                com_pages.append(p)
            elif scores["tech"] > scores["com"] and scores["tech"] >= 1:
                classifications[p] = "technical"
                tech_pages.append(p)
            else:
                # 无明确关键词，暂不标记
                pass

        # 如果没有足够分类，放弃
        if len(classifications) < self.total_pages * 0.1:
            return None

        tech_start = min(tech_pages) if tech_pages else 0
        com_start = min(com_pages) if com_pages else -1

        # 填充未分类页面
        if com_pages:
            first_com = min(com_pages)
            for p in range(self.total_pages):
                if p not in classifications:
                    classifications[p] = "commercial" if p >= first_com else "technical"
        elif tech_pages:
            for p in range(self.total_pages):
                if p not in classifications:
                    classifications[p] = "technical"

        return classifications, tech_start, com_start
