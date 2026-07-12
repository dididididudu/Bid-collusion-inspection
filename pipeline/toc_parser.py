"""
技术标/商务标页级分类器。

分类策略按可靠性从高到低执行：
1. 目录条目页码 + 正文标题锚点校准。
2. 正文标题锚点边界。
3. 页级关键词密度兜底。

输出 page_classifications 使用 PDF 物理页码（0-based），供技术标/商务标
两个接口分别过滤段落和图片，避免内容相似度检测互相污染。
"""

import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


TECH_KEYWORDS = [
    "技术标", "技术部分", "技术方案", "施工组织设计",
    "施工方案", "施工组织", "技术措施", "主要施工方法",
    "工程概况", "项目概况", "总体概述", "项目管理",
    "质量保证", "安全保证", "工期保证", "施工部署",
    "施工进度", "资源配置", "主要机械", "劳动力安排",
    "项目概述", "项目背景", "需求分析", "功能需求分析",
    "系统设计", "实施方案", "测试方案", "售后服务及培训方案",
]

COM_KEYWORDS = [
    "商务标", "商务部分", "投标报价", "报价表", "工程量清单",
    "报价汇总", "投标总价", "单项报价", "分项报价",
    "报价文件", "商务文件", "价格文件", "报价书",
    "报价说明", "清单报价", "投标价格", "投标函",
    "投标书", "授权委托书", "开标一览表", "投标分项报价表",
    "合同条款偏离表", "采购需求偏离表", "资格证明资料",
    "投标保证金", "业绩合同", "中标服务费承诺书",
]

TECH_STRONG = [
    "技术标", "技术部分", "技术方案", "施工组织设计",
    "项目概述", "项目背景", "功能需求分析", "系统设计",
]
COM_STRONG = [
    "商务标", "商务部分", "投标报价", "工程量清单", "报价文件",
    "报价书", "投标书", "开标一览表", "投标分项报价表",
    "资格证明资料", "授权委托书",
]

_TRAILING_PAGE_RE = re.compile(
    r"(?P<title>[\u4e00-\u9fffA-Za-z0-9（）()《》、：:\-\s]{2,80}?)"
    r"(?:[\.·•…\s_—-]{2,}|第?\s*)"
    r"(?P<page>\d{1,4})\s*页?\s*$"
)
_TOC_NOISE_RE = re.compile(r"[\.·•…_]{2,}")


class TOCParser:
    """将 PDF 页分类为 technical/commercial/unknown。"""

    def __init__(
        self,
        paragraphs: List[str],
        paragraph_page_nums: List[int],
        total_pages: int,
    ):
        self.paragraphs = paragraphs
        self.page_nums = paragraph_page_nums
        self.total_pages = max(0, total_pages)
        self.front_limit = min(self.total_pages, max(8, int(self.total_pages * 0.12)))
        self.page_texts = self._build_page_texts()

    def parse(self) -> dict:
        classifications = {}
        method = "none"
        confidence = 0.0
        tech_start = 0
        com_start = -1

        result = self._detect_by_toc_entries()
        if result:
            classifications, tech_start, com_start, confidence = result
            method = "toc_entries"

        if not classifications:
            result = self._detect_by_body_anchors()
            if result:
                classifications, tech_start, com_start, confidence = result
                method = "body_anchors"

        if not classifications:
            result = self._detect_by_keyword_density()
            if result:
                classifications, tech_start, com_start, confidence = result
                method = "keyword_density"

        if not classifications:
            classifications = {p: "unknown" for p in range(self.total_pages)}

        for p in range(self.total_pages):
            classifications.setdefault(p, "unknown")

        return {
            "page_classifications": classifications,
            "method": method,
            "confidence": confidence,
            "tech_start_page": tech_start,
            "com_start_page": com_start,
        }

    def _build_page_texts(self) -> Dict[int, str]:
        page_texts = defaultdict(list)
        for para, page_num in zip(self.paragraphs, self.page_nums):
            if 0 <= page_num < self.total_pages and para:
                page_texts[page_num].append(str(para))
        return {p: "\n".join(parts) for p, parts in page_texts.items()}

    @staticmethod
    def _label_for_text(text: str) -> Optional[str]:
        if any(kw in text for kw in COM_KEYWORDS):
            return "commercial"
        if any(kw in text for kw in TECH_KEYWORDS):
            return "technical"
        return None

    def _is_toc_like_page(self, page_num: int, text: str) -> bool:
        if page_num >= self.front_limit:
            return False
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if "目录" in text[:200]:
            return True
        if not lines:
            return False
        tocish = 0
        for line in lines:
            if _TOC_NOISE_RE.search(line) or _TRAILING_PAGE_RE.search(line):
                tocish += 1
        return tocish >= max(2, len(lines) // 4)

    def _extract_toc_entries(self) -> List[Tuple[str, int, int, str]]:
        entries = []
        for page_num in range(self.front_limit):
            text = self.page_texts.get(page_num, "")
            if not text or not self._is_toc_like_page(page_num, text):
                continue
            candidates = []
            for para in text.splitlines():
                candidates.extend(seg.strip() for seg in re.split(r"[\n\r]", para))
            candidates.append(text[:400])
            for line in candidates:
                if not line:
                    continue
                m = _TRAILING_PAGE_RE.search(line)
                if not m:
                    continue
                title = m.group("title").strip()
                label = self._label_for_text(title)
                if not label:
                    continue
                printed_page = int(m.group("page"))
                if 1 <= printed_page <= self.total_pages + 50:
                    entries.append((label, printed_page, page_num, line[:120]))
        return entries

    def _find_body_title_anchors(self) -> Dict[str, List[int]]:
        anchors = {"technical": [], "commercial": []}
        toc_pages = {
            p for p, text in self.page_texts.items()
            if self._is_toc_like_page(p, text)
        }
        for para, page_num in zip(self.paragraphs, self.page_nums):
            if page_num in toc_pages or page_num < 0:
                continue
            text = str(para).strip()
            if not text:
                continue
            head = re.sub(r"\s+", "", text[:120])
            is_title_like = len(head) <= 80 or head.startswith(("第", "一、", "二、", "三、"))
            if not is_title_like:
                continue
            if any(kw in head for kw in COM_STRONG):
                anchors["commercial"].append(page_num)
            elif any(kw in head for kw in TECH_STRONG):
                anchors["technical"].append(page_num)
        return {k: sorted(set(v)) for k, v in anchors.items()}

    def _detect_by_toc_entries(self):
        entries = self._extract_toc_entries()
        if not entries:
            return None

        tech_prints = [p for label, p, _src, _line in entries if label == "technical"]
        com_prints = [p for label, p, _src, _line in entries if label == "commercial"]
        if not tech_prints and not com_prints:
            return None

        anchors = self._find_body_title_anchors()
        com_start = -1
        tech_start = 0
        confidence = 0.72

        com_anchor = min(anchors["commercial"]) if anchors["commercial"] else None
        tech_anchor = min(anchors["technical"]) if anchors["technical"] else None

        if tech_prints:
            tech_start = tech_anchor if tech_anchor is not None else self._printed_to_pdf_page(min(tech_prints))
        elif anchors["technical"]:
            tech_start = tech_anchor

        if com_prints:
            com_start = com_anchor if com_anchor is not None else self._printed_to_pdf_page(min(com_prints))
        elif anchors["commercial"]:
            com_start = com_anchor

        if tech_start >= 0 and com_start >= 0 and tech_start != com_start:
            if tech_start < com_start:
                classifications = self._split_two_sections("technical", com_start)
            else:
                classifications = self._split_two_sections("commercial", tech_start)
            confidence = 0.92 if (tech_anchor is not None or com_anchor is not None) else confidence
            logger.info(
                f"TOC 目录/正文联合分类: tech_start={tech_start}, "
                f"commercial_start={com_start}, entries={len(entries)}, "
                f"confidence={confidence:.2f}"
            )
            return classifications, tech_start, com_start, confidence

        if com_start <= 0 or com_start >= self.total_pages:
            # 只有技术目录的文件按全技术标处理；只有商务目录则按全商务标处理。
            if tech_prints or anchors["technical"]:
                return self._single_dimension("technical", tech_start, confidence)
            if com_prints or anchors["commercial"]:
                return self._single_dimension("commercial", 0, confidence)
            return None

        classifications = self._split_two_sections("technical", com_start)
        logger.info(
            f"TOC 目录页码分类: commercial_start={com_start}, "
            f"entries={len(entries)}, confidence={confidence:.2f}"
        )
        return classifications, tech_start, com_start, confidence

    def _printed_to_pdf_page(self, printed_page: int) -> int:
        # 无法校准时使用最保守映射：印刷页 1 -> PDF 第 0 页。
        return max(0, min(self.total_pages - 1, printed_page - 1))

    def _detect_by_body_anchors(self):
        anchors = self._find_body_title_anchors()
        tech_start = min(anchors["technical"]) if anchors["technical"] else -1
        com_start = min(anchors["commercial"]) if anchors["commercial"] else -1
        if tech_start >= 0 and com_start >= 0 and tech_start != com_start:
            if tech_start < com_start:
                classifications = self._split_two_sections("technical", com_start)
            else:
                classifications = self._split_two_sections("commercial", tech_start)
            logger.info(
                f"TOC 正文标题分类: tech_start={tech_start}, commercial_start={com_start}"
            )
            return classifications, tech_start, com_start, 0.84
        if anchors["technical"] and not anchors["commercial"]:
            return self._single_dimension("technical", min(anchors["technical"]), 0.70)
        if anchors["commercial"] and not anchors["technical"]:
            return self._single_dimension("commercial", min(anchors["commercial"]), 0.70)
        return None

    def _detect_by_keyword_density(self):
        page_scores = {}
        for page_num, text in self.page_texts.items():
            compact = re.sub(r"\s+", "", text)
            tech = sum(compact.count(kw) for kw in TECH_KEYWORDS)
            com = sum(compact.count(kw) for kw in COM_KEYWORDS)
            if tech or com:
                page_scores[page_num] = {"tech": tech, "com": com}

        if not page_scores:
            return None

        labeled = {}
        for p, scores in page_scores.items():
            if scores["com"] > scores["tech"]:
                labeled[p] = "commercial"
            elif scores["tech"] > scores["com"]:
                labeled[p] = "technical"

        if len(labeled) < max(1, self.total_pages * 0.02):
            return None

        com_pages = [p for p, label in labeled.items() if label == "commercial"]
        tech_pages = [p for p, label in labeled.items() if label == "technical"]
        if com_pages and tech_pages:
            com_start = min(com_pages)
            tech_start = min(tech_pages)
            if tech_start < com_start:
                classifications = self._split_two_sections("technical", com_start)
            else:
                classifications = self._split_two_sections("commercial", tech_start)
            return classifications, tech_start, com_start, 0.58
        if tech_pages and not com_pages:
            return self._single_dimension("technical", min(tech_pages), 0.55)
        if com_pages and not tech_pages:
            return self._single_dimension("commercial", min(com_pages), 0.55)
        return None

    def _split_two_sections(self, first_label: str, second_start: int) -> Dict[int, str]:
        second_label = "commercial" if first_label == "technical" else "technical"
        return {
            p: (first_label if p < second_start else second_label)
            for p in range(self.total_pages)
        }

    def _single_dimension(
        self, label: str, start_page: int, confidence: float
    ) -> Tuple[Dict[int, str], int, int, float]:
        classifications = {p: label for p in range(self.total_pages)}
        tech_start = start_page if label == "technical" else -1
        com_start = start_page if label == "commercial" else -1
        logger.info(f"TOC 单维度分类: {label}, start={start_page}, confidence={confidence:.2f}")
        return classifications, tech_start, com_start, confidence
