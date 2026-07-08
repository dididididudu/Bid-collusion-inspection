"""
联系人指纹提取 — 从 PDF 全文中提取公司名称、联系人姓名、电话、邮箱

串标检测价值:
- 不同公司的标书出现相同的联系人 → 同一人操办 → 强串标信号
- 相同的公司名称出现在不同标书正文（非元数据）→ 模板复用
"""

import re
import logging
from typing import List, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ============================================================
# 预编译正则（类级别，避免每次调用重新编译）
# ============================================================

# 公司名称 — 匹配中文公司全称
# 后缀必须包含"公司"或"集团"，去掉"管理""服务""信息"等通用词以避免假阳性
_RE_COMPANY = re.compile(
    r'(?:[一-龥()（）]{2,30}(?:有限公司|股份有限公司|有限责任公司|'
    r'集团有限公司|集团公司|控股有限公司|实业有限公司|集团))'
)

# 联系人姓名 — 匹配角色标注后的人名
# 使用捕获组仅提取人名部分，去除"在""负"等尾部上下文
# 注意：不含"项目经理"（易与"负责整体"等动词短语混淆）
_RE_CONTACT_NAME = re.compile(
    r'(?:联系人|项目负责人|法定代表人|授权代表|技术负责人|投标人|'
    r'委托代理人|被授权人|经办人|签字人|投标代表|'
    r'技术总监|资深工程师)'
    r'[：:]?\s*([一-龥]{2,4})'
)

# 联系人姓名 — 表格格式：姓名 + 职务 + (可选手机号)
# 以手机号为锚点确保准确性，同时覆盖无序号的表格行
_RE_CONTACT_TABLE = re.compile(
    r'(?:^|\s)([一-龥]{2,4})\s+'
    r'(?:联系人|项目负责人|法定代表人|授权代表|技术负责人|投标人|'
    r'项目经理|技术总监|资深工程师|测试主管|产品经理|负责人)'
    r'(?:\s+(?:1[3-9]\d{9}))?'
)

# 手机号
_RE_MOBILE = re.compile(r'1[3-9]\d{9}')

# 固话（含区号）
_RE_PHONE = re.compile(r'(?:0\d{2,3}[)-]\d{7,8}|\d{3,4}-\d{7,8})')

# 邮箱
_RE_EMAIL = re.compile(r'[\w.\-+]+@[\w.\-]+\.[a-zA-Z]{2,}')

# 身份证号（后 4 位打码的也匹配）
_RE_ID_CARD = re.compile(r'\d{17}[\dXx]|\d{6}\*{8}\d{4}')

# 统一社会信用代码（18 位）
_RE_CREDIT_CODE = re.compile(r'[0-9A-HJ-NPQRTUWXY]{2}\d{6}[0-9A-HJ-NPQRTUWXY]{10}')

# 潜在人名上下文区域：含"联系方式"等关键词或手机号的文本段
_RE_NAME_CONTEXT = re.compile(
    r'(?:[^。\n]{0,80}'
    r'(?:联系方式|联系人|项目团队|团队成员|项目人员|技术团队|管理团队|'
    r'团队介绍|主要负责人|联.?系.?电.?话|联系电话|手机号码)'
    r'[^。\n]{0,80})'
    r'|'
    r'(?:[^。\n]{0,80}(?:1[3-9]\d{9})[^。\n]{0,80})'
)

# ============================================================
# 数据容器
# ============================================================

@dataclass
class ContactFingerprint:
    """文档的联系人指纹"""
    doc_id: str = ""
    company_names: List[str] = field(default_factory=list)   # 公司全称
    contact_names: List[str] = field(default_factory=list)   # 联系人姓名
    mobile_phones: List[str] = field(default_factory=list)   # 手机号
    landline_phones: List[str] = field(default_factory=list) # 固话
    emails: List[str] = field(default_factory=list)           # 邮箱
    credit_codes: List[str] = field(default_factory=list)    # 统一社会信用代码
    member_ids: List[str] = field(default_factory=list)     # 会员号（外部 API 注入）
    potential_names: List[str] = field(default_factory=list) # 宽松匹配潜在人名（无需前置标注词）

    def to_json(self) -> str:
        import json
        return json.dumps({
            'company_names': self.company_names,
            'contact_names': self.contact_names,
            'member_ids': self.member_ids,
            'mobile_phones': self.mobile_phones,
            'landline_phones': self.landline_phones,
            'emails': self.emails,
            'credit_codes': self.credit_codes,
            'potential_names': self.potential_names,
        }, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> 'ContactFingerprint':
        import json
        try:
            d = json.loads(raw)
        except Exception:
            d = {}
        return cls(
            company_names=d.get('company_names', []),
            contact_names=d.get('contact_names', []),
            mobile_phones=d.get('mobile_phones', []),
            landline_phones=d.get('landline_phones', []),
            emails=d.get('emails', []),
            credit_codes=d.get('credit_codes', []),
            member_ids=d.get('member_ids', []),
            potential_names=d.get('potential_names', []),
        )


# ============================================================
# 提取函数
# ============================================================

# 常见跟在人名后的虚词/动词（人名尾部误捕获时切除）
_CONTACT_NAME_TRAILING_TRIM = frozenset('的在是负有及与和这那了着负责整体项目进行')

# 在联系方式上下文中常见但不是人名的 2-4 字中文词
_POTENTIAL_NAME_STOPWORDS = frozenset({
    '联系方式', '联系人', '联系电话', '手机号码', '项目团队', '团队成员',
    '技术团队', '管理团队', '项目人员', '主要负责人', '联系地址', '邮政编码',
    '传真号码', '电子邮箱', '公司名称', '统一社会信用', '项目概况',
    '总体概述', '背景分析', '建设目标', '设计原则', '总体架构', '技术方案',
    '系统设计', '项目管理', '项目概况', '总体设计', '架构设计',
    '质量管理', '安全管理', '网络架构', '基础架构', '核心功能',
    '工程师', '技术员', '主管经理', '副总经理', '总经理',
})

# jieba 分词中可拆分成多个词的、明显不是人名的角色职务词
_ROLE_WORDS = frozenset({
    '工程师', '技术员', '主管', '经理', '总监', '总经理',
    '副总经理', '部长', '主任', '组长', '科长', '处长',
    '助理', '专员', '秘书', '顾问', '专家',
})

# 中文常见姓氏（前 100 个）
_COMMON_SURNAMES = frozenset(
    '王李张刘陈杨黄赵周吴徐孙马胡朱郭何罗高林'
    '郑梁谢宋唐韩曹许邓冯萧程蔡彭潘袁董余苏叶'
    '吕魏蒋田杜丁沈姜范江傅钟卢汪戴崔任陆廖姚'
    '方金邱夏谭石贾邹熊孟秦薛侯雷白龙段郝孔邵'
    '史毛常万顾赖武康贺严峻尹钱施牛洪龚程'
)

# 招标/业主类关键词：出现在公司名前→表明是招标方而非投标方
# 覆盖：招标公司、招标人、致：XX公司、业主单位等常见标书表述
_TENDER_PREFIX_KEYWORDS = frozenset({
    '招标人', '招标单位', '招标代理', '招标机构', '招标公司', '业主',
    '建设单位', '采购人', '采购单位', '项目业主', '发包人', '委托方',
    '甲方', '招标方', '项目招标人', '项目单位', '主管单位', '监督单位',
})


def _is_tendering_entity(text: str, match_start: int) -> bool:
    """检查公司名前 20 字内（同一行内）是否出现招标方关键词

    只检查 match_start 之前 20 字范围，且遇到换行符即停止，
    避免上一行的"招标人"污染当前行的"投标单位"。

    Args:
        text: 全文
        match_start: 公司名匹配的起始位置

    Returns:
        True 表示可能是招标方（非投标方），应排除
    """
    # 从 match_start 往前搜索，最多 20 字或遇到换行
    start = max(0, match_start - 20)
    before = text[start:match_start]
    # 如果跨行了，只取最后一个换行之后的内容
    last_newline = before.rfind('\n')
    if last_newline >= 0:
        before = before[last_newline + 1:]
    before_clean = before.replace(' ', '').replace('　', '').replace('：', ':')
    # 标准关键词匹配
    if any(kw in before_clean for kw in _TENDER_PREFIX_KEYWORDS):
        return True
    # 致 [：: ] + 公司名 → 投标函抬头格式，表明是收件方（招标方）
    import re
    if re.search(r'致[：: \t]', before):
        return True
    return False


def _clean_contact_name(raw: str) -> str:
    """清理捕获的人名：切除尾部虚词，保留有效人名部分

    Args:
        raw: 正则捕获组返回的原始字符串（如"张明远在""负责整体"）

    Returns:
        清理后的人名字符串（如"张明远"），无效返回空字符串
    """
    name = raw.strip()
    while len(name) > 2 and name[-1] in _CONTACT_NAME_TRAILING_TRIM:
        name = name[:-1]
    # 至少保留 2 个汉字
    return name if len(name) >= 2 and all('一' <= c <= '鿿' for c in name) else ''


def _extract_potential_names(text: str) -> List[str]:
    """从联系方式/手机号上下文提取潜在人名（宽松匹配，无需前置标注词）

    两篇文档中出现相同的不常见 2-4 字中文词 → 可能是共享联系人。
    策略：jieba 分词 → 保留 2-4 字中文词 → 姓氏合并（如"张"+"明远"="张明远"）
    """
    import jieba

    candidates = set()
    for m in _RE_NAME_CONTEXT.finditer(text):
        segment = m.group()
        words = [w.strip() for w in jieba.lcut(segment) if w.strip()]

        # Step 1: 相邻的 [姓氏 + 2字词] 合并为 3-4 字人名
        i = 0
        merged = []
        while i < len(words):
            w = words[i]
            # 如果当前是单字姓氏，且下一个词 1-3 字（纯中文），合并
            if (len(w) == 1 and w in _COMMON_SURNAMES
                    and i + 1 < len(words)
                    and len(words[i + 1]) in (1, 2, 3)
                    and all('一' <= c <= '鿿' for c in words[i + 1])):
                merged.append(w + words[i + 1])
                i += 2
            else:
                merged.append(w)
                i += 1

        # Step 2: 筛选可能的姓名
        for w in merged:
            w = w.strip()
            if len(w) < 2 or len(w) > 4:
                continue
            if not all('一' <= c <= '鿿' for c in w):
                continue
            if w in _POTENTIAL_NAME_STOPWORDS:
                continue
            if w in _ROLE_WORDS:
                continue
            if any(kw in w for kw in ['联系', '电话', '手机', '邮箱', '项目']):
                continue

            # 2 字词：必须包含姓氏才认为是人名（避免"技术""产品"等误报）
            if len(w) == 2 and w[0] not in _COMMON_SURNAMES:
                continue
            candidates.add(w)

    return sorted(candidates)


def extract_contacts_from_text(full_text: str) -> ContactFingerprint:
    """从文档全文提取联系人指纹

    Args:
        full_text: 文档完整文本（多页拼接）

    Returns:
        ContactFingerprint 对象
    """
    if not full_text:
        return ContactFingerprint()

    # 联系人名：两个正则（角色+姓名 / 表格格式）+ 清理 → 去重
    raw_names = _RE_CONTACT_NAME.findall(full_text) + _RE_CONTACT_TABLE.findall(full_text)
    cleaned_names = list(dict.fromkeys(
        n for n in (_clean_contact_name(r) for r in raw_names) if n
    ))

    # 宽松匹配：联系方式上下文中的潜在人名
    potential = _extract_potential_names(full_text)

    # 公司名：排除招标方（出现在"招标人："等关键词后的公司名）
    companies = []
    for m in _RE_COMPANY.finditer(full_text):
        if not _is_tendering_entity(full_text, m.start()):
            companies.append(m.group())

    return ContactFingerprint(
        company_names=list(set(companies)),
        contact_names=cleaned_names,
        mobile_phones=list(set(_RE_MOBILE.findall(full_text))),
        landline_phones=list(set(_RE_PHONE.findall(full_text))),
        emails=list(set(_RE_EMAIL.findall(full_text))),
        credit_codes=list(set(_RE_CREDIT_CODE.findall(full_text))),
        potential_names=potential,
    )


def extract_contacts_from_sqlite(doc_id: str, cache) -> ContactFingerprint:
    """从 SQLite 缓存中加载文档全文并提取联系人指纹

    使用 chunks 表的完整文本（压缩存储），而非 paragraphs 表的分段文本，
    避免短字段（联系人、电话等）被段落分割过滤掉。

    Args:
        doc_id: 文档 ID
        cache: DocumentCache 实例

    Returns:
        ContactFingerprint 对象
    """
    try:
        # 从 chunks 表加载完整文本（按 chunk_index 拼接）
        full_text = ''
        cursor = cache.conn.execute(
            "SELECT chunk_index FROM chunks WHERE doc_id = ? ORDER BY chunk_index",
            (doc_id,)
        )
        chunk_indices = [row[0] for row in cursor.fetchall()]
        if not chunk_indices:
            # 回退：从 paragraphs 表加载
            paragraphs = cache.load_all_paragraphs_text(doc_id)
            full_text = '\n'.join(paragraphs) if paragraphs else ''
        else:
            texts = []
            for ci in chunk_indices:
                chunk_text = cache.load_chunk_text(doc_id, ci)
                if chunk_text:
                    texts.append(chunk_text)
            full_text = '\n'.join(texts)
    except Exception:
        logger.warning(f"无法加载文档 {doc_id} 的段落文本")
        full_text = ''

    return extract_contacts_from_text(full_text)
