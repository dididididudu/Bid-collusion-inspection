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
        )


# ============================================================
# 提取函数
# ============================================================

# 常见跟在人名后的虚词/动词（人名尾部误捕获时切除）
_CONTACT_NAME_TRAILING_TRIM = frozenset('的在是负有及与和这那了着负责整体项目进行')

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


def extract_contacts_from_text(full_text: str) -> ContactFingerprint:
    """从文档全文提取联系人指纹

    Args:
        full_text: 文档完整文本（多页拼接）

    Returns:
        ContactFingerprint 对象
    """
    if not full_text:
        return ContactFingerprint()

    # 联系人名：正则带捕获组 → 清理尾部 → 去重
    raw_names = _RE_CONTACT_NAME.findall(full_text)
    cleaned_names = list(dict.fromkeys(
        n for n in (_clean_contact_name(r) for r in raw_names) if n
    ))

    return ContactFingerprint(
        company_names=list(set(_RE_COMPANY.findall(full_text))),
        contact_names=cleaned_names,
        mobile_phones=list(set(_RE_MOBILE.findall(full_text))),
        landline_phones=list(set(_RE_PHONE.findall(full_text))),
        emails=list(set(_RE_EMAIL.findall(full_text))),
        credit_codes=list(set(_RE_CREDIT_CODE.findall(full_text))),
    )


def extract_contacts_from_sqlite(doc_id: str, cache) -> ContactFingerprint:
    """从 SQLite 缓存中加载文档全文并提取联系人指纹

    Args:
        doc_id: 文档 ID
        cache: DocumentCache 实例

    Returns:
        ContactFingerprint 对象
    """
    # 从 chunks 表拼接全文（已有压缩存储，按需解压）
    try:
        paragraphs = cache.load_all_paragraphs_text(doc_id)
        full_text = '\n'.join(paragraphs) if paragraphs else ''
    except Exception:
        logger.warning(f"无法加载文档 {doc_id} 的段落文本")
        full_text = ''

    return extract_contacts_from_text(full_text)
