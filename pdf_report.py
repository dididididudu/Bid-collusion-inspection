"""
PDF 报告生成 — 投标文件串标围标检测报告（专业版）
视觉设计：仪表盘风格封面、卡片化维度展示、进度条、双栏文本对比
"""
import os, io, base64, logging
from typing import Dict, Optional
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import HexColor, white, black, Color
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, Image, Frame, PageTemplate, BaseDocTemplate
)
from reportlab.pdfgen import canvas
from data_structures import GlobalReport, PairwiseResult

logger = logging.getLogger(__name__)

# ================================================================
# 色彩系统
# ================================================================
C_PRIMARY   = HexColor('#1e3a5f')   # 深蓝 — 标题/页眉
C_SECONDARY = HexColor('#4a90a4')   # 青蓝 — 次级标题
C_ACCENT    = HexColor('#e8f0f8')   # 浅蓝 — 卡片交替行
C_RISK_HIGH = HexColor('#e74c3c')   # 红色 — 高风险
C_RISK_MED  = HexColor('#f39c12')   # 橙色 — 中风险
C_SAFE      = HexColor('#27ae60')   # 绿色 — 正常
C_BG        = HexColor('#f8f9fa')   # 浅灰 — 卡片底色
C_TEXT      = HexColor('#2c3e50')   # 正文
C_MUTED     = HexColor('#7f8c8d')   # 辅助文字
C_HIGHLIGHT = HexColor('#fff3cd')   # 高亮黄
C_BORDER    = HexColor('#dce1e8')   # 边框灰

# ================================================================
# 字体注册
# ================================================================
_FONT_DIR = "C:/Windows/Fonts"
_FONTS_OK = False

def _reg_fonts():
    global _FONTS_OK
    if _FONTS_OK: return
    for name, fname, fallback in [
        ('Hei', 'msyh.ttc', 'SimHei'), ('Hei-Bd', 'msyhbd.ttc', 'SimHei'),
        ('Song', 'simsun.ttc', 'SimSun'), ('Song-Bd', 'simsunb.ttf', 'SimSun'),
    ]:
        try:
            p = os.path.join(_FONT_DIR, fname)
            if os.path.exists(p):
                pdfmetrics.registerFont(TTFont(name, p))
            else:
                p2 = os.path.join(_FONT_DIR, fallback)
                if os.path.exists(p2):
                    pdfmetrics.registerFont(TTFont(name, p2))
        except:
            pass
    from reportlab.lib.fonts import addMapping
    addMapping('Hei', 0, 0, 'Hei'); addMapping('Hei', 1, 0, 'Hei-Bd')
    addMapping('Hei-Bd', 0, 0, 'Hei-Bd'); addMapping('Hei-Bd', 1, 0, 'Hei-Bd')
    _FONTS_OK = True

# ================================================================
# 段落样式
# ================================================================
def _sty():
    _reg_fonts()
    return {
        'cover_title': ParagraphStyle('ct', fontName='Hei-Bd', fontSize=26, leading=32,
            textColor=C_PRIMARY, alignment=TA_CENTER, spaceAfter=4*mm),
        'cover_id': ParagraphStyle('ci', fontName='Hei', fontSize=8, leading=11,
            textColor=C_MUTED, alignment=TA_RIGHT),
        'stat_num': ParagraphStyle('sn', fontName='Hei-Bd', fontSize=26, leading=30,
            textColor=C_PRIMARY, alignment=TA_CENTER),
        'stat_label': ParagraphStyle('sl', fontName='Hei', fontSize=9, leading=13,
            textColor=C_TEXT, alignment=TA_CENTER),
        'section': ParagraphStyle('se', fontName='Hei-Bd', fontSize=16, leading=22,
            textColor=C_PRIMARY, spaceBefore=8*mm, spaceAfter=2*mm),
        'subsection': ParagraphStyle('us', fontName='Hei-Bd', fontSize=11, leading=16,
            textColor=C_SECONDARY, spaceBefore=5*mm, spaceAfter=2*mm),
        'pair_h': ParagraphStyle('ph', fontName='Hei-Bd', fontSize=10, leading=14,
            textColor=C_PRIMARY, spaceBefore=3*mm, spaceAfter=2*mm),
        'body': ParagraphStyle('bo', fontName='Hei', fontSize=9, leading=14,
            textColor=C_TEXT, spaceAfter=2*mm),
        'muted': ParagraphStyle('mu', fontName='Hei', fontSize=7, leading=10,
            textColor=C_MUTED),
        'tag': ParagraphStyle('ta', fontName='Hei', fontSize=7, leading=10,
            textColor=C_SECONDARY, backColor=C_ACCENT, borderPadding=2),
        'common_label': ParagraphStyle('cl', fontName='Hei-Bd', fontSize=8, leading=12,
            textColor=HexColor('#856404'), spaceBefore=1*mm),
        'common_text': ParagraphStyle('ctext', fontName='Song', fontSize=8, leading=12,
            textColor=HexColor('#856404'), leftIndent=8*mm, spaceAfter=1*mm,
            backColor=C_HIGHLIGHT, borderPadding=3),
        'footer': ParagraphStyle('ft', fontName='Hei', fontSize=7, leading=9,
            textColor=C_MUTED, alignment=TA_CENTER),
    }

# ================================================================
# 辅助工具
# ================================================================
def _esc(t):
    return str(t).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')

def _thumb(b64, mw=180):
    if not b64: return None
    try:
        if ',' in b64: b64 = b64.split(',',1)[1]
        buf = io.BytesIO(base64.b64decode(b64))
        from PIL import Image as PILImage
        pi = PILImage.open(buf); w, h = pi.size
        if w > mw: r = mw/w; w, h = mw, h*r
        from reportlab.platypus import Image as RLImage
        return RLImage(buf, width=w, height=h)
    except: return None

def _color_for_sim(sim):
    if sim >= 0.8: return C_RISK_HIGH
    if sim >= 0.6: return C_RISK_MED
    return C_SAFE

# ================================================================
# 页眉页脚
# ================================================================
_HEADER_TEXT = "投标文件串标围标检测报告"

def _header_footer(cv, doc):
    """画页眉页脚（封面不画）"""
    w, h = A4
    page = doc.page
    if page == 1: return  # 封面无页眉页脚
    cv.saveState()
    # 页眉线
    cv.setStrokeColor(C_PRIMARY)
    cv.setLineWidth(0.5)
    cv.line(2.5*cm, h-1.5*cm, w-2.5*cm, h-1.5*cm)
    # 页眉左
    cv.setFont('Hei', 8)
    cv.setFillColor(C_PRIMARY)
    cv.drawString(2.5*cm, h-1.3*cm, _HEADER_TEXT)
    # 页眉右（页码）
    cv.setFillColor(C_MUTED)
    cv.drawRightString(w-2.5*cm, h-1.3*cm, f"第 {page-1} 页")
    # 页脚
    cv.setFont('Hei', 7)
    cv.setFillColor(C_MUTED)
    cv.drawCentredString(w/2, 1*cm, "本报告由系统自动生成，结果仅供参考")
    cv.restoreState()

# ================================================================
# 封面页
# ================================================================
def _build_cover(story, s, report):
    """封面页 — 仪表盘风格"""
    story.append(Spacer(1, 20*mm))
    # 标题
    story.append(Paragraph('投标文件串标围标检测报告', s['cover_title']))
    story.append(Paragraph('Bid Collusion Inspection Report',
        ParagraphStyle('en', fontName='Hei', fontSize=11, leading=15,
            textColor=C_SECONDARY, alignment=TA_CENTER, spaceAfter=15*mm)))

    # 统计卡片（3 个横向排列）
    items = [
        ('检测文件数', str(report.total_files), C_PRIMARY),
        ('比对总对数', str(report.total_pairs), C_SECONDARY),
        ('有雷同项对数', str(report.suspicious_pairs), C_RISK_HIGH),
    ]
    cards = []
    for lbl, val, color in items:
        card = Table([
            [Paragraph(val, ParagraphStyle('cv', fontName='Hei-Bd', fontSize=28,
                leading=34, textColor=color, alignment=TA_CENTER))],
            [Paragraph(lbl, s['stat_label'])],
        ], colWidths=[50*mm])
        card.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), C_BG),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (0,0), 8*mm),
            ('BOTTOMPADDING', (0,-1), (-1,-1), 5*mm),
            ('LEFTPADDING', (0,0), (-1,-1), 5*mm),
            ('RIGHTPADDING', (0,0), (-1,-1), 5*mm),
            ('LEFTPADDING', (0,0), (-1,-1), 5*mm),
            ('LINEBELOW', (0,0), (-1,0), 3, color),
        ]))
        cards.append(card)

    card_table = Table([cards], colWidths=[55*mm]*3, hAlign='CENTER')
    card_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('LEFTPADDING', (0,0), (-1,-1), 3*mm),
        ('RIGHTPADDING', (0,0), (-1,-1), 3*mm),
    ]))
    story.append(card_table)

    story.append(Spacer(1, 12*mm))

    # 涉嫌围标团伙色块
    if report.risk_clusters:
        from reportlab.platypus import HRFlowable
        for cl in report.risk_clusters:
            warn_text = f"⚠ 发现涉嫌围标团伙: {len(cl.doc_ids)} 份文件 ({cl.cluster_type})"
            warn = Table(
                [[Paragraph(warn_text, ParagraphStyle('warn', fontName='Hei-Bd',
                    fontSize=11, textColor=C_RISK_HIGH, alignment=TA_CENTER))]],
                colWidths=[160*mm], hAlign='CENTER')
            warn.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), HexColor('#fdf0f0')),
                ('LINELEFT', (0,0), (0,0), 3, C_RISK_HIGH),
                ('TOPPADDING', (0,0), (-1,-1), 4*mm),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4*mm),
            ]))
            story.append(warn)

    story.append(Spacer(1, 15*mm))

    # 报告元信息
    info_style = ParagraphStyle('mi', fontName='Hei', fontSize=8, leading=12,
        textColor=C_MUTED, alignment=TA_CENTER)
    story.append(Paragraph(f"报告ID: {report.report_id}", info_style))
    story.append(Paragraph(f"生成时间: {report.generated_at}", info_style))

    story.append(Spacer(1, 20*mm))
    story.append(Paragraph("本报告由自动检测系统生成，结果仅供参考，请结合人工审核确认",
        ParagraphStyle('disc', fontName='Hei', fontSize=7, leading=10,
            textColor=C_MUTED, alignment=TA_CENTER)))
    story.append(PageBreak())


# ================================================================
# 简单维度章节（元数据、联系人等）
# ================================================================
def _has_dim(pair, key):
    e = pair.evidence
    if key == 'file_id':       return e.metadata_evidence.same_file_id
    if key == 'author':        return 'author' in e.metadata_evidence.matched_fields
    if key == 'editor':        return any(f in e.metadata_evidence.matched_fields for f in ['creator', 'producer', 'software_fingerprint'])
    if key == 'contact':       return bool(e.contact_evidence.common_mobiles or e.contact_evidence.common_emails or e.contact_evidence.common_contacts)
    if key == 'company_name':  return bool(e.contact_evidence.common_companies)
    if key == 'credit_code':   return bool(e.contact_evidence.common_credit_codes)
    if key == 'text_sim':      return bool(e.text_evidence.paragraph_matches)
    if key == 'image_sim':     return bool(e.image_evidence.common_image_hashes or e.image_evidence.text_identical_count > 0)
    return False

def _dim_detail(pair, key):
    e = pair.evidence; r = []
    if key == 'file_id' and e.metadata_evidence.same_file_id:
        r.append('两份 PDF 文件码 /ID[0] 相同，从同一源文件生成')
    elif key == 'author' and 'author' in e.metadata_evidence.matched_fields:
        r.append(f"作者相同: {e.metadata_evidence.matched_values.get('author','')}")
    elif key == 'editor':
        for f in ['creator','producer','software_fingerprint']:
            if f in e.metadata_evidence.matched_fields:
                r.append(f"{f} 相同: {e.metadata_evidence.matched_values.get(f,'')}")
    elif key == 'contact':
        c = e.contact_evidence
        if c.common_mobiles:   r.append(f"相同手机号: {'、'.join(c.common_mobiles[:3])}")
        if c.common_emails:    r.append(f"相同邮箱: {'、'.join(c.common_emails[:3])}")
        if c.common_contacts:  r.append(f"相同联系人: {'、'.join(c.common_contacts[:3])}")
    elif key == 'company_name' and e.contact_evidence.common_companies:
        r.append(f"相同公司名: {'、'.join(e.contact_evidence.common_companies[:3])}")
    elif key == 'credit_code' and e.contact_evidence.common_credit_codes:
        r.append(f"统一社会信用代码相同: {'、'.join(e.contact_evidence.common_credit_codes[:3])}")
    return r

def _simple_dim(story, s, dim_key, dim_title, pairs, profiles):
    story.append(Paragraph(dim_title, s['section']))
    # 标题下装饰线
    story.append(Table([['']], colWidths=[40*mm], rowHeights=[2]))
    story[-1].setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), C_PRIMARY),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(Spacer(1, 4*mm))

    for pair in pairs:
        fa = _esc(getattr(profiles.get(pair.doc_a_id,{}),'filename',pair.doc_a_id))
        fb = _esc(getattr(profiles.get(pair.doc_b_id,{}),'filename',pair.doc_b_id))
        details = _dim_detail(pair, dim_key)

        # 卡片
        card_data = [[Paragraph(f'<b>{fa}</b> ↔ <b>{fb}</b>', s['pair_h'])]]
        for det in details:
            card_data.append([Paragraph(f'<font color="{C_RISK_HIGH.hexval()}">●</font> {_esc(det)}', s['body'])])
        card = Table(card_data, colWidths=[160*mm], hAlign='CENTER')
        card.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), C_BG),
            ('TOPPADDING', (0,0), (-1,0), 3*mm),
            ('BOTTOMPADDING', (0,-1), (-1,-1), 3*mm),
            ('LEFTPADDING', (0,0), (-1,-1), 4*mm),
            ('RIGHTPADDING', (0,0), (-1,-1), 4*mm),
            ('LINEBELOW', (0,0), (-1,0), 0.5, C_BORDER),
            ('ROUNDEDCORNERS', [2,2,2,2]),
        ]))
        story.append(KeepTogether(card))
        story.append(Spacer(1, 3*mm))

    story.append(Spacer(1, 6*mm))


# ================================================================
# 进度条绘制
# ================================================================
class ProgressBar:
    """水平进度条 flowable"""
    def __init__(self, value, width=120, height=8, color=None):
        self.value = max(0, min(1, value))
        self._w = width
        self._h = height
        self._color = color or _color_for_sim(value)
        self.width = width + 30  # 预留文字空间
        self.height = height + 4

    def wrap(self, availWidth, availHeight):
        return (self.width, self.height)

    def draw(self):
        c = self.canv
        bar_w = self._w; bar_h = self._h
        fill_w = bar_w * self.value
        y0 = 2
        # 背景条
        c.setFillColor(C_BG)
        c.setStrokeColor(C_BORDER)
        c.setLineWidth(0.5)
        c.roundRect(0, y0, bar_w, bar_h, 3, fill=1, stroke=1)
        # 填充条
        if fill_w > 1:
            c.setFillColor(self._color)
            c.roundRect(0, y0, fill_w, bar_h, 3, fill=1, stroke=0)
        # 百分比
        c.setFont('Hei', 7)
        c.setFillColor(C_TEXT)
        c.drawString(bar_w + 5, y0, f"{self.value*100:.0f}%")


# ================================================================
# 文本相似度章节
# ================================================================
def _text_sim(story, s, pairs, profiles):
    story.append(Paragraph('内容相似度 — 文本', s['section']))
    story.append(Table([['']], colWidths=[40*mm], rowHeights=[2]))
    story[-1].setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,-1),C_PRIMARY),
        ('BOTTOMPADDING',(0,0),(-1,-1),0),('TOPPADDING',(0,0),(-1,-1),0),
    ]))
    story.append(Spacer(1, 4*mm))

    for pair in pairs:
        te = pair.evidence.text_evidence
        if not te.paragraph_matches: continue
        fa = _esc(getattr(profiles.get(pair.doc_a_id,{}),'filename',pair.doc_a_id))
        fb = _esc(getattr(profiles.get(pair.doc_b_id,{}),'filename',pair.doc_b_id))
        sim = pair.similarity_scores.get('text_local', 0)

        # --- Pair 标题 + 相似度进度条 ---
        bar = ProgressBar(sim, width=100, height=7)
        title_data = [
            [Paragraph(f'<b>{fa}</b> <font color="{C_MUTED.hexval()}">↔</font> <b>{fb}</b>', s['pair_h']),
             bar]
        ]
        title_tbl = Table(title_data, colWidths=[110*mm, 50*mm], hAlign='CENTER')
        title_tbl.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('LEFTPADDING', (0,0), (0,0), 4*mm),
            ('RIGHTPADDING', (-1,-1), (-1,-1), 2*mm),
        ]))
        story.append(title_tbl)

        # 摘要
        story.append(Paragraph(
            f"匹配段落: {len(te.paragraph_matches)} 对 | "
            f"连续克隆块: {len(te.continuous_clone_blocks)} 个",
            s['muted']))

        # 克隆块
        for block in te.continuous_clone_blocks:
            story.append(Paragraph(
                f'<font color="{C_RISK_HIGH.hexval()}">⚠</font> '
                f'连续克隆块: {block.get("length",0)} 段连续, '
                f'平均相似度 {block.get("similarity",0):.4f}',
                ParagraphStyle('cb', fontName='Hei-Bd', fontSize=8, leading=12,
                    textColor=C_RISK_HIGH, spaceBefore=2*mm, leftIndent=4*mm)))

        # --- 逐对匹配卡片 ---
        for match in te.paragraph_matches[:30]:
            ia = match.get('paragraph_a_index','?')
            ib = match.get('paragraph_b_index','?')
            pa = f'（第{match.get("page_num_a",-1)+1}页）' if match.get('page_num_a',-1)>=0 else ''
            pb = f'（第{match.get("page_num_b",-1)+1}页）' if match.get('page_num_b',-1)>=0 else ''
            sm = match.get('similarity', 0)
            method = match.get('detection_method','')
            clr = _color_for_sim(sm)

            # 卡片头部: 位置 + 相似度标签
            loc = f'<b>A</b> 第[{ia}]段{pa} ↔ <b>B</b> 第[{ib}]段{pb}'
            tag = f'<font color="{clr.hexval()}">■</font> {sm:.4f} | {_esc(method)}'
            header = [
                [Paragraph(loc, ParagraphStyle('loc', fontName='Hei', fontSize=8,
                    leading=12, textColor=C_TEXT)),
                 Paragraph(tag, ParagraphStyle('tag', fontName='Hei', fontSize=7,
                    leading=10, textColor=C_MUTED, alignment=TA_RIGHT))]
            ]

            # A/B 原文左右分栏
            ta = _esc(match.get('paragraph_a',''))
            tb = _esc(match.get('paragraph_b',''))
            if len(ta) > 400: ta = ta[:400] + '…[下略]'
            if len(tb) > 400: tb = tb[:400] + '…[下略]'
            texts = [
                [Paragraph(f'<font color="{C_PRIMARY.hexval()}">A 原文</font>',
                     ParagraphStyle('la', fontName='Hei-Bd', fontSize=7, leading=10, textColor=C_PRIMARY)),
                 Paragraph(f'<font color="{C_PRIMARY.hexval()}">B 原文</font>',
                     ParagraphStyle('lb', fontName='Hei-Bd', fontSize=7, leading=10, textColor=C_PRIMARY))],
                [Paragraph(f'<font face="Song" size="8">{ta}</font>',
                     ParagraphStyle('ta', fontName='Song', fontSize=8, leading=12)),
                 Paragraph(f'<font face="Song" size="8">{tb}</font>',
                     ParagraphStyle('tb', fontName='Song', fontSize=8, leading=12))],
            ]
            text_tbl = Table(texts, colWidths=[80*mm, 80*mm], hAlign='CENTER')
            text_tbl.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (0,0), HexColor('#eef5fb')),
                ('BACKGROUND', (1,0), (1,0), HexColor('#fdf5f0')),
                ('BACKGROUND', (0,1), (0,1), HexColor('#f5f9fc')),
                ('BACKGROUND', (1,1), (1,1), HexColor('#fef9f5')),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('TOPPADDING', (0,0), (-1,0), 2*mm),
                ('BOTTOMPADDING', (0,-1), (-1,-1), 2*mm),
                ('LEFTPADDING', (0,0), (-1,-1), 3*mm),
                ('RIGHTPADDING', (0,0), (-1,-1), 3*mm),
                ('GRID', (0,0), (-1,-1), 0.3, C_BORDER),
            ]))

            # 共同部分
            common = match.get('common_parts', [])
            common_rows = []
            if common:
                clen = sum(len(p) for p in common)
                common_rows.append([Paragraph(
                    f'<font color="{HexColor("#856404").hexval()}">共同部分（{len(common)} 处, {clen} 字）:</font>',
                    s['common_label'])])
                for ci, part in enumerate(common, 1):
                    pt = _esc(part.strip()[:200])
                    common_rows.append([Paragraph(
                        f'<font face="Song" size="8">{pt}</font>',
                        s['common_text'])])

            # 组装卡片
            card_parts = header + [[text_tbl]] + common_rows
            card = Table(card_parts, colWidths=[160*mm], hAlign='CENTER')
            card.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), white),
                ('TOPPADDING', (0,0), (-1,0), 2*mm),
                ('BOTTOMPADDING', (0,0), (-1,0), 2*mm),
                ('LEFTPADDING', (0,0), (-1,-1), 3*mm),
                ('RIGHTPADDING', (0,0), (-1,-1), 3*mm),
                ('BOX', (0,0), (-1,-1), 0.5, C_BORDER),
                ('LINEBELOW', (0,0), (-1,0), 0.3, C_BORDER),
            ]))
            story.append(KeepTogether(card))
            story.append(Spacer(1, 3*mm))

    story.append(Spacer(1, 6*mm))


# ================================================================
# 图片相似度章节
# ================================================================
def _image_sim(story, s, pairs, profiles):
    story.append(Paragraph('内容相似度 — 图片', s['section']))
    story.append(Table([['']], colWidths=[40*mm], rowHeights=[2]))
    story[-1].setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,-1),C_PRIMARY),
        ('BOTTOMPADDING',(0,0),(-1,-1),0),('TOPPADDING',(0,0),(-1,-1),0),
    ]))
    story.append(Spacer(1, 4*mm))

    for pair in pairs:
        ie = pair.evidence.image_evidence
        has_img = bool(ie.matched_image_pairs or ie.common_image_hashes or ie.text_identical_count>0)
        if not has_img and ie.shared_typo_count==0 and not ie.ps_suspicious: continue
        fa = _esc(getattr(profiles.get(pair.doc_a_id,{}),'filename',pair.doc_a_id))
        fb = _esc(getattr(profiles.get(pair.doc_b_id,{}),'filename',pair.doc_b_id))
        story.append(Paragraph(f'<b>{fa}</b> ↔ <b>{fb}</b>', s['pair_h']))

        parts = []
        if ie.exact_image_count:    parts.append(f'完全相同 {ie.exact_image_count} 对')
        if ie.near_identical_count: parts.append(f'高度相似 {ie.near_identical_count} 对')
        if ie.ps_suspicious:        parts.append(f'PS嫌疑 {ie.ps_suspicious_count} 对')
        if ie.shared_typo_count:    parts.append(f'相同错别字 {ie.shared_typo_count} 个')
        if ie.text_identical_count: parts.append(f'文字完全相同 {ie.text_identical_count} 对')
        if parts:
            story.append(Paragraph(' | '.join(parts), s['muted']))

        for ip in ie.matched_image_pairs[:8]:
            img_a = _thumb(ip.get('thumbnail_base64_a',''),130)
            img_b = _thumb(ip.get('thumbnail_base64_b',''),130)
            if not img_a and not img_b: continue
            tbl = Table([
                [Paragraph('文档A', ParagraphStyle('la',fontName='Hei',fontSize=7,
                    textColor=C_MUTED, alignment=TA_CENTER)),
                 Paragraph('文档B', ParagraphStyle('lb',fontName='Hei',fontSize=7,
                    textColor=C_MUTED, alignment=TA_CENTER))],
                [img_a or '', img_b or ''],
            ], colWidths=[80*mm, 80*mm], hAlign='CENTER')
            tbl.setStyle(TableStyle([
                ('ALIGN',(0,0),(-1,-1),'CENTER'),
                ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
            ]))
            story.append(tbl)
            story.append(Paragraph(f'  置信度: {ip.get("confidence",0):.3f}', s['muted']))

        if ie.shared_typos:
            ts = '、'.join(ie.shared_typos[:8])
            if ie.shared_typo_count>8: ts+=f' 等{ie.shared_typo_count}个'
            story.append(Paragraph(f'⚠ 相同错别字: {_esc(ts)}', s['body']))

    story.append(Spacer(1, 6*mm))


# ================================================================
# 主入口
# ================================================================
def generate_pdf(report: GlobalReport, output_path: str,
                 enabled_dims: Optional[dict] = None) -> str:
    _reg_fonts()
    ss = _sty()
    if enabled_dims is None: enabled_dims = {}

    # 维度数据准备
    dim_pairs = {}
    for dk in ['file_id','author','editor','contact','company_name','credit_code']:
        if enabled_dims.get(dk, True):
            dim_pairs[dk] = [p for p in report.pairwise_results if _has_dim(p, dk)]
    if enabled_dims.get('content_similarity', True):
        dim_pairs['text_sim'] = [p for p in report.pairwise_results if _has_dim(p, 'text_sim')]
        dim_pairs['image_sim'] = [p for p in report.pairwise_results if _has_dim(p, 'image_sim')]

    # 构建文档
    doc = SimpleDocTemplate(output_path, pagesize=A4,
        topMargin=2*cm, bottomMargin=2*cm,
        leftMargin=2.5*cm, rightMargin=2.5*cm,
        title='投标文件串标围标检测报告', author='BidCollusionDetector')

    story = []
    _build_cover(story, ss, report)

    dim_order = [
        ('file_id','文件码雷同'), ('author','文档作者雷同'), ('editor','编辑经办人雷同'),
        ('contact','单位联系人雷同'), ('company_name','公司名称雷同'), ('credit_code','信用代码雷同'),
    ]
    for dk, dt in dim_order:
        if dim_pairs.get(dk):
            if story: story.append(Spacer(1, 2*mm))
            _simple_dim(story, ss, dk, dt, dim_pairs[dk], report.file_profiles)

    if dim_pairs.get('text_sim'):
        story.append(Spacer(1, 5*mm))
        _text_sim(story, ss, dim_pairs['text_sim'], report.file_profiles)

    if dim_pairs.get('image_sim'):
        story.append(Spacer(1, 5*mm))
        _image_sim(story, ss, dim_pairs['image_sim'], report.file_profiles)

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    logger.info(f"PDF 报告已生成: {output_path}")
    return output_path
