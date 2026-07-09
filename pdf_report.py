"""PDF 报告 — 投标文件串标围标检测（专业版）"""
import os, io, base64, logging
from typing import Dict, Optional
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import HexColor, white
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, Image
)
from reportlab.pdfgen import canvas
from data_structures import GlobalReport

logger = logging.getLogger(__name__)

C_PRIMARY   = HexColor('#1e3a5f')
C_SECONDARY = HexColor('#4a90a4')
C_RISK_HIGH = HexColor('#e74c3c')
C_RISK_MED  = HexColor('#f39c12')
C_SAFE      = HexColor('#27ae60')
C_BG        = HexColor('#f8f9fa')
C_TEXT      = HexColor('#2c3e50')
C_MUTED     = HexColor('#7f8c8d')
C_HIGHLIGHT = HexColor('#fff3cd')
C_BORDER    = HexColor('#dce1e8')

_FONT_DIR = "C:/Windows/Fonts"
_FONTS_OK = False
PG_W = 160*mm  # 页面可用宽度：210 - 25*2 mm

def _reg():
    global _FONTS_OK
    if _FONTS_OK: return
    for name, fn in [('Hei','msyh.ttc'),('Hei-Bd','msyhbd.ttc'),('Song','simsun.ttc')]:
        p = os.path.join(_FONT_DIR, fn)
        try: pdfmetrics.registerFont(TTFont(name, p))
        except: pass
    from reportlab.lib.fonts import addMapping
    addMapping('Hei',0,0,'Hei'); addMapping('Hei',1,0,'Hei-Bd')
    _FONTS_OK = True

def S(**kw):
    kw.setdefault('fontName','Hei'); kw.setdefault('textColor',C_TEXT)
    return ParagraphStyle('S',**kw)

def _sty():
    _reg()
    return {
        'cover_title': S(fontName='Hei-Bd',fontSize=24,textColor=C_PRIMARY,alignment=TA_CENTER,spaceAfter=4*mm),
        'stat_num': S(fontName='Hei-Bd',fontSize=26,textColor=C_PRIMARY,alignment=TA_CENTER),
        'stat_label': S(fontSize=9,leading=13,alignment=TA_CENTER),
        'section': S(fontName='Hei-Bd',fontSize=15,textColor=C_PRIMARY,spaceBefore=8*mm,spaceAfter=2*mm),
        'pair_h': S(fontName='Hei-Bd',fontSize=10,textColor=C_PRIMARY,spaceBefore=3*mm,spaceAfter=2*mm),
        'body': S(fontSize=9,leading=14,spaceAfter=2*mm),
        'muted': S(fontSize=7,textColor=C_MUTED),
        'common_label': S(fontName='Hei-Bd',fontSize=8,textColor=HexColor('#856404')),
        'common_text': S(fontName='Song',fontSize=8,leading=12,textColor=HexColor('#856404'),
            leftIndent=4*mm,backColor=C_HIGHLIGHT,borderPadding=3),
    }

def _esc(t): return str(t).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
def _thumb(b64,mw=180):
    if not b64: return None
    try:
        if ',' in b64: b64=b64.split(',',1)[1]
        buf=io.BytesIO(base64.b64decode(b64))
        from PIL import Image as PILImage
        pi=PILImage.open(buf); w,h=pi.size
        if w>mw: r=mw/w; w,h=mw,h*r
        return Image(buf,width=w,height=h)
    except: return None
def _sim_clr(s): return C_RISK_HIGH if s>=0.8 else (C_RISK_MED if s>=0.6 else C_SAFE)

# ================================================================
# 页眉页脚
# ================================================================
def _hf(cv, doc):
    w,h=A4; pg=doc.page
    if pg==1: return
    cv.saveState()
    cv.setStrokeColor(C_PRIMARY); cv.setLineWidth(0.5)
    cv.line(2.5*cm,h-1.5*cm,w-2.5*cm,h-1.5*cm)
    cv.setFont('Hei',8); cv.setFillColor(C_PRIMARY)
    cv.drawString(2.5*cm,h-1.3*cm,"投标文件串标围标检测报告")
    cv.setFillColor(C_MUTED)
    cv.drawRightString(w-2.5*cm,h-1.3*cm,f"第 {pg-1} 页")
    cv.setFont('Hei',7); cv.drawCentredString(w/2,1*cm,"本报告由系统自动生成，结果仅供参考")
    cv.restoreState()

# ================================================================
# 维度检测
# ================================================================
def _has_dim(pair, key):
    e=pair.evidence
    if key=='file_id': return e.metadata_evidence.same_file_id
    if key=='author': return 'author' in e.metadata_evidence.matched_fields
    if key=='editor': return any(f in e.metadata_evidence.matched_fields for f in ['creator','producer','software_fingerprint'])
    if key=='contact': return bool(e.contact_evidence.common_mobiles or e.contact_evidence.common_emails or e.contact_evidence.common_contacts)
    if key=='company_name': return bool(e.contact_evidence.common_companies)
    if key=='credit_code': return bool(e.contact_evidence.common_credit_codes)
    if key=='text_sim': return bool(e.text_evidence.paragraph_matches)
    if key=='image_sim': return bool(e.image_evidence.matched_image_pairs or e.image_evidence.common_image_hashes or e.image_evidence.text_identical_count>0)
    return False

def _dim_detail(pair,key):
    e=pair.evidence; r=[]
    if key=='file_id' and e.metadata_evidence.same_file_id: r.append('两份 PDF 文件码相同')
    elif key=='author' and 'author' in e.metadata_evidence.matched_fields: r.append(f"作者: {e.metadata_evidence.matched_values.get('author','')}")
    elif key=='editor':
        for f in ['creator','producer','software_fingerprint']:
            if f in e.metadata_evidence.matched_fields: r.append(f"{f}: {e.metadata_evidence.matched_values.get(f,'')}")
    elif key=='contact':
        c=e.contact_evidence
        if c.common_mobiles: r.append(f"手机: {'、'.join(c.common_mobiles[:3])}")
        if c.common_emails: r.append(f"邮箱: {'、'.join(c.common_emails[:3])}")
        if c.common_contacts: r.append(f"联系人: {'、'.join(c.common_contacts[:3])}")
    elif key=='company_name' and e.contact_evidence.common_companies: r.append(f"公司: {'、'.join(e.contact_evidence.common_companies[:3])}")
    elif key=='credit_code' and e.contact_evidence.common_credit_codes: r.append(f"信用代码: {'、'.join(e.contact_evidence.common_credit_codes[:3])}")
    return r

# ================================================================
# 封面
# ================================================================
def _cover(story,s,report):
    story.append(Spacer(1,20*mm))
    story.append(Paragraph('投标文件串标围标检测报告',s['cover_title']))
    story.append(Paragraph('Bid Collusion Inspection Report',S(fontSize=11,textColor=C_SECONDARY,alignment=TA_CENTER,spaceAfter=15*mm)))

    items=[('检测文件数',str(report.total_files),C_PRIMARY),('比对总对数',str(report.total_pairs),C_SECONDARY),('有雷同项对数',str(report.suspicious_pairs),C_RISK_HIGH)]
    cards=[]
    for lbl,val,cl in items:
        tb=Table([[Paragraph(val,S(fontName='Hei-Bd',fontSize=28,textColor=cl,alignment=TA_CENTER))],[Paragraph(lbl,s['stat_label'])]],colWidths=[50*mm])
        tb.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),C_BG),('ALIGN',(0,0),(-1,-1),'CENTER'),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(0,0),8*mm),('BOTTOMPADDING',(0,-1),(-1,-1),5*mm),('LINEBELOW',(0,0),(-1,0),3,cl)]))
        cards.append(tb)
    ct=Table([cards],colWidths=[55*mm]*3,hAlign='CENTER')
    ct.setStyle(TableStyle([('ALIGN',(0,0),(-1,-1),'CENTER'),('LEFTPADDING',(0,0),(-1,-1),3*mm),('RIGHTPADDING',(0,0),(-1,-1),3*mm)]))
    story.append(ct); story.append(Spacer(1,12*mm))

    if report.risk_clusters:
        for cl in report.risk_clusters:
            wt=Table([[Paragraph(f'⚠ 涉嫌围标团伙: {len(cl.doc_ids)} 份文件 ({cl.cluster_type})',S(fontName='Hei-Bd',fontSize=11,textColor=C_RISK_HIGH,alignment=TA_CENTER))]],colWidths=[PG_W],hAlign='CENTER')
            wt.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),HexColor('#fdf0f0')),('LINELEFT',(0,0),(0,0),3,C_RISK_HIGH),('TOPPADDING',(0,0),(-1,-1),4*mm),('BOTTOMPADDING',(0,0),(-1,-1),4*mm)]))
            story.append(wt)

    info=S(fontSize=8,textColor=C_MUTED,alignment=TA_CENTER)
    story.append(Spacer(1,15*mm))
    story.append(Paragraph(f"报告ID: {report.report_id}",info))
    story.append(Paragraph(f"生成时间: {report.generated_at}",info))
    story.append(Spacer(1,20*mm))
    story.append(Paragraph("本报告由系统自动生成，结果仅供参考，请结合人工审核确认",S(fontSize=7,textColor=C_MUTED,alignment=TA_CENTER)))
    story.append(PageBreak())

# ================================================================
# 简单维度
# ================================================================
def _sd(story, s, dk, dt, pairs, profiles, groups=None):
    """维度展示 — 优先展示聚合组，无组时降级为 pairwise"""
    story.append(Paragraph(dt, s['section']))
    story.append(Table([['']], colWidths=[40 * mm], rowHeights=[2]))
    story[-1].setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), C_PRIMARY)]))
    story.append(Spacer(1, 4 * mm))

    if groups:
        # ── 聚合组展示 ──
        _render_metadata_groups(story, s, dk, groups)
    else:
        # ── 降级：pairwise 展示 ──
        for pair in pairs:
            fa = _esc(getattr(profiles.get(pair.doc_a_id, {}), 'filename', pair.doc_a_id))
            fb = _esc(getattr(profiles.get(pair.doc_b_id, {}), 'filename', pair.doc_b_id))
            details = _dim_detail(pair, dk)
            cd = [[Paragraph(f'<b>{fa}</b> ↔ <b>{fb}</b>', s['pair_h'])]]
            for det in details:
                cd.append([Paragraph(f'<font color="{C_RISK_HIGH.hexval()}">●</font> {_esc(det)}', s['body'])])
            card = Table(cd, colWidths=[PG_W], hAlign='CENTER')
            card.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), C_BG),
                ('TOPPADDING', (0, 0), (-1, 0), 3 * mm),
                ('BOTTOMPADDING', (0, -1), (-1, -1), 3 * mm),
                ('LEFTPADDING', (0, 0), (-1, -1), 4 * mm),
                ('RIGHTPADDING', (0, 0), (-1, -1), 4 * mm),
                ('LINEBELOW', (0, 0), (-1, 0), 0.5, C_BORDER),
            ]))
            story.append(KeepTogether(card))
            story.append(Spacer(1, 3 * mm))

    story.append(Spacer(1, 6 * mm))


def _render_metadata_groups(story, s, dk, groups):
    """渲染元数据聚合组 — 一组共享同一个值的文档聚合成一条"""
    # 维度标签映射
    label_map = {
        'author': '文档作者',
        'file_id': '文件码',
        'editor': '编辑经办人',
        'contact_mobile': '联系电话',
        'contact_email': '电子邮箱',
        'contact_name': '联系人',
        'company_name': '公司名称',
        'credit_code': '信用代码',
    }
    # 如果 dk 是顶层维度名且 groups 包含子类型，自动按子类型取标签
    sub_type_labels = {
        'contact_mobile': '联系电话',
        'contact_email': '电子邮箱',
        'contact_name': '联系人',
    }

    for g in groups:
        filenames = '、'.join(_esc(fn) for fn in (g.filenames or g.doc_ids))
        # 确定实际标签：优先用子类型标签，其次用 dk 映射
        actual_label = sub_type_labels.get(g.group_type, label_map.get(dk, dk))
        cd = [[Paragraph(f'<b>{filenames}</b>', s['pair_h'])]]
        cd.append([Paragraph(
            f'<font color="{C_RISK_HIGH.hexval()}">●</font> 共同{actual_label}: <b>{_esc(g.shared_value)}</b>',
            s['body'],
        )])
        card = Table(cd, colWidths=[PG_W], hAlign='CENTER')
        card.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), C_BG),
            ('TOPPADDING', (0, 0), (-1, 0), 3 * mm),
            ('BOTTOMPADDING', (0, -1), (-1, -1), 3 * mm),
            ('LEFTPADDING', (0, 0), (-1, -1), 4 * mm),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4 * mm),
            ('LINEBELOW', (0, 0), (-1, 0), 0.5, C_BORDER),
        ]))
        story.append(KeepTogether(card))
        story.append(Spacer(1, 3 * mm))

# ================================================================
# 进度条
# ================================================================
class PBar:
    def __init__(self,val,w=100,h=7):
        self.val=max(0,min(1,val)); self._w=w; self._h=h
        self._c=_sim_clr(val); self.width=w+30; self.height=h+4
    def wrap(self,aw,ah): return (self.width,self.height)
    def draw(self):
        c=self.canv; fw=self._w*self.val; y0=2
        c.setFillColor(C_BG); c.setStrokeColor(C_BORDER); c.setLineWidth(0.5)
        c.roundRect(0,y0,self._w,self._h,3,fill=1,stroke=1)
        if fw>1: c.setFillColor(self._c); c.roundRect(0,y0,fw,self._h,3,fill=1,stroke=0)
        c.setFont('Hei',7); c.setFillColor(C_TEXT)
        c.drawString(self._w+5,y0,f"{self.val*100:.0f}%")

# ================================================================
# 文本相似度
# ================================================================
def _ts(story,s,pairs,profiles):
    story.append(Paragraph('内容相似度 — 文本',s['section']))
    story.append(Table([['']],colWidths=[40*mm],rowHeights=[2]))
    story[-1].setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),C_PRIMARY)]))
    story.append(Spacer(1,4*mm))

    for pair in pairs:
        te=pair.evidence.text_evidence
        if not te.paragraph_matches: continue
        fa=_esc(getattr(profiles.get(pair.doc_a_id,{}),'filename',pair.doc_a_id))
        fb=_esc(getattr(profiles.get(pair.doc_b_id,{}),'filename',pair.doc_b_id))
        sim=pair.similarity_scores.get('text_local',0)

        # Pair 标题 + 进度条
        bar=PBar(sim,100,7)
        td=[[Paragraph(f'<b>{fa}</b> <font color="{C_MUTED.hexval()}">↔</font> <b>{fb}</b>',s['pair_h']),bar]]
        tt=Table(td,colWidths=[110*mm,50*mm],hAlign='CENTER')
        tt.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE')]))
        story.append(tt)
        story.append(Paragraph(f"匹配段落: {len(te.paragraph_matches)} 对 | 克隆块: {len(te.continuous_clone_blocks)} 个",s['muted']))

        for block in te.continuous_clone_blocks:
            story.append(Paragraph(f'<font color="{C_RISK_HIGH.hexval()}">⚠</font> 连续克隆块: {block.get("length",0)} 段, 相似度 {block.get("similarity",0):.4f}',
                S(fontName='Hei-Bd',fontSize=8,textColor=C_RISK_HIGH,spaceBefore=2*mm,leftIndent=4*mm)))

        for match in te.paragraph_matches[:30]:
            ia=match.get('paragraph_a_index','?'); ib=match.get('paragraph_b_index','?')
            pa=f'（第{match.get("page_num_a",-1)+1}页）' if match.get('page_num_a',-1)>=0 else ''
            pb=f'（第{match.get("page_num_b",-1)+1}页）' if match.get('page_num_b',-1)>=0 else ''
            sm=match.get('similarity',0); method=match.get('detection_method','')
            clr=_sim_clr(sm)

            # 头部表格（单行双列，宽度 = PG_W）
            loc=f'<b>A</b> 第[{ia}]段{pa} ↔ <b>B</b> 第[{ib}]段{pb}'
            tag=f'<font color="{clr.hexval()}">■</font> {sm:.4f} | {_esc(method)}'
            ht=Table([[Paragraph(loc,S(fontSize=8,leading=12)),Paragraph(tag,S(fontSize=7,textColor=C_MUTED,alignment=TA_RIGHT))]],
                colWidths=[110*mm,50*mm],hAlign='CENTER')
            ht.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE')]))

            # A/B 原文
            ta=_esc(match.get('paragraph_a','')); tb=_esc(match.get('paragraph_b',''))
            if len(ta)>400: ta=ta[:400]+'…'
            if len(tb)>400: tb=tb[:400]+'…'
            # 使用 PG_W-6mm 给左右 padding 留空间
            cw = (PG_W - 6*mm) / 2
            txt=[[Paragraph('<font color="'+C_PRIMARY.hexval()+'">A 原文</font>',S(fontName='Hei-Bd',fontSize=7,textColor=C_PRIMARY)),
                  Paragraph('<font color="'+C_PRIMARY.hexval()+'">B 原文</font>',S(fontName='Hei-Bd',fontSize=7,textColor=C_PRIMARY))],
                 [Paragraph(f'<font face="Song" size="8">{ta}</font>',S(fontName='Song',fontSize=8,leading=12)),
                  Paragraph(f'<font face="Song" size="8">{tb}</font>',S(fontName='Song',fontSize=8,leading=12))]]
            txt_tbl=Table(txt,colWidths=[cw,cw],hAlign='CENTER')
            txt_tbl.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(0,0),HexColor('#eef5fb')),('BACKGROUND',(1,0),(1,0),HexColor('#fdf5f0')),
                ('BACKGROUND',(0,1),(0,1),HexColor('#f5f9fc')),('BACKGROUND',(1,1),(1,1),HexColor('#fef9f5')),
                ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,0),2*mm),('BOTTOMPADDING',(0,-1),(-1,-1),2*mm),
                ('LEFTPADDING',(0,0),(-1,-1),2*mm),('RIGHTPADDING',(0,0),(-1,-1),2*mm),
                ('GRID',(0,0),(-1,-1),0.3,C_BORDER),
            ]))

            # 共同部分
            cr=[]
            common=match.get('common_parts',[])
            if common:
                clen=sum(len(p) for p in common)
                cr.append([Paragraph(f'共同部分（{len(common)}处,{clen}字）:',s['common_label'])])
                for ci,part in enumerate(common,1):
                    pt=_esc(part.strip()[:200])
                    cr.append([Paragraph(f'<font face="Song" size="8">[{ci}] {pt}</font>',s['common_text'])])

            # 卡片：所有子元素都是单行表格，宽度 PG_W
            card_rows=[[ht],[txt_tbl]]+cr
            card=Table(card_rows,colWidths=[PG_W],hAlign='CENTER')
            card.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,-1),white),
                ('TOPPADDING',(0,0),(-1,0),1*mm),('BOTTOMPADDING',(0,-1),(-1,-1),1*mm),
                ('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),0),
                ('BOX',(0,0),(-1,-1),0.5,C_BORDER),
            ]))
            story.append(KeepTogether(card)); story.append(Spacer(1,3*mm))
    story.append(Spacer(1,6*mm))

# ================================================================
# 图片相似度
# ================================================================
def _is(story,s,pairs,profiles):
    story.append(Paragraph('内容相似度 — 图片',s['section']))
    story.append(Table([['']],colWidths=[40*mm],rowHeights=[2]))
    story[-1].setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),C_PRIMARY)]))
    story.append(Spacer(1,4*mm))
    for pair in pairs:
        ie=pair.evidence.image_evidence
        has_img=bool(ie.matched_image_pairs or ie.common_image_hashes or ie.text_identical_count>0)
        if not has_img and ie.shared_typo_count==0 and not ie.ps_suspicious: continue
        fa=_esc(getattr(profiles.get(pair.doc_a_id,{}),'filename',pair.doc_a_id))
        fb=_esc(getattr(profiles.get(pair.doc_b_id,{}),'filename',pair.doc_b_id))
        story.append(Paragraph(f'<b>{fa}</b> ↔ <b>{fb}</b>',s['pair_h']))
        parts=[]
        if ie.exact_image_count: parts.append(f'完全相同 {ie.exact_image_count} 对')
        if ie.near_identical_count: parts.append(f'高度相似 {ie.near_identical_count} 对')
        if ie.ps_suspicious: parts.append(f'PS嫌疑 {ie.ps_suspicious_count} 对')
        if ie.shared_typo_count: parts.append(f'错别字 {ie.shared_typo_count} 个')
        if ie.text_identical_count: parts.append(f'文字相同 {ie.text_identical_count} 对')
        if parts: story.append(Paragraph(' | '.join(parts),s['muted']))
        for ip in ie.matched_image_pairs[:8]:
            ia=_thumb(ip.get('thumbnail_base64_a',''),130); ib=_thumb(ip.get('thumbnail_base64_b',''),130)
            if not ia and not ib: continue
            cw2=(PG_W-6*mm)/2
            tbl=Table([[Paragraph('文档A',S(fontSize=7,textColor=C_MUTED,alignment=TA_CENTER)),Paragraph('文档B',S(fontSize=7,textColor=C_MUTED,alignment=TA_CENTER))],[ia or '',ib or '']],colWidths=[cw2,cw2],hAlign='CENTER')
            tbl.setStyle(TableStyle([('ALIGN',(0,0),(-1,-1),'CENTER'),('VALIGN',(0,0),(-1,-1),'MIDDLE')]))
            story.append(tbl)
            story.append(Paragraph(f'  置信度: {ip.get("confidence",0):.3f}',s['muted']))
        if ie.shared_typos:
            ts='、'.join(ie.shared_typos[:8])
            if ie.shared_typo_count>8: ts+=f' 等{ie.shared_typo_count}个'
            story.append(Paragraph(f'⚠ 错别字: {_esc(ts)}',s['body']))
    story.append(Spacer(1,6*mm))

# ================================================================
# 主入口
# ================================================================
def generate_pdf(report:GlobalReport,output_path:str,enabled_dims:Optional[dict]=None)->str:
    _reg(); ss=_sty()
    if enabled_dims is None: enabled_dims={}
    dp={}; gp={}

    # 元数据维度：按 group_type 过滤聚合组
    # enabled_dims 的 key 是原始维度名，group_type 是细分类型
    dim_to_group_types = {
        'file_id': ['file_id'],
        'author': ['author'],
        'editor': ['editor'],
        'contact': ['contact_mobile', 'contact_email', 'contact_name'],
        'company_name': ['company_name'],
        'credit_code': ['credit_code'],
    }

    for dk, gts in dim_to_group_types.items():
        if not enabled_dims.get(dk, True):
            continue
        groups_for_dim = [g for g in report.metadata_groups if g.group_type in gts]
        if groups_for_dim:
            gp[dk] = groups_for_dim
        else:
            # fallback: pairwise
            dp[dk] = [p for p in report.pairwise_results if _has_dim(p, dk)]

    if enabled_dims.get('content_similarity',True):
        dp['text_sim']=[p for p in report.pairwise_results if _has_dim(p,'text_sim')]
        dp['image_sim']=[p for p in report.pairwise_results if _has_dim(p,'image_sim')]

    doc=SimpleDocTemplate(output_path,pagesize=A4,topMargin=2*cm,bottomMargin=2*cm,leftMargin=2.5*cm,rightMargin=2.5*cm,title='投标文件串标围标检测报告')
    story=[]
    _cover(story,ss,report)
    for dk,dt in [('file_id','文件码雷同'),('author','文档作者雷同'),('editor','编辑经办人雷同'),
                  ('contact','单位联系人雷同'),('company_name','公司名称雷同'),('credit_code','信用代码雷同')]:
        if gp.get(dk):
            _sd(story, ss, dk, dt, [], report.file_profiles, groups=gp[dk])
        elif dp.get(dk):
            _sd(story, ss, dk, dt, dp[dk], report.file_profiles)
    if dp.get('text_sim'): _ts(story,ss,dp['text_sim'],report.file_profiles)
    if dp.get('image_sim'): _is(story,ss,dp['image_sim'],report.file_profiles)

    doc.build(story,onFirstPage=_hf,onLaterPages=_hf)
    logger.info(f"PDF 报告已生成: {output_path}")
    return output_path
