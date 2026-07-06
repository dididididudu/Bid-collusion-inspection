"""
模块 E：报告生成引擎
- 输出完整段落内容
- 高亮标记相似部分
- 生成HTML可视化报告
"""
import os
import re
import json
import logging
from typing import Any
from dataclasses import asdict

from data_structures import GlobalReport
from config import DetectionConfig

logger = logging.getLogger(__name__)


class ReportGenerator:
    """报告生成器"""

    def __init__(self, config: DetectionConfig):
        self.config = config

    def generate(self, report: GlobalReport, output_dir: str) -> None:
        """生成检测报告"""
        os.makedirs(output_dir, exist_ok=True)

        # 1. 生成JSON报告
        self._generate_json_report(report, output_dir)

        # 2. 生成HTML可视化报告
        self._generate_html_report(report, output_dir)

        logger.info(f"报告已生成到: {output_dir}")

    def _generate_json_report(self, report: GlobalReport, output_dir: str) -> None:
        """生成JSON格式的完整报告"""
        json_path = os.path.join(output_dir, "detection_report.json")

        report_dict = self._dataclass_to_dict(report)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"JSON报告已生成: {json_path}")

    def _dataclass_to_dict(self, obj: Any) -> Any:
        """递归转换dataclass为字典（bytes→base64）"""
        if hasattr(obj, '__dataclass_fields__'):
            result = {}
            for field_name, field_value in obj.__dict__.items():
                result[field_name] = self._dataclass_to_dict(field_value)
            return result
        elif isinstance(obj, list):
            return [self._dataclass_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._dataclass_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, bytes):
            import base64
            return base64.b64encode(obj).decode('ascii')
        else:
            return obj

    def _format_similarity(self, similarity: float) -> str:
        """格式化相似度描述"""
        if similarity >= 0.95:
            return "极度相似 ⚠️"
        elif similarity >= 0.85:
            return "高度相似 ⚠️"
        elif similarity >= 0.70:
            return "中度相似"
        elif similarity >= 0.50:
            return "轻度相似"
        elif similarity >= 0.30:
            return "略有相似"
        else:
            return "基本不相似"

    def _build_dimension_summary(self, result, evidence, fn_a: str, fn_b: str) -> list:
        """构建 7 项检测维度总览"""
        me = evidence.metadata_evidence
        ce = evidence.contact_evidence
        te = evidence.text_evidence
        scores = result.similarity_scores

        dims = []

        # 1. 内容雷同
        text_sim = scores.get('text_local', 0)
        para_count = len(te.paragraph_matches)
        clone_count = len(te.continuous_clone_blocks)
        dims.append({
            'name': '内容雷同', 'icon': '📝',
            'hit': text_sim >= 0.3,
            'detail': f'相似度 {text_sim:.2f}, {para_count} 对匹配段, {clone_count} 个克隆块',
        })

        # 2. 文件码雷同
        dims.append({
            'name': '文件码雷同', 'icon': '🔑',
            'hit': me.same_file_id,
            'detail': 'PDF /ID[0] 相同 — 两份文件从同一源文件生成',
        })

        # 3. 编辑经办人雷同 (creator/producer/software_fingerprint)
        editor_fields = [f for f in ['creator', 'producer', 'software_fingerprint'] if f in me.matched_fields]
        dims.append({
            'name': '编辑经办人雷同', 'icon': '✏️',
            'hit': len(editor_fields) > 0,
            'detail': f'{", ".join(editor_fields)} 相同' if editor_fields else '',
        })

        # 4. 文档作者雷同
        dims.append({
            'name': '文档作者雷同', 'icon': '👤',
            'hit': 'author' in me.matched_fields,
            'detail': me.matched_values.get('author', '') if 'author' in me.matched_fields else '',
        })

        # 5. 同标段单位联系人雷同
        contact_hits = []
        if ce.common_mobiles:
            contact_hits.append(f'手机 {",".join(ce.common_mobiles)}')
        if ce.common_emails:
            contact_hits.append(f'邮箱 {",".join(ce.common_emails)}')
        if ce.common_contacts:
            contact_hits.append(f'联系人 {"、".join(ce.common_contacts[:3])}')
        dims.append({
            'name': '单位联系人雷同', 'icon': '📞',
            'hit': len(contact_hits) > 0,
            'detail': '; '.join(contact_hits) if contact_hits else '',
        })

        # 6. 投标文件公司名称异常
        company_hits = ce.common_companies
        dims.append({
            'name': '公司名称异常', 'icon': '🏢',
            'hit': len(company_hits) > 0,
            'detail': f'相同公司: {"、".join(company_hits[:3])}' if company_hits else '',
        })

        # 7. 信用代码雷同
        dims.append({
            'name': '信用代码雷同', 'icon': '🏛️',
            'hit': len(ce.common_credit_codes) > 0,
            'detail': f'{"、".join(ce.common_credit_codes[:3])} 相同' if ce.common_credit_codes else '',
        })

        # 8. 会员号雷同（外部 API 注入，当前为空）
        dims.append({
            'name': '会员号雷同', 'icon': '🆔',
            'hit': len(ce.common_member_ids) > 0,
            'detail': f'{"、".join(ce.common_member_ids[:3])} 相同' if ce.common_member_ids else '',
        })

        return dims

    def _generate_html_report(self, report: GlobalReport, output_dir: str) -> None:
        """生成HTML可视化报告 — 展示所有匹配证据，无过滤"""
        html_path = os.path.join(output_dir, "detection_report.html")

        all_results = report.pairwise_results
        all_results.sort(key=lambda x: x.risk_score, reverse=True)

        html = self._build_html_content(report, all_results)

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"HTML报告已生成: {html_path}")

    def _build_html_content(self, report: GlobalReport, results: list) -> str:
        """构建HTML报告内容 — 使用 list+join 避免 O(n^2) 字符串拼接"""
        risk_colors = {"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#3498db", "NONE": "#95a5a6"}

        parts = []  # 使用列表收集片段，最后一次性 join

        suspicious = sum(1 for r in results if r.has_evidence())
        parts.append(f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>投标文件串标围标检测报告</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Microsoft YaHei', '微软雅黑', sans-serif; background: #f5f6fa; color: #2c3e50; line-height: 1.6; }}
.container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
.header {{ background: linear-gradient(135deg, #2c3e50, #3498db); color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; }}
.header h1 {{ font-size: 24px; margin-bottom: 10px; }}
.header .meta {{ font-size: 14px; opacity: 0.8; }}
.stats {{ display: flex; gap: 15px; flex-wrap: wrap; margin-bottom: 20px; }}
.stat-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); flex: 1; min-width: 150px; text-align: center; }}
.stat-card .number {{ font-size: 32px; font-weight: bold; color: #2c3e50; }}
.stat-card .label {{ font-size: 13px; color: #7f8c8d; margin-top: 5px; }}
.stat-card.hit .number {{ color: #e74c3c; }}
.result-card {{ background: white; padding: 25px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 20px; }}
.result-card h2 {{ font-size: 18px; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 2px solid #ecf0f1; }}
.match-item {{ background: #f8f9fa; padding: 15px; border-radius: 6px; margin: 10px 0; border-left: 4px solid #3498db; }}
.match-item.clone {{ border-left-color: #e74c3c; }}
.match-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; flex-wrap: wrap; gap: 10px; }}
.match-sim {{ font-size: 18px; font-weight: bold; }}
.clone-block {{ background: #fff3cd; border: 1px solid #ffc107; padding: 12px; border-radius: 6px; margin: 10px 0; }}
.text-compare {{ display: flex; gap: 15px; margin-top: 10px; }}
.text-col {{ flex: 1; background: white; padding: 12px; border-radius: 4px; border: 1px solid #dee2e6; font-size: 13px; max-height: 600px; overflow-y: auto; white-space: pre-wrap; word-break: break-all; }}
.text-col h4 {{ margin-bottom: 8px; color: #495057; position: sticky; top: 0; background: white; padding-bottom: 5px; border-bottom: 1px solid #eee; }}
.highlight {{ background: #ffeb3b; padding: 2px 4px; border-radius: 2px; font-weight: bold; }}
.common-text {{ background: #e8f5e9; padding: 10px; border-radius: 4px; margin: 8px 0; font-size: 13px; border-left: 3px solid #4caf50; white-space: pre-wrap; word-break: break-all; }}
.common-text .label {{ font-weight: bold; color: #2e7d32; font-size: 12px; }}
summary {{ cursor: pointer; padding: 8px; background: #e3f2fd; border-radius: 4px; font-weight: bold; }}
summary:hover {{ background: #bbdefb; }}
.footer {{ text-align: center; padding: 20px; color: #95a5a6; font-size: 13px; }}
.dimension-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 8px; margin: 12px 0; }}
.dim-item {{ background: #f8f9fa; padding: 10px 12px; border-radius: 6px; border-left: 3px solid #dee2e6; }}
.dim-item.dim-hit {{ background: #e8f5e9; border-left-color: #4caf50; }}
.dim-icon {{ font-size: 18px; display: inline; margin-right: 6px; }}
.dim-name {{ display: inline; font-weight: bold; font-size: 13px; color: #495057; }}
.dim-value {{ margin-top: 4px; font-size: 12px; color: #666; }}
.section-title {{ background: #2c3e50; color: white; padding: 8px 15px; border-radius: 4px; margin: 15px 0 10px 0; font-size: 15px; font-weight: bold; }}
</style>
</head>
<body>
<div class="container">
<div class="header">
<h1>📋 投标文件串标围标检测报告</h1>
<div class="meta">报告ID: {report.report_id} | 生成时间: {report.generated_at}</div>
</div>

<div class="stats">
<div class="stat-card"><div class="number">{report.total_files}</div><div class="label">检测文件数</div></div>
<div class="stat-card"><div class="number">{report.total_pairs}</div><div class="label">比对总对数</div></div>
<div class="stat-card hit"><div class="number">{suspicious}</div><div class="label">有雷同项的对数</div></div>
</div>
""")

        # === 雷同文档对详情 ===
        if results:
            parts.append("<h2 style='margin-bottom:15px;'>🔍 雷同文档对详情</h2>")

            total_pairs = len(results)
            for i, result in enumerate(results, 1):
                profile_a = report.file_profiles.get(result.doc_a_id)
                profile_b = report.file_profiles.get(result.doc_b_id)
                text_local = result.similarity_scores.get('text_local', 0)
                evidence = result.evidence
                risk_color = risk_colors.get(result.risk_level, "#95a5a6")

                filename_a = profile_a.filename if profile_a else result.doc_a_id
                filename_b = profile_b.filename if profile_b else result.doc_b_id

                para_matches = evidence.text_evidence.paragraph_matches
                clone_blocks = evidence.text_evidence.continuous_clone_blocks

                # === 检测维度总览 ===
                dims = self._build_dimension_summary(result, evidence, filename_a, filename_b)
                dims_html = ''.join(
                    f'<div class="dim-item {"dim-hit" if d["hit"] else ""}">'
                    f'<div class="dim-icon">{d["icon"]}</div>'
                    f'<div class="dim-name">{d["name"]}</div>'
                    f'<div class="dim-value">{"✅ " + d["detail"] if d["hit"] else "—"}</div>'
                    f'</div>'
                    for d in dims
                )

                # 构建元数据和联系人证据 HTML
                meta_html = self._build_metadata_evidence_html(evidence.metadata_evidence)
                contact_html = self._build_contact_evidence_html(evidence.contact_evidence)

                parts.append(f"""
<div class="result-card">
<h2>#{i} {filename_a} ↔ {filename_b}</h2>
<div style="display:flex;gap:15px;flex-wrap:wrap;margin-bottom:15px;">
<div><strong>📊 文本相似度:</strong> <span style="font-size:18px;">{text_local:.4f}</span></div>
<div><strong>🔗 相似段落:</strong> {len(para_matches)} 对</div>
<div><strong>📎 克隆块:</strong> {len(clone_blocks)} 个</div>
</div>
<div class="dimension-grid">
{dims_html}
</div>
{meta_html}
{contact_html}""")

                # === 图片雷同证据（HTML） ===
                self._append_image_evidence_html(parts, evidence.image_evidence)

                if not para_matches:
                    parts.append("<p>（无相似段落详情）</p></div>")
                    continue

                # 按相似度排序
                sorted_matches = sorted(para_matches, key=lambda x: x.get('similarity', 0), reverse=True)

                # 收集克隆块中的匹配键
                clone_match_keys = set()
                for block in clone_blocks:
                    for pair in block.get('pairs', []):
                        clone_match_keys.add((pair['a_index'], pair['b_index']))

                # === 连续克隆块展示 ===
                if clone_blocks:
                    parts.append("<div class='section-title'>⚠ 连续克隆块 — 连续雷同段落（最强围标证据）</div>")

                    for block in clone_blocks:
                        block_id = block.get('group_id', '?')
                        block_len = block.get('length', 0)
                        block_sim = block.get('similarity', 0)
                        block_pairs_keys = [
                            (p['a_index'], p['b_index']) for p in block.get('pairs', [])
                        ]
                        block_matches = [
                            m for m in sorted_matches
                            if (m.get('paragraph_a_index'), m.get('paragraph_b_index')) in block_pairs_keys
                        ]

                        parts.append(f"""
<div class="clone-block">
<strong>克隆块 [{block_id}]</strong> |
连续 <strong>{block_len}</strong> 段雷同 |
平均相似度: <strong>{block_sim:.4f}</strong><br>
<strong>段落序列:</strong> {', '.join(f'A[{p["a_index"]}]↔B[{p["b_index"]}]' for p in block.get('pairs', []))}
</div>""")

                        for bm in block_matches:
                            parts.append(self._build_match_html(bm, filename_a, filename_b, is_clone=True))

                # === 其他独立相似段落 ===
                non_clone_matches = [
                    m for m in sorted_matches
                    if (m.get('paragraph_a_index'), m.get('paragraph_b_index')) not in clone_match_keys
                ]

                if non_clone_matches:
                    parts.append(f"<div class='section-title'>📝 其他相似段落（共 {len(non_clone_matches)} 对）</div>")

                    for nm in non_clone_matches:
                        parts.append(self._build_match_html(nm, filename_a, filename_b, is_clone=False))

                parts.append("</div>")

                # 进度日志（每处理完一对输出）
                logger.info(f"HTML报告生成进度: {i}/{total_pairs} 对")

        parts.append("""
<div class="footer">
<p>本报告由 投标文件串标围标检测系统 自动生成 | 结果仅供参考，请结合人工审核</p>
</div>
</div>
</body>
</html>""")

        return ''.join(parts)

    @staticmethod
    def _build_metadata_evidence_html(me) -> str:
        """构建元数据雷同证据 HTML"""
        parts = []
        if me.matched_fields:
            parts.append("<div class='section-title'>📋 元数据雷同</div>")
            parts.append("<div style='margin:8px 0;font-size:13px;'>")
            for f in me.matched_fields:
                parts.append(f"<div style='padding:4px 8px;margin:2px 0;background:#e8f5e9;border-radius:4px;'>"
                             f"✅ <strong>{f}</strong>: {me.matched_values.get(f, '')}</div>")
            if me.same_time_bucket:
                parts.append("<div style='padding:4px 8px;margin:2px 0;background:#e8f5e9;border-radius:4px;'>"
                             "✅ 文件创建时间在同一小时</div>")
            if me.same_file_id:
                parts.append("<div style='padding:4px 8px;margin:2px 0;background:#ffebee;border-radius:4px;'>"
                             "🔴 <strong>PDF文件码相同 — 两份文件从同一源文件生成</strong></div>")
            parts.append("</div>")
        return ''.join(parts)

    @staticmethod
    def _build_contact_evidence_html(ce) -> str:
        """构建联系人雷同证据 HTML"""
        parts = []
        items = []
        if ce.common_companies:
            items.append(f"🏢 相同公司: {'、'.join(ce.common_companies)}")
        if ce.common_contacts:
            items.append(f"👤 相同联系人: {'、'.join(ce.common_contacts)}")
        if ce.common_mobiles:
            items.append(f"📱 相同手机: {'、'.join(ce.common_mobiles)}")
        if ce.common_emails:
            items.append(f"📧 相同邮箱: {'、'.join(ce.common_emails)}")
        if ce.common_credit_codes:
            items.append(f"🏛️ 相同信用代码: {'、'.join(ce.common_credit_codes)}")
        if ce.common_member_ids:
            items.append(f"🆔 相同会员号: {'、'.join(ce.common_member_ids)}")
        if items:
            parts.append("<div class='section-title'>📞 联系人/公司雷同</div>")
            parts.append("<div style='margin:8px 0;font-size:13px;'>")
            for item in items:
                parts.append(f"<div style='padding:4px 8px;margin:2px 0;background:#fff3e0;border-radius:4px;'>"
                             f"⚠️ {item}</div>")
            parts.append("</div>")
        return ''.join(parts)

    @staticmethod
    def _append_image_evidence_html(parts: list, image_ev) -> None:
        """追加图片雷同证据到 HTML 片段列表（含逐对详情）"""
        ie = image_ev

        has_any = (getattr(ie, 'exact_image_count', 0) > 0
                   or getattr(ie, 'near_identical_count', 0) > 0
                   or getattr(ie, 'similar_image_count', 0) > 0
                   or getattr(ie, 'ps_suspicious', False)
                   or getattr(ie, 'shared_typo_count', 0) > 0
                   or getattr(ie, 'text_identical_count', 0) > 0
                   or getattr(ie, 'text_similar_count', 0) > 0
                   or getattr(ie, 'matched_image_pairs', None)
                   or getattr(ie, 'matched_text_pairs', None)
                   or getattr(ie, 'ps_detail_list', None))
        if not has_any:
            return

        parts.append("<div class='section-title'>📷 图片雷同证据</div>")
        parts.append("<div style='margin:10px 0;'>")

        # === 汇总 ===
        if getattr(ie, 'exact_image_count', 0) > 0:
            parts.append(f"<p><strong>完全相同图片:</strong> {ie.exact_image_count} 对</p>")
        if getattr(ie, 'near_identical_count', 0) > 0:
            parts.append(f"<p><strong>高度相似图片:</strong> {ie.near_identical_count} 对</p>")
        if getattr(ie, 'similar_image_count', 0) > 0:
            parts.append(f"<p><strong>相似图片:</strong> {ie.similar_image_count} 对</p>")
        if getattr(ie, 'ps_suspicious', False):
            parts.append(f"<p><strong>⚠ PS嫌疑:</strong> {ie.ps_suspicious_count} 对</p>")
        if getattr(ie, 'shared_typo_count', 0) > 0:
            typos_str = ', '.join(getattr(ie, 'shared_typos', [])[:5])
            parts.append(f"<p><strong>相同错别字:</strong> {ie.shared_typo_count} 个 "
                         f"({typos_str}{'...' if ie.shared_typo_count > 5 else ''})</p>")
        if getattr(ie, 'text_identical_count', 0) > 0:
            parts.append(f"<p><strong>图片文字完全相同:</strong> {ie.text_identical_count} 对</p>")
        if getattr(ie, 'text_similar_count', 0) > 0:
            parts.append(f"<p><strong>图片文字高度相似:</strong> {ie.text_similar_count} 对</p>")
        parts.append(f"<p><strong>图片风险分:</strong> {getattr(ie, 'image_risk_score', 0)}/30</p>")

        # === 逐对图片匹配详情 ===
        image_pairs = getattr(ie, 'matched_image_pairs', None) or []
        if image_pairs:
            parts.append("<details open><summary style='margin-top:15px;'>🖼 逐对图片匹配详情（全部展示）</summary>")
            for idx, pair in enumerate(image_pairs):
                thumb_a = pair.get('thumbnail_base64_a', '')
                thumb_b = pair.get('thumbnail_base64_b', '')
                conf = pair.get('confidence', 0)
                orb = pair.get('orb_match_ratio', 0)
                hist = pair.get('histogram_correlation', 0)
                reasons = pair.get('reasons', [])

                # 每层检测状态标签
                l1 = pair.get('l1_pass', False)
                l2 = pair.get('l2_pass', False)
                l3 = pair.get('l3_pass', False)
                _layer = lambda ok, lbl, emoji: f"<span style='margin:0 4px;padding:1px 6px;border-radius:10px;font-size:11px;background:{'#e8f5e8' if ok else '#ffebee'};color:{'#2e7d32' if ok else '#c62828'};'>{emoji} {lbl}</span>"

                # 标记该对的问题类型
                issue_tags = []
                if l1:
                    issue_tags.append("<span style='background:#e3f2fd;padding:2px 8px;border-radius:10px;font-size:11px;'>🔵 L1哈希匹配</span>")
                if not l2 and l1 and thumb_a and thumb_b:
                    issue_tags.append("<span style='background:#fff3e0;padding:2px 8px;border-radius:10px;font-size:11px;'>🟠 L2结构可能不同(ORB)</span>")
                if not l3 and l2 and thumb_a and thumb_b:
                    issue_tags.append("<span style='background:#ffebee;padding:2px 8px;border-radius:10px;font-size:11px;'>🔴 L3颜色可能不同(直方图)</span>")

                parts.append(f"""
<div style='border:1px solid #ddd;border-radius:8px;padding:12px;margin:10px 0;background:#fafafa;'>
  <div style='font-size:13px;color:#2c3e50;margin-bottom:8px;'>
    <strong>#{idx+1}</strong>
    {_layer(l1, 'L1哈希', '✅')}
    {_layer(bool(l2 or not thumb_a), 'L2结构', '✅' if l2 else '⏭️')}
    {_layer(bool(l3 or not thumb_a), 'L3颜色', '✅' if l3 else '⏭️')}
    <span style='margin-left:8px;font-size:12px;'>置信度: {conf:.3f}</span>
  </div>
  <div style='margin-bottom:6px;font-size:12px;'>{' '.join(issue_tags)}</div>
  <div style='font-size:11px;color:#7f8c8d;margin-bottom:6px;'>{' | '.join(reasons)}</div>
  <div style='display:flex;gap:15px;flex-wrap:wrap;'>
    <div style='text-align:center;flex:1;min-width:120px;'>
      <div style='font-size:11px;color:#7f8c8d;margin-bottom:4px;'>文档A — {pair.get('source_a','')}</div>
      {'<img src="'+thumb_a+'" style="max-width:200px;max-height:150px;border:1px solid #ddd;border-radius:4px;" />' if thumb_a else '<div style="width:120px;height:80px;background:#eee;display:inline-flex;align-items:center;justify-content:center;font-size:11px;color:#999;">无缩略图</div>'}
    </div>
    <div style='text-align:center;flex:1;min-width:120px;'>
      <div style='font-size:11px;color:#7f8c8d;margin-bottom:4px;'>文档B — {pair.get('source_b','')}</div>
      {'<img src="'+thumb_b+'" style="max-width:200px;max-height:150px;border:1px solid #ddd;border-radius:4px;" />' if thumb_b else '<div style="width:120px;height:80px;background:#eee;display:inline-flex;align-items:center;justify-content:center;font-size:11px;color:#999;">无缩略图</div>'}
    </div>
  </div>
  <div style='font-size:12px;color:#555;margin-top:6px;display:flex;gap:15px;flex-wrap:wrap;'>
    <span>pHashes距离: {pair.get('phash_dist','-')} / dHash距离: {pair.get('dhash_dist','-')}</span>
    <span>ORB匹配率: {orb:.2f}</span>
    <span>直方图相关性: {hist:.3f}</span>
  </div>
</div>""")
            parts.append("</details>")

        # === 逐对文字匹配详情 ===
        text_pairs = getattr(ie, 'matched_text_pairs', None) or []
        if text_pairs:
            parts.append("<details><summary style='margin-top:15px;'>📝 逐对图片文字匹配详情（全部展示）</summary>")
            for idx, t in enumerate(text_pairs):
                parts.append(f"""
<div style='border:1px solid #ddd;border-radius:6px;padding:10px;margin:8px 0;background:#fafafa;'>
  <div style='font-size:12px;color:#555;margin-bottom:6px;'>
    <strong>#{idx+1}</strong> SBERT相似度: {t.get('similarity',0):.4f} | 方法: {t.get('method','?')}
  </div>
  <div class='text-compare'>
    <div class='text-col'><h4>文档A</h4>{ReportGenerator._escape_html(t.get('text_a',''))}</div>
    <div class='text-col'><h4>文档B</h4>{ReportGenerator._escape_html(t.get('text_b',''))}</div>
  </div>
</div>""")
            parts.append("</details>")

        # === PS 嫌疑详情 ===
        ps_list = getattr(ie, 'ps_detail_list', None) or []
        if ps_list:
            parts.append("<details><summary style='margin-top:15px;'>⚠ PS 嫌疑详情（全部展示）</summary>")
            for idx, p in enumerate(ps_list):
                parts.append(f"""
<div style='border:1px solid #ffc107;border-radius:6px;padding:10px;margin:8px 0;background:#fffde7;'>
  <div style='font-size:12px;color:#555;margin-bottom:6px;'>
    <strong>#{idx+1}</strong> 文字相似度: {p.get('text_sim',0):.3f} |
    non_text距离: {p.get('non_text_dist','?')} |
    证据数: {p.get('evidence_count',1)}
  </div>
  <div class='text-compare'>
    <div class='text-col'><h4>文档A</h4>{ReportGenerator._escape_html(p.get('text_a',''))}</div>
    <div class='text-col'><h4>文档B</h4>{ReportGenerator._escape_html(p.get('text_b',''))}</div>
  </div>
</div>""")
            parts.append("</details>")

        # === 嵌入匹配的原图（旧版路径，保留兼容） ===
        matched_paths = getattr(ie, 'matched_image_paths', {}) or {}
        if matched_paths:
            paths_a = matched_paths.get('doc_a', {})
            paths_b = matched_paths.get('doc_b', {})
            common_hashes = set(paths_a.keys()) & set(paths_b.keys())
            if common_hashes:
                parts.append("<div style='margin-top:15px;'>")
                parts.append("<p><strong>🖼 匹配图片对比（PDF源图）:</strong></p>")
                from image_analysis.image_exporter import image_to_base64
                for h in sorted(common_hashes):
                    b64_a = image_to_base64(paths_a[h])
                    b64_b = image_to_base64(paths_b[h])
                    if b64_a and b64_b:
                        parts.append(f"""
<div style='display:inline-block;margin:10px;border:2px solid #e74c3c;border-radius:8px;padding:10px;background:#fff;'>
  <div style='display:flex;gap:10px;align-items:flex-start;'>
    <div style='text-align:center;'>
      <div style='font-size:11px;color:#7f8c8d;margin-bottom:4px;'>文档A</div>
      <img src='{b64_a}' style='max-width:300px;max-height:200px;border:1px solid #ddd;' />
    </div>
    <div style='text-align:center;'>
      <div style='font-size:11px;color:#7f8c8d;margin-bottom:4px;'>文档B</div>
      <img src='{b64_b}' style='max-width:300px;max-height:200px;border:1px solid #ddd;' />
    </div>
  </div>
  <div style='font-size:10px;color:#95a5a6;margin-top:4px;text-align:center;'>hash: {h[:16]}...</div>
</div>""")
                parts.append("</div>")
        parts.append("</div>")

    def _build_match_html(self, match: dict, filename_a: str, filename_b: str,
                           is_clone: bool = False) -> str:
        """构建单个段落匹配的HTML片段（完整文本，无截断）

        Args:
            match: 段落匹配字典
            filename_a: 文档A的文件名
            filename_b: 文档B的文件名
            is_clone: 是否属于连续克隆块
        """
        sim = match.get('similarity', 0)
        method = match.get('detection_method', '?')
        idx_a = match.get('paragraph_a_index', '?')
        idx_b = match.get('paragraph_b_index', '?')
        clone_class = " clone" if is_clone else ""
        clone_label = " [连续克隆]" if is_clone else ""

        html = f"""
<div class="match-item{clone_class}">
<div class="match-header">
<span class="match-sim">相似度: {sim:.4f}</span>
<span style="font-size:13px;color:#7f8c8d;">方法: {method}{clone_label}</span>
<span style="font-size:12px;color:#95a5a6;">
  A[{filename_a}] 第[{idx_a}]段 ↔ B[{filename_b}] 第[{idx_b}]段
</span>
</div>"""

        # 共同文本片段（全部输出）
        common_parts = match.get('common_parts', [])
        if common_parts:
            html += f"<div style='margin:8px 0;'><strong>📎 共同文本片段（共 {len(common_parts)} 处）:</strong></div>"
            for k, part in enumerate(common_parts, 1):
                html += f"<div class='common-text'><span class='label'>[{k}]</span> {self._escape_html(part)}</div>"

        # 高亮文本对比（完整输出，无截断）
        highlighted_a = match.get('highlighted_text_a', '')
        highlighted_b = match.get('highlighted_text_b', '')
        para_a = match.get('paragraph_a', '')
        para_b = match.get('paragraph_b', '')

        text_a_to_show = highlighted_a if highlighted_a else para_a
        text_b_to_show = highlighted_b if highlighted_b else para_b

        if text_a_to_show or text_b_to_show:
            html += "<div class='text-compare'>"
            if text_a_to_show:
                html += f"""<div class='text-col'>
<h4>📄 文档A — {filename_a} 第[{idx_a}]段</h4>
{self._format_highlighted_html(text_a_to_show)}
</div>"""
            if text_b_to_show:
                html += f"""<div class='text-col'>
<h4>📄 文档B — {filename_b} 第[{idx_b}]段</h4>
{self._format_highlighted_html(text_b_to_show)}
</div>"""
            html += "</div>"

        html += "</div>"
        return html

    def _escape_html(self, text: str) -> str:
        """转义HTML特殊字符"""
        return (text.replace('&', '&amp;')
                    .replace('<', '&lt;')
                    .replace('>', '&gt;')
                    .replace('"', '&quot;'))

    # 预编译正则，避免每次调用时重新编译
    _HL_RE = re.compile(r'【(.*?)】')

    def _format_highlighted_html(self, text: str) -> str:
        """将【】标记的文本转换为HTML高亮格式"""
        # 先转义HTML
        text = self._escape_html(text)

        # 将【xxx】转换为高亮span（使用预编译正则）
        text = self._HL_RE.sub(r'<span class="highlight">\1</span>', text)

        return text
