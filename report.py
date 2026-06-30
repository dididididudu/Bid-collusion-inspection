"""
模块 E：报告生成引擎
"""
import os
import json
import logging
from typing import Dict, Any
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
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 1. 生成JSON报告
        self._generate_json_report(report, output_dir)

        # 2. 生成文本摘要报告
        self._generate_summary_report(report, output_dir)

        # 3. 生成CSV表格（可疑对列表）
        self._generate_csv_report(report, output_dir)

        logger.info(f"报告已生成到: {output_dir}")

    def _generate_json_report(self, report: GlobalReport, output_dir: str) -> None:
        """生成JSON格式的完整报告"""
        json_path = os.path.join(output_dir, "detection_report.json")

        # 将dataclass转换为字典
        report_dict = self._dataclass_to_dict(report)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"JSON报告已生成: {json_path}")

    def _dataclass_to_dict(self, obj: Any) -> Any:
        """递归转换dataclass为字典"""
        if hasattr(obj, '__dataclass_fields__'):
            result = {}
            for field_name, field_value in obj.__dict__.items():
                result[field_name] = self._dataclass_to_dict(field_value)
            return result
        elif isinstance(obj, list):
            return [self._dataclass_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._dataclass_to_dict(v) for k, v in obj.items()}
        else:
            return obj

    def _generate_summary_report(self, report: GlobalReport, output_dir: str) -> None:
        """生成文本摘要报告 - 简化版：只保留相似度、风险评级和相似段落"""
        summary_path = os.path.join(output_dir, "summary.txt")

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("投标文件串标围标检测报告\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"报告ID: {report.report_id}\n")
            f.write(f"生成时间: {report.generated_at}\n\n")

            f.write("【总体统计】\n")
            f.write(f"- 检测文件总数: {report.total_files}\n")
            f.write(f"- 比对总对数: {report.total_pairs}\n")
            f.write(f"- 高相似对数: {report.high_risk_pairs}\n\n")

            f.write("【相似文档对详情】\n")
            all_results = report.pairwise_results
            all_results.sort(key=lambda x: x.risk_score, reverse=True)

            if all_results:
                for i, result in enumerate(all_results, 1):
                    profile_a = report.file_profiles.get(result.doc_a_id)
                    profile_b = report.file_profiles.get(result.doc_b_id)

                    text_local = result.similarity_scores.get('text_local', 0)
                    evidence = result.evidence
                    
                    if text_local < 0.4:
                        continue

                    f.write(f"\n{'═' * 80}\n")
                    f.write(f"{i}. 检测结果\n")
                    f.write(f"{'═' * 80}\n")

                    if profile_a and profile_b:
                        f.write(f"   文档A: {profile_a.filename}\n")
                        f.write(f"   文档B: {profile_b.filename}\n")

                    f.write(f"   相似度: {text_local:.4f}\n")
                    f.write(f"   串标围标风险评级: {result.risk_level}\n\n")

                    if evidence.text_evidence.continuous_clone_blocks:
                        f.write("   ⚠️ 【连续克隆块警告】\n")
                        for block in evidence.text_evidence.continuous_clone_blocks:
                            f.write(f"      克隆块组ID: {block['group_id']}\n")
                            f.write(f"      连续雷同段落数: {block['length']}\n")
                            f.write(f"      平均相似度: {block['similarity']:.4f}\n")
                            f.write(f"      涉及段落索引对: {[(p['a_index'], p['b_index']) for p in block['pairs']]}\n")
                        f.write("\n")

                    if evidence.text_evidence.paragraph_matches:
                        f.write("   【相似段落】\n")
                        f.write(f"   共发现 {len(evidence.text_evidence.paragraph_matches)} 对相似段落:\n")
                        for j, match in enumerate(evidence.text_evidence.paragraph_matches, 1):
                            sim = match.get('similarity', 0)
                            method = match.get('detection_method', '')
                            is_clone = match.get('is_continuous_clone', False)
                            highlighted_a = match.get('highlighted_text_a', '')
                            highlighted_b = match.get('highlighted_text_b', '')
                            idx_a = match.get('paragraph_a_index', 0)
                            idx_b = match.get('paragraph_b_index', 0)
                            
                            clone_mark = " [连续克隆]" if is_clone else ""
                            f.write(f"\n   {j}. 相似度: {sim:.4f} (检测方法: {method}){clone_mark}\n")
                            
                            if highlighted_a:
                                f.write(f"      文档A段落[{idx_a}]: {highlighted_a[:300]}...\n")
                            else:
                                f.write(f"      文档A段落[{idx_a}]: {match.get('paragraph_a', '')[:150]}...\n")
                            
                            if highlighted_b:
                                f.write(f"      文档B段落[{idx_b}]: {highlighted_b[:300]}...\n")
                            else:
                                f.write(f"      文档B段落[{idx_b}]: {match.get('paragraph_b', '')[:150]}...\n")

                    if result.risk_factors:
                        f.write("\n   【检测详情】\n")
                        for factor in result.risk_factors:
                            f.write(f"   • {factor}\n")
            else:
                f.write("  未发现相似文档对\n")

            f.write("\n" + "=" * 80 + "\n")

            f.write("【单文档风险评估】\n")
            f.write("说明：只要存在一篇极其相似的文件，该文件评级即为高风险\n\n")
            
            for doc_id, profile in report.file_profiles.items():
                f.write(f"• {profile.filename}\n")
                if profile.max_risk_level == "HIGH":
                    f.write(f"  评级: 🔴 高风险\n")
                    f.write(f"  关联相似文档数: {profile.related_suspicious_count}\n")
                elif profile.max_risk_level == "LOW":
                    f.write(f"  评级: 🟡 低风险\n")
                    f.write(f"  关联相似文档数: {profile.related_suspicious_count}\n")
                else:
                    f.write(f"  评级: 🟢 无风险\n")
                    f.write(f"  关联相似文档数: {0}\n")

            f.write("\n" + "=" * 80 + "\n")

        logger.info(f"摘要报告已生成: {summary_path}")
    
    def _format_similarity(self, similarity: float) -> str:
        """格式化相似度描述"""
        if similarity >= 0.95:
            return "极度相似"
        elif similarity >= 0.85:
            return "高度相似"
        elif similarity >= 0.70:
            return "中度相似"
        elif similarity >= 0.50:
            return "轻度相似"
        else:
            return "基本不相似"

    def _generate_csv_report(self, report: GlobalReport, output_dir: str) -> None:
        """生成CSV格式的可疑对列表 - 简化版"""
        csv_path = os.path.join(output_dir, "suspicious_pairs.csv")

        all_results = report.pairwise_results
        all_results.sort(key=lambda x: x.risk_score, reverse=True)

        with open(csv_path, 'w', encoding='utf-8-sig') as f:
            f.write("文档A,文档B,相似度,串标围标风险评级,相似段落数,连续克隆块数,检测详情\n")

            for result in all_results:
                text_local = result.similarity_scores.get('text_local', 0)
                if text_local < 0.4:
                    continue

                profile_a = report.file_profiles.get(result.doc_a_id)
                profile_b = report.file_profiles.get(result.doc_b_id)

                filename_a = profile_a.filename if profile_a else result.doc_a_id
                filename_b = profile_b.filename if profile_b else result.doc_b_id

                evidence = result.evidence
                para_count = len(evidence.text_evidence.paragraph_matches)
                clone_count = len(evidence.text_evidence.continuous_clone_blocks)
                risk_factors_str = "; ".join(result.risk_factors)

                f.write(f'"{filename_a}","{filename_b}",{text_local:.4f},{result.risk_level},'
                       f'{para_count},{clone_count},"{risk_factors_str}"\n')

        logger.info(f"CSV报告已生成: {csv_path}")
