"""
模块 E：报告生成引擎
- 输出 JSON 格式检测数据
- 输出 PDF 可视化报告
"""
import os
import json
import logging
from typing import Any

from data_structures import GlobalReport
from config import DetectionConfig
from pdf_report import generate_pdf

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

        # 2. 生成PDF可视化报告
        try:
            self._generate_pdf_report(report, output_dir)
        except Exception as e:
            logger.warning(f"PDF报告生成失败，已保留JSON报告: {e}", exc_info=True)

        logger.info(f"报告已生成到: {output_dir}")

    def _generate_json_report(self, report: GlobalReport, output_dir: str) -> None:
        """生成JSON格式的完整报告"""
        json_path = os.path.join(output_dir, "detection_report.json")
        report_dict = self._dataclass_to_dict(report)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"JSON报告已生成: {json_path}")

    def _generate_pdf_report(self, report: GlobalReport, output_dir: str) -> None:
        """生成PDF可视化报告"""
        pdf_path = os.path.join(output_dir, "detection_report.pdf")
        generate_pdf(report, pdf_path, enabled_dims=self.config.ENABLED_DIMENSIONS)

    def _dataclass_to_dict(self, obj: Any) -> Any:
        """递归转换dataclass为字典（bytes->base64）"""
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
