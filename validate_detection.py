"""
相似度检测验证脚本

用于验证：
1. 所有相似段落是否被正确识别
2. 相似度计算是否准确
3. 是否有遗漏的相似内容
"""

import json
import logging
from typing import Dict, List, Set, Tuple
from difflib import SequenceMatcher

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_report(report_path: str) -> dict:
    """加载检测报告"""
    with open(report_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def validate_similarity_accuracy(report: dict) -> None:
    """验证相似度计算的准确性"""
    logger.info("=" * 80)
    logger.info("验证相似度计算准确性")
    logger.info("=" * 80)

    for result in report['pairwise_results']:
        pair_id = result['pair_id']
        text_similarity = result['similarity_scores'].get('text_local', 0)

        # 获取段落匹配
        paragraph_matches = result['evidence']['text_evidence']['paragraph_matches']

        if not paragraph_matches:
            continue

        # 验证每个段落匹配的相似度
        mismatches = []
        for match in paragraph_matches:
            reported_sim = match.get('similarity', 0)
            para_a = match.get('paragraph_a', '')
            para_b = match.get('paragraph_b', '')

            if not para_a or not para_b:
                continue

            # 使用SequenceMatcher重新计算相似度
            actual_sim = SequenceMatcher(None, para_a, para_b).ratio()

            # 允许一定误差范围（因为不同算法）
            if abs(reported_sim - actual_sim) > 0.3:
                mismatches.append({
                    'a_index': match.get('paragraph_a_index'),
                    'b_index': match.get('paragraph_b_index'),
                    'reported': reported_sim,
                    'actual': actual_sim,
                    'method': match.get('detection_method'),
                })

        if mismatches:
            logger.warning(f"\n文档对 {pair_id} 发现相似度偏差:")
            for m in mismatches[:5]:  # 只显示前5个
                logger.warning(f"  段落 [{m['a_index']}↔{m['b_index']}]: "
                             f"报告={m['reported']:.3f}, 实际={m['actual']:.3f}, "
                             f"方法={m['method']}")


def validate_completeness(report: dict) -> None:
    """验证是否有遗漏的相似内容"""
    logger.info("\n" + "=" * 80)
    logger.info("验证内容完整性")
    logger.info("=" * 80)

    for result in report['pairwise_results']:
        pair_id = result['pair_id']
        paragraph_matches = result['evidence']['text_evidence']['paragraph_matches']

        logger.info(f"\n文档对: {pair_id}")
        logger.info(f"  检测到的相似段落对数: {len(paragraph_matches)}")

        # 统计相似度分布
        if paragraph_matches:
            similarities = [m.get('similarity', 0) for m in paragraph_matches]
            logger.info(f"  相似度范围: {min(similarities):.3f} - {max(similarities):.3f}")
            logger.info(f"  平均相似度: {sum(similarities)/len(similarities):.3f}")

            # 按相似度分段统计
            high_sim = sum(1 for s in similarities if s >= 0.9)
            medium_sim = sum(1 for s in similarities if 0.7 <= s < 0.9)
            low_sim = sum(1 for s in similarities if s < 0.7)

            logger.info(f"  高相似度(≥0.9): {high_sim} 对")
            logger.info(f"  中相似度(0.7-0.9): {medium_sim} 对")
            logger.info(f"  低相似度(<0.7): {low_sim} 对")

            # 检查检测方法分布
            methods = {}
            for m in paragraph_matches:
                method = m.get('detection_method', 'Unknown')
                methods[method] = methods.get(method, 0) + 1

            logger.info(f"  检测方法分布:")
            for method, count in methods.items():
                logger.info(f"    {method}: {count} 对")


def validate_clone_blocks(report: dict) -> None:
    """验证连续克隆块检测"""
    logger.info("\n" + "=" * 80)
    logger.info("验证连续克隆块检测")
    logger.info("=" * 80)

    for result in report['pairwise_results']:
        pair_id = result['pair_id']
        clone_blocks = result['evidence']['text_evidence']['continuous_clone_blocks']

        if not clone_blocks:
            continue

        logger.info(f"\n文档对: {pair_id}")
        logger.info(f"  检测到 {len(clone_blocks)} 个连续克隆块")

        for block in clone_blocks[:10]:  # 只显示前10个
            logger.info(f"    克隆块 {block['group_id']}: "
                       f"长度={block['length']}, "
                       f"平均相似度={block['similarity']:.3f}")


def check_report_limits(report: dict) -> None:
    """检查是否有内容因限制被截断"""
    logger.info("\n" + "=" * 80)
    logger.info("检查报告限制")
    logger.info("=" * 80)

    total_pairs = len(report['pairwise_results'])
    logger.info(f"总文档对数: {total_pairs}")

    for result in report['pairwise_results']:
        pair_id = result['pair_id']
        paragraph_matches = result['evidence']['text_evidence']['paragraph_matches']

        # 检查是否达到了配置的限制
        if len(paragraph_matches) >= 10000:
            logger.warning(f"⚠️ 文档对 {pair_id} 的匹配数达到上限 {len(paragraph_matches)}，可能有内容被截断")
        elif len(paragraph_matches) >= 200:
            logger.info(f"✓ 文档对 {pair_id} 有 {len(paragraph_matches)} 对匹配（已提高限制）")


def generate_validation_summary(report: dict) -> None:
    """生成验证摘要"""
    logger.info("\n" + "=" * 80)
    logger.info("验证摘要")
    logger.info("=" * 80)

    total_files = report['total_files']
    total_pairs = report['total_pairs']
    suspicious_pairs = report['suspicious_pairs']
    high_risk_pairs = report['high_risk_pairs']

    logger.info(f"检测文件总数: {total_files}")
    logger.info(f"比对总对数: {total_pairs}")
    logger.info(f"可疑对数: {suspicious_pairs}")
    logger.info(f"高风险对数: {high_risk_pairs}")

    # 统计所有匹配的段落总数
    total_matches = 0
    for result in report['pairwise_results']:
        paragraph_matches = result['evidence']['text_evidence']['paragraph_matches']
        total_matches += len(paragraph_matches)

    logger.info(f"检测到的相似段落总数: {total_matches}")

    if suspicious_pairs > 0:
        avg_matches = total_matches / suspicious_pairs
        logger.info(f"平均每对文档的相似段落数: {avg_matches:.1f}")

    logger.info("\n✓ 验证完成")


def main(report_path: str):
    """主验证流程"""
    logger.info("开始验证检测报告...")
    logger.info(f"报告路径: {report_path}\n")

    # 加载报告
    report = load_report(report_path)

    # 执行各项验证
    validate_completeness(report)
    validate_clone_blocks(report)
    validate_similarity_accuracy(report)
    check_report_limits(report)
    generate_validation_summary(report)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python validate_detection.py <report_json_path>")
        print("示例: python validate_detection.py test_data/output/detection_report.json")
        sys.exit(1)

    report_path = sys.argv[1]
    main(report_path)
