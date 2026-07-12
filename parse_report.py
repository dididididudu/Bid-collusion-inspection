#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""解析 detection_report.json 并输出可读摘要"""
import json
import sys
import os

report_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "output", "detection_report.json"
)

with open(report_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("=" * 70)
print("检测报告摘要")
print("=" * 70)
print(f"报告ID: {data['report_id']}")
print(f"生成时间: {data['generated_at']}")
print(f"总文件数: {data['total_files']}")
print(f"总对数: {data['total_pairs']}")
print(f"候选对数: {data['candidate_pairs']}")
print(f"可疑对数: {data['suspicious_pairs']}")
print(f"高风险对数: {data['high_risk_pairs']}")
print(f"风险聚类数: {len(data.get('risk_clusters', []))}")
print(f"元数据聚合组数: {len(data.get('metadata_groups', []))}")
print(f"文档对结果数: {len(data.get('pairwise_results', []))}")

print("\n" + "=" * 70)
print("文档对结果详情")
print("=" * 70)
for i, pr in enumerate(data.get("pairwise_results", []), 1):
    te = pr.get("evidence", {}).get("text_evidence", {})
    ie = pr.get("evidence", {}).get("image_evidence", {})
    me = pr.get("evidence", {}).get("metadata_evidence", {})
    ce = pr.get("evidence", {}).get("contact_evidence", {})
    pm_list = te.get("paragraph_matches", [])

    print(f"\n--- 对 {i}: {pr['doc_a_id'][:12]} vs {pr['doc_b_id'][:12]} ---")
    print(f"  风险等级: {pr['risk_level']}, 风险分: {pr['risk_score']}")
    print(f"  风险因子: {pr.get('risk_factors', [])}")
    print(f"  文本相似度: {pr.get('similarity_scores', {}).get('text_local', 0):.4f}")
    print(f"  段落匹配数: {len(pm_list)}")
    print(f"  图片匹配数: {ie.get('common_image_count', 0)}")
    print(f"  元数据匹配字段: {me.get('matched_fields', [])}")
    print(f"  相同文件ID: {me.get('same_file_id', False)}")

    # 连续克隆块
    clone_blocks = te.get("continuous_clone_blocks", [])
    if clone_blocks:
        print(f"  连续克隆块: {len(clone_blocks)} 组")
        for cb in clone_blocks[:3]:
            print(f"    组 {cb.get('group_id','')}: {cb.get('paragraph_count',0)} 段, 相似度 {cb.get('avg_similarity',0):.4f}")

    # 段落匹配详情（前5条）
    if pm_list:
        print(f"  段落匹配详情（前5条）:")
        for j, pm in enumerate(pm_list[:5], 1):
            pa = pm.get("paragraph_a", "")[:80]
            pb = pm.get("paragraph_b", "")[:80]
            sim = pm.get("similarity", 0)
            method = pm.get("detection_method", "")
            clone = " [连续克隆]" if pm.get("is_continuous_clone") else ""
            print(f"    {j}. sim={sim:.4f} method={method}{clone}")
            print(f"       A: {pa}...")
            print(f"       B: {pb}...")

    # 图片匹配详情（前3条）
    img_pairs = ie.get("matched_image_pairs", [])
    if img_pairs:
        print(f"  图片匹配详情（前3条）:")
        for j, ip in enumerate(img_pairs[:3], 1):
            print(f"    {j}. {ip.get('source_a','')} vs {ip.get('source_b','')}")
            print(f"       confidence={ip.get('confidence',0):.4f}, reasons={ip.get('reasons',[])}")
            if ip.get("ocr_text_a"):
                print(f"       OCR_A: {ip.get('ocr_text_a','')[:60]}")
            if ip.get("ocr_text_b"):
                print(f"       OCR_B: {ip.get('ocr_text_b','')[:60]}")

print("\n" + "=" * 70)
print("元数据聚合组")
print("=" * 70)
for mg in data.get("metadata_groups", []):
    print(f"  [{mg['group_type']}] \"{mg['shared_value']}\"")
    print(f"    -> {mg['doc_count']}个文档: {mg.get('filenames', [])}")

print("\n" + "=" * 70)
print("单文档风险等级")
print("=" * 70)
for doc_id, risk in data.get("single_doc_risks", {}).items():
    print(f"  {doc_id[:12]}: {risk}")

print("\n" + "=" * 70)
print("错误日志")
print("=" * 70)
errors = data.get("error_log", [])
if errors:
    for e in errors[:10]:
        print(f"  {e}")
else:
    print("  (无错误)")
