#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TECH_BID_SIMILAR evidence 结构单元测试 — 直接测试证据组装逻辑
无需运行完整管道（30+ 分钟），使用 mock 数据验证 evidence 字段结构。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(line_buffering=True)

from collusive_check_api import (
    CompanyInfo, _get_company_results_from_pipeline,
)


def make_mock_pipeline_result(dim: str = "technical"):
    """构造包含段落匹配和图片匹配的 mock 管道结果"""
    return {
        "dimension": dim,
        "text_results": {
            "701_702": {
                "similarity": 0.8723,
                "tech_similarity": 5,
                "com_similarity": 0,
                "has_tech_match": True,
                "has_com_match": False,
                "pair_ids": {"a": 701, "b": 702},
                "paragraph_matches": [
                    {
                        "paragraph_a": "本项目施工方案采用先进的预制装配式建筑技术，预制率达到 40%以上。",
                        "paragraph_b": "本项目施工方案采用先进的预制装配式建筑技术，预制率达到 40%以上。",
                        "similarity": 0.9856,
                        "paragraph_a_index": 12,
                        "paragraph_b_index": 8,
                        "detection_method": "SBERT",
                    },
                    {
                        "paragraph_a": "安全管理体系：建立以项目经理为第一责任人的安全生产管理体系。",
                        "paragraph_b": "安全管理体系：建立以项目经理为第一责任人的安全生产管理体系。",
                        "similarity": 0.9712,
                        "paragraph_a_index": 15,
                        "paragraph_b_index": 11,
                        "detection_method": "SBERT",
                    },
                ],
            }
        },
        "image_results": {
            "701_702": {
                "total_image_matches": 3,
                "tech_image_matches": 3,
                "com_image_matches": 0,
                "has_tech_image": True,
                "has_com_image": False,
                "pair_ids": {"a": 701, "b": 702},
                "matched_image_pairs": [
                    {
                        "source_a": "page5_img0.png",
                        "source_b": "page3_img0.png",
                        "confidence": 0.9234,
                        "reasons": ["phash_match", "orb_match"],
                        "ocr_text_a": "施工平面布置图",
                        "ocr_text_b": "施工平面布置图",
                    },
                    {
                        "source_a": "page12_img1.png",
                        "source_b": "page10_img1.png",
                        "confidence": 0.8812,
                        "reasons": ["phash_match", "histogram_match"],
                        "ocr_text_a": "组织架构图",
                        "ocr_text_b": "组织架构图",
                    },
                ],
            }
        },
    }


def test_tech_bid_evidence_structure():
    """验证 TECH_BID_SIMILAR FAILED evidence 结构"""
    print("=" * 70, flush=True)
    print("[单元测试] TECH_BID_SIMILAR evidence 结构验证（mock 数据）", flush=True)
    print("=" * 70, flush=True)

    companies = [
        CompanyInfo(
            companyRecordId=701, registrationCompanyId=301,
            sectionId=11, bidderName="测试公司A", bidFileUrl="http://example.com/a.pdf",
        ),
        CompanyInfo(
            companyRecordId=702, registrationCompanyId=302,
            sectionId=11, bidderName="测试公司B", bidFileUrl="http://example.com/b.pdf",
        ),
    ]

    mock_result = make_mock_pipeline_result("technical")
    results = _get_company_results_from_pipeline(companies, mock_result, "TECH_BID_SIMILAR")

    assert len(results) == 2, f"应返回 2 个结果，实际 {len(results)}"

    for r in results:
        print(f"\n  公司 {r.companyRecordId}: status={r.status}", flush=True)
        print(f"    summary: {r.summary}", flush=True)

        assert r.status == "FAILED", f"应为 FAILED，实际 {r.status}"
        ev = r.evidence
        print(f"    evidence keys: {list(ev.keys())}", flush=True)

        # 验证顶层字段
        assert "similarCompanyRecordIds" in ev, f"缺少 similarCompanyRecordIds"
        assert "similarParagraphs" in ev, f"缺少 similarParagraphs"
        assert "similarImages" in ev, f"缺少 similarImages"

        similar_ids = ev["similarCompanyRecordIds"]
        print(f"    similarCompanyRecordIds: {similar_ids}", flush=True)
        assert len(similar_ids) > 0, "similarCompanyRecordIds 不能为空"
        assert r.companyRecordId not in similar_ids, "不应包含自身 ID"

        # 验证 similarParagraphs 结构
        sp_list = ev["similarParagraphs"]
        print(f"    similarParagraphs: {len(sp_list)} 条", flush=True)
        assert len(sp_list) > 0, "similarParagraphs 不能为空"

        sp = sp_list[0]
        print(f"    第一条 keys: {list(sp.keys())}", flush=True)
        assert "companyRecordId" in sp, f"段落缺少 companyRecordId"
        assert "similarity" in sp, f"段落缺少 similarity"
        assert "paragraphMatches" in sp, f"段落缺少 paragraphMatches"

        assert sp["companyRecordId"] != r.companyRecordId, "companyRecordId 不应是自身"
        assert sp["similarity"] > 0, "similarity 应大于 0"
        print(f"    companyRecordId={sp['companyRecordId']}, similarity={sp['similarity']}", flush=True)

        pm_list = sp["paragraphMatches"]
        print(f"    paragraphMatches: {len(pm_list)} 对", flush=True)
        assert len(pm_list) > 0, "paragraphMatches 不能为空"

        for i, pm in enumerate(pm_list):
            print(f"    段落对 {i+1} keys: {list(pm.keys())}", flush=True)
            assert "paragraph_a" in pm, f"缺少 paragraph_a"
            assert "paragraph_b" in pm, f"缺少 paragraph_b"
            assert "similarity" in pm, f"缺少 similarity"
            assert "paragraph_a_index" in pm, f"缺少 paragraph_a_index"
            assert "paragraph_b_index" in pm, f"缺少 paragraph_b_index"
            assert "detection_method" in pm, f"缺少 detection_method"

            assert len(pm["paragraph_a"]) > 0, "paragraph_a 不能为空"
            assert len(pm["paragraph_b"]) > 0, "paragraph_b 不能为空"
            assert pm["similarity"] > 0, "段落相似度应大于 0"
            assert pm["detection_method"] in ("SBERT", "Jaccard", "Exact", "SequenceMatcher"), \
                f"未知检测方法: {pm['detection_method']}"

            print(f"      paragraph_a (前60字): {pm['paragraph_a'][:60]}...", flush=True)
            print(f"      paragraph_b (前60字): {pm['paragraph_b'][:60]}...", flush=True)
            print(f"      similarity={pm['similarity']}, method={pm['detection_method']}", flush=True)

        print(f"    ✓ 段落 evidence 结构正确", flush=True)

        # 验证 similarImages 结构
        si_list = ev["similarImages"]
        print(f"    similarImages: {len(si_list)} 条", flush=True)
        assert len(si_list) > 0, "similarImages 不能为空"

        si = si_list[0]
        print(f"    第一条 keys: {list(si.keys())}", flush=True)
        assert "companyRecordId" in si, f"图片缺少 companyRecordId"
        assert "imageMatchCount" in si, f"图片缺少 imageMatchCount"
        assert "similarImages" in si, f"图片缺少 similarImages"

        assert si["companyRecordId"] != r.companyRecordId, "companyRecordId 不应是自身"
        assert si["imageMatchCount"] > 0, "imageMatchCount 应大于 0"
        print(f"    companyRecordId={si['companyRecordId']}, imageMatchCount={si['imageMatchCount']}", flush=True)

        img_list = si["similarImages"]
        print(f"    图片对: {len(img_list)} 对", flush=True)
        assert len(img_list) > 0, "图片对不能为空"

        for i, img in enumerate(img_list):
            print(f"    图片对 {i+1} keys: {list(img.keys())}", flush=True)
            assert "source_a" in img, f"缺少 source_a"
            assert "source_b" in img, f"缺少 source_b"
            assert "confidence" in img, f"缺少 confidence"
            assert "reasons" in img, f"缺少 reasons"
            assert "ocr_text_a" in img, f"缺少 ocr_text_a"
            assert "ocr_text_b" in img, f"缺少 ocr_text_b"

            assert len(img["source_a"]) > 0, "source_a 不能为空"
            assert len(img["source_b"]) > 0, "source_b 不能为空"
            assert img["confidence"] > 0, "confidence 应大于 0"
            assert isinstance(img["reasons"], list), "reasons 应为列表"

            print(f"      source_a: {img['source_a']}", flush=True)
            print(f"      source_b: {img['source_b']}", flush=True)
            print(f"      confidence={img['confidence']}, reasons={img['reasons']}", flush=True)
            print(f"      ocr_text_a: {img['ocr_text_a']}", flush=True)
            print(f"      ocr_text_b: {img['ocr_text_b']}", flush=True)

        print(f"    ✓ 图片 evidence 结构正确", flush=True)

    print(f"\n✓ TECH_BID_SIMILAR evidence 结构验证通过 (2/2 FAILED)", flush=True)


def test_success_empty_evidence():
    """验证无相似时 SUCCESS + 空 evidence"""
    print("\n" + "=" * 70, flush=True)
    print("[单元测试] 无相似时 SUCCESS + 空 evidence 验证", flush=True)
    print("=" * 70, flush=True)

    companies = [
        CompanyInfo(
            companyRecordId=801, registrationCompanyId=401,
            sectionId=11, bidderName="测试公司C", bidFileUrl="http://example.com/c.pdf",
        ),
        CompanyInfo(
            companyRecordId=802, registrationCompanyId=402,
            sectionId=11, bidderName="测试公司D", bidFileUrl="http://example.com/d.pdf",
        ),
    ]

    mock_result = {
        "dimension": "technical",
        "text_results": {},
        "image_results": {},
    }
    results = _get_company_results_from_pipeline(companies, mock_result, "TECH_BID_SIMILAR")

    for r in results:
        print(f"  公司 {r.companyRecordId}: status={r.status}, evidence={r.evidence}", flush=True)
        assert r.status == "SUCCESS", f"应为 SUCCESS，实际 {r.status}"
        assert len(r.evidence) == 0, f"evidence 应为空，实际 {r.evidence}"

    print(f"\n✓ SUCCESS 空 evidence 验证通过 (2/2 SUCCESS)", flush=True)


def test_commercial_bid_evidence():
    """验证 BID_COMPANY_NAME_ABNORMAL (商务标) evidence 结构"""
    print("\n" + "=" * 70, flush=True)
    print("[单元测试] BID_COMPANY_NAME_ABNORMAL evidence 结构验证", flush=True)
    print("=" * 70, flush=True)

    companies = [
        CompanyInfo(
            companyRecordId=701, registrationCompanyId=301,
            sectionId=11, bidderName="测试公司E", bidFileUrl="http://example.com/e.pdf",
        ),
        CompanyInfo(
            companyRecordId=702, registrationCompanyId=302,
            sectionId=11, bidderName="测试公司F", bidFileUrl="http://example.com/f.pdf",
        ),
    ]

    mock_result = make_mock_pipeline_result("commercial")
    results = _get_company_results_from_pipeline(companies, mock_result, "BID_COMPANY_NAME_ABNORMAL")

    for r in results:
        print(f"  公司 {r.companyRecordId}: status={r.status}", flush=True)
        assert r.status == "FAILED", f"应为 FAILED，实际 {r.status}"
        ev = r.evidence
        assert "similarCompanyRecordIds" in ev
        assert "similarParagraphs" in ev
        assert "similarImages" in ev
        print(f"    ✓ 商务标 evidence 结构正确: {list(ev.keys())}", flush=True)

    print(f"\n✓ BID_COMPANY_NAME_ABNORMAL evidence 验证通过 (2/2 FAILED)", flush=True)


if __name__ == "__main__":
    print("=" * 70, flush=True)
    print("TECH_BID_SIMILAR / BID_COMPANY_NAME_ABNORMAL evidence 单元测试", flush=True)
    print("=" * 70, flush=True)

    all_pass = True
    try:
        test_tech_bid_evidence_structure()
    except Exception as e:
        all_pass = False
        print(f"✗ 技术标 evidence 测试失败: {e}", flush=True)
        import traceback
        traceback.print_exc()

    try:
        test_success_empty_evidence()
    except Exception as e:
        all_pass = False
        print(f"✗ SUCCESS 空 evidence 测试失败: {e}", flush=True)
        import traceback
        traceback.print_exc()

    try:
        test_commercial_bid_evidence()
    except Exception as e:
        all_pass = False
        print(f"✗ 商务标 evidence 测试失败: {e}", flush=True)
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70, flush=True)
    if all_pass:
        print("✓ 所有单元测试通过", flush=True)
    else:
        print("✗ 部分测试失败", flush=True)
    print("=" * 70, flush=True)

    sys.exit(0 if all_pass else 1)
