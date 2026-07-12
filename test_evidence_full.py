#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
触发 FAILED 证据验证 — 用同一份 PDF 作为两家公司，强制触发文件码/作者雷同。
同时提交 TECH_BID_SIMILAR（后台运行）验证段落+图片 evidence 结构。
"""
import sys
import json
import time
import requests

sys.stdout.reconfigure(line_buffering=True)

API = "http://127.0.0.1:8001/api/v1/collusive-check/items/analyze"
PDF_URL = "http://127.0.0.1:18080/%E6%8A%95%E6%A0%87%E6%96%87%E4%BB%B6.pdf"

# 两家公司使用同一份 PDF → 必然触发文件码/作者雷同
COMPANIES_SAME = [
    {
        "companyRecordId": 601,
        "registrationCompanyId": 201,
        "sectionId": 11,
        "bidderName": "测试公司X",
        "bidFileUrl": PDF_URL,
    },
    {
        "companyRecordId": 602,
        "registrationCompanyId": 202,
        "sectionId": 11,
        "bidderName": "测试公司Y",
        "bidFileUrl": PDF_URL,
    },
]

# 两家公司使用不同 PDF → 用于技术标/商务标检测
COMPANIES_DIFF = [
    {
        "companyRecordId": 701,
        "registrationCompanyId": 301,
        "sectionId": 11,
        "bidderName": "测试公司A",
        "bidFileUrl": "http://127.0.0.1:18080/%E6%8A%95%E6%A0%87%E6%96%87%E4%BB%B6-%E9%9B%80%E7%BF%BC0828.pdf",
    },
    {
        "companyRecordId": 702,
        "registrationCompanyId": 302,
        "sectionId": 11,
        "bidderName": "测试公司B",
        "bidFileUrl": "http://127.0.0.1:18080/%E6%8A%95%E6%A0%87%E6%96%87%E4%BB%B6.pdf",
    },
]


def run(item_code: str, companies, timeout: int = 300) -> dict:
    payload = {
        "batchId": int(time.time()) % 100000,
        "projectId": 1,
        "checkMode": "SAME_SECTION",
        "itemCode": item_code,
        "companies": companies,
    }
    print(f"\n>>> 调用 {item_code} ({len(companies)} 家公司) ...", flush=True)
    t0 = time.time()
    resp = requests.post(API, json=payload, timeout=timeout)
    dt = time.time() - t0
    print(f"<<< HTTP {resp.status_code} 耗时 {dt:.2f}s", flush=True)
    assert resp.status_code == 200, resp.text[:500]
    return resp.json()


def verify_file_code_failed():
    """验证 FILE_CODE_SIMILAR FAILED evidence 结构（同文件 → 必然雷同）"""
    print("=" * 70, flush=True)
    print("[测试A] FILE_CODE_SIMILAR FAILED evidence 验证（同一 PDF）", flush=True)
    print("=" * 70, flush=True)
    data = run("FILE_CODE_SIMILAR", COMPANIES_SAME)

    failed = 0
    for r in data.get("results", []):
        cid = r["companyRecordId"]
        status = r["status"]
        ev = r.get("evidence", {})
        print(f"  公司 {cid}: status={status}", flush=True)
        print(f"    summary: {r.get('summary', '')}", flush=True)
        print(f"    evidence: {json.dumps(ev, ensure_ascii=False)}", flush=True)

        if status == "FAILED":
            failed += 1
            assert "fileId" in ev, f"缺少 fileId: {ev}"
            assert "similarCompanyRecordIds" in ev, f"缺少 similarCompanyRecordIds: {ev}"
            assert len(ev["fileId"]) > 0, "fileId 不能为空"
            assert len(ev["similarCompanyRecordIds"]) > 0, "similarCompanyRecordIds 不能为空"
            print(f"    ✓ fileId={ev['fileId'][:16]}... similarIds={ev['similarCompanyRecordIds']}", flush=True)

    assert failed == 2, f"应有两个 FAILED 结果，实际 {failed}"
    print(f"\n✓ FILE_CODE_SIMILAR FAILED 验证通过 (2/2 FAILED)", flush=True)


def verify_doc_author_failed():
    """验证 DOC_AUTHOR_SIMILAR FAILED evidence 结构（同文件 → 作者必然相同）"""
    print("\n" + "=" * 70, flush=True)
    print("[测试B] DOC_AUTHOR_SIMILAR FAILED evidence 验证（同一 PDF）", flush=True)
    print("=" * 70, flush=True)
    data = run("DOC_AUTHOR_SIMILAR", COMPANIES_SAME)

    failed = 0
    for r in data.get("results", []):
        cid = r["companyRecordId"]
        status = r["status"]
        ev = r.get("evidence", {})
        print(f"  公司 {cid}: status={status}", flush=True)
        print(f"    summary: {r.get('summary', '')}", flush=True)
        print(f"    evidence: {json.dumps(ev, ensure_ascii=False)}", flush=True)

        if status == "FAILED":
            failed += 1
            assert "author" in ev, f"缺少 author: {ev}"
            assert "similarCompanyRecordIds" in ev, f"缺少 similarCompanyRecordIds: {ev}"
            assert len(ev["author"]) > 0, "author 不能为空"
            assert len(ev["similarCompanyRecordIds"]) > 0, "similarCompanyRecordIds 不能为空"
            print(f"    ✓ author='{ev['author']}' similarIds={ev['similarCompanyRecordIds']}", flush=True)

    assert failed == 2, f"应有两个 FAILED 结果，实际 {failed}"
    print(f"\n✓ DOC_AUTHOR_SIMILAR FAILED 验证通过 (2/2 FAILED)", flush=True)


def verify_tech_bid_evidence(timeout: int = 1800):
    """验证 TECH_BID_SIMILAR evidence 结构（段落内容 + 图片引用）"""
    print("\n" + "=" * 70, flush=True)
    print("[测试C] TECH_BID_SIMILAR evidence 验证（段落+图片）", flush=True)
    print("=" * 70, flush=True)
    data = run("TECH_BID_SIMILAR", COMPANIES_DIFF, timeout=timeout)

    print(f"itemCode={data.get('itemCode')}, itemName={data.get('itemName')}", flush=True)
    failed = 0
    for r in data.get("results", []):
        cid = r["companyRecordId"]
        status = r["status"]
        ev = r.get("evidence", {})
        print(f"\n  公司 {cid}: status={status}", flush=True)
        print(f"    summary: {r.get('summary', '')}", flush=True)

        if status == "FAILED":
            failed += 1
            print(f"    evidence keys: {list(ev.keys())}", flush=True)
            assert "similarCompanyRecordIds" in ev, f"缺少 similarCompanyRecordIds: {ev.keys()}"
            assert "similarParagraphs" in ev, f"缺少 similarParagraphs: {ev.keys()}"
            assert "similarImages" in ev, f"缺少 similarImages: {ev.keys()}"

            # 验证段落结构
            sp_list = ev.get("similarParagraphs", [])
            print(f"    similarParagraphs: {len(sp_list)} 条", flush=True)
            if sp_list:
                sp = sp_list[0]
                print(f"    第一条段落详情 keys: {list(sp.keys())}", flush=True)
                assert "companyRecordId" in sp, f"段落缺少 companyRecordId: {sp.keys()}"
                assert "similarity" in sp, f"段落缺少 similarity: {sp.keys()}"
                assert "paragraphMatches" in sp, f"段落缺少 paragraphMatches: {sp.keys()}"

                pm_list = sp.get("paragraphMatches", [])
                print(f"    paragraphMatches: {len(pm_list)} 对", flush=True)
                if pm_list:
                    pm = pm_list[0]
                    print(f"    第一对段落 keys: {list(pm.keys())}", flush=True)
                    assert "paragraph_a" in pm, f"缺少 paragraph_a: {pm.keys()}"
                    assert "paragraph_b" in pm, f"缺少 paragraph_b: {pm.keys()}"
                    assert "similarity" in pm, f"缺少 similarity: {pm.keys()}"
                    pa = pm.get("paragraph_a", "")
                    pb = pm.get("paragraph_b", "")
                    print(f"    paragraph_a (前80字): {pa[:80]}...", flush=True)
                    print(f"    paragraph_b (前80字): {pb[:80]}...", flush=True)
                    assert len(pa) > 0, "paragraph_a 不能为空"
                    assert len(pb) > 0, "paragraph_b 不能为空"
                    print(f"    ✓ 段落 evidence 结构正确", flush=True)

            # 验证图片结构
            si_list = ev.get("similarImages", [])
            print(f"    similarImages: {len(si_list)} 条", flush=True)
            if si_list:
                si = si_list[0]
                print(f"    第一条图片详情 keys: {list(si.keys())}", flush=True)
                assert "companyRecordId" in si, f"图片缺少 companyRecordId: {si.keys()}"
                assert "imageMatchCount" in si, f"图片缺少 imageMatchCount: {si.keys()}"
                assert "similarImages" in si, f"图片缺少 similarImages: {si.keys()}"

                img_list = si.get("similarImages", [])
                print(f"    图片对: {len(img_list)} 对", flush=True)
                if img_list:
                    img = img_list[0]
                    print(f"    第一对图片 keys: {list(img.keys())}", flush=True)
                    assert "source_a" in img, f"缺少 source_a: {img.keys()}"
                    assert "source_b" in img, f"缺少 source_b: {img.keys()}"
                    assert "confidence" in img, f"缺少 confidence: {img.keys()}"
                    print(f"    source_a: {img.get('source_a', '')}", flush=True)
                    print(f"    source_b: {img.get('source_b', '')}", flush=True)
                    print(f"    confidence: {img.get('confidence', 0)}", flush=True)
                    print(f"    ✓ 图片 evidence 结构正确", flush=True)

    print(f"\n  汇总: {failed} FAILED, {len(data.get('results', [])) - failed} SUCCESS", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-heavy", action="store_true", help="跳过重量级 TECH_BID 测试")
    parser.add_argument("--heavy-timeout", type=int, default=1800)
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("evidence 字段完整验证 — 触发 FAILED 结果 + 段落/图片详情", flush=True)
    print("=" * 70, flush=True)

    all_pass = True
    try:
        verify_file_code_failed()
    except Exception as e:
        all_pass = False
        print(f"✗ FILE_CODE FAILED 验证失败: {e}", flush=True)

    try:
        verify_doc_author_failed()
    except Exception as e:
        all_pass = False
        print(f"✗ DOC_AUTHOR FAILED 验证失败: {e}", flush=True)

    if not args.skip_heavy:
        try:
            verify_tech_bid_evidence(timeout=args.heavy_timeout)
        except Exception as e:
            all_pass = False
            print(f"✗ TECH_BID 验证失败: {e}", flush=True)
    else:
        print("\n[跳过] TECH_BID_SIMILAR 重量级测试", flush=True)

    print("\n" + "=" * 70, flush=True)
    if all_pass:
        print("✓ 所有 evidence 测试通过", flush=True)
    else:
        print("✗ 部分测试失败", flush=True)
    print("=" * 70, flush=True)

    sys.exit(0 if all_pass else 1)
