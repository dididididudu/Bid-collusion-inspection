#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
轻量级 evidence 字段验证 — 直接调用 API 验证 FILE_CODE_SIMILAR 和 DOC_AUTHOR_SIMILAR
使用 unbuffered 输出，确保日志实时写入。
"""
import sys
import json
import time
import requests

# 强制 unbuffered
sys.stdout.reconfigure(line_buffering=True)

API = "http://127.0.0.1:8001/api/v1/collusive-check/items/analyze"

COMPANIES = [
    {
        "companyRecordId": 501,
        "registrationCompanyId": 101,
        "sectionId": 11,
        "bidderName": "测试公司A",
        "bidFileUrl": "http://127.0.0.1:18080/%E6%8A%95%E6%A0%87%E6%96%87%E4%BB%B6-%E9%9B%80%E7%BF%BC0828.pdf",
    },
    {
        "companyRecordId": 502,
        "registrationCompanyId": 102,
        "sectionId": 11,
        "bidderName": "测试公司B",
        "bidFileUrl": "http://127.0.0.1:18080/%E6%8A%95%E6%A0%87%E6%96%87%E4%BB%B6.pdf",
    },
]


def run(item_code: str) -> dict:
    payload = {
        "batchId": int(time.time()) % 100000,
        "projectId": 1,
        "checkMode": "SAME_SECTION",
        "itemCode": item_code,
        "companies": COMPANIES,
    }
    print(f"\n>>> 调用 {item_code} ...", flush=True)
    t0 = time.time()
    resp = requests.post(API, json=payload, timeout=300)
    dt = time.time() - t0
    print(f"<<< HTTP {resp.status_code} 耗时 {dt:.2f}s", flush=True)
    assert resp.status_code == 200, resp.text[:500]
    return resp.json()


def check_file_code():
    print("=" * 70, flush=True)
    print("[测试1] FILE_CODE_SIMILAR evidence 结构验证", flush=True)
    print("=" * 70, flush=True)
    data = run("FILE_CODE_SIMILAR")
    print(f"itemCode={data.get('itemCode')}, itemName={data.get('itemName')}", flush=True)

    failed_count = 0
    success_count = 0
    for r in data.get("results", []):
        cid = r["companyRecordId"]
        status = r["status"]
        ev = r.get("evidence", {})
        print(f"  公司 {cid}: status={status}", flush=True)
        print(f"    summary: {r.get('summary', '')}", flush=True)
        print(f"    evidence: {json.dumps(ev, ensure_ascii=False)}", flush=True)

        if status == "FAILED":
            failed_count += 1
            assert "fileId" in ev, f"FAILED 结果缺少 fileId: {ev}"
            assert "similarCompanyRecordIds" in ev, f"FAILED 结果缺少 similarCompanyRecordIds: {ev}"
            assert isinstance(ev["similarCompanyRecordIds"], list), "similarCompanyRecordIds 必须为列表"
            assert len(ev["fileId"]) > 0, "fileId 不能为空"
            print(f"    ✓ fileId + similarCompanyRecordIds 结构正确", flush=True)
        elif status == "SUCCESS":
            success_count += 1
            assert len(ev) == 0, f"SUCCESS 结果 evidence 必须为空: {ev}"
            print(f"    ✓ evidence 为空（符合 SUCCESS 规范）", flush=True)

    print(f"\n汇总: {failed_count} FAILED, {success_count} SUCCESS", flush=True)
    return data


def check_doc_author():
    print("\n" + "=" * 70, flush=True)
    print("[测试2] DOC_AUTHOR_SIMILAR evidence 结构验证", flush=True)
    print("=" * 70, flush=True)
    data = run("DOC_AUTHOR_SIMILAR")
    print(f"itemCode={data.get('itemCode')}, itemName={data.get('itemName')}", flush=True)

    failed_count = 0
    success_count = 0
    for r in data.get("results", []):
        cid = r["companyRecordId"]
        status = r["status"]
        ev = r.get("evidence", {})
        print(f"  公司 {cid}: status={status}", flush=True)
        print(f"    summary: {r.get('summary', '')}", flush=True)
        print(f"    evidence: {json.dumps(ev, ensure_ascii=False)}", flush=True)

        if status == "FAILED":
            failed_count += 1
            assert "author" in ev, f"FAILED 结果缺少 author: {ev}"
            assert "similarCompanyRecordIds" in ev, f"FAILED 结果缺少 similarCompanyRecordIds: {ev}"
            assert isinstance(ev["similarCompanyRecordIds"], list), "similarCompanyRecordIds 必须为列表"
            assert len(ev["author"]) > 0, "author 不能为空"
            print(f"    ✓ author + similarCompanyRecordIds 结构正确", flush=True)
        elif status == "SUCCESS":
            success_count += 1
            assert len(ev) == 0, f"SUCCESS 结果 evidence 必须为空: {ev}"
            print(f"    ✓ evidence 为空（符合 SUCCESS 规范）", flush=True)

    print(f"\n汇总: {failed_count} FAILED, {success_count} SUCCESS", flush=True)
    return data


def check_editor_signer():
    print("\n" + "=" * 70, flush=True)
    print("[测试3] EDITOR_SIGNER_SIMILAR evidence 结构验证", flush=True)
    print("=" * 70, flush=True)
    data = run("EDITOR_SIGNER_SIMILAR")
    print(f"itemCode={data.get('itemCode')}, itemName={data.get('itemName')}", flush=True)

    for r in data.get("results", []):
        cid = r["companyRecordId"]
        status = r["status"]
        ev = r.get("evidence", {})
        print(f"  公司 {cid}: status={status}", flush=True)
        print(f"    summary: {r.get('summary', '')}", flush=True)
        print(f"    evidence: {json.dumps(ev, ensure_ascii=False)}", flush=True)

        if status == "FAILED":
            assert "editor" in ev, f"FAILED 结果缺少 editor: {ev}"
            assert "similarCompanyRecordIds" in ev, f"FAILED 结果缺少 similarCompanyRecordIds: {ev}"
            print(f"    ✓ editor + similarCompanyRecordIds 结构正确", flush=True)
        elif status == "SUCCESS":
            assert len(ev) == 0, f"SUCCESS 结果 evidence 必须为空: {ev}"
            print(f"    ✓ evidence 为空", flush=True)

    return data


if __name__ == "__main__":
    print("=" * 70, flush=True)
    print("evidence 字段验证 — 轻量级端点（FILE_CODE / DOC_AUTHOR / EDITOR_SIGNER）", flush=True)
    print("=" * 70, flush=True)

    all_pass = True
    try:
        check_file_code()
    except Exception as e:
        all_pass = False
        print(f"✗ FILE_CODE_SIMILAR 测试失败: {e}", flush=True)

    try:
        check_doc_author()
    except Exception as e:
        all_pass = False
        print(f"✗ DOC_AUTHOR_SIMILAR 测试失败: {e}", flush=True)

    try:
        check_editor_signer()
    except Exception as e:
        all_pass = False
        print(f"✗ EDITOR_SIGNER_SIMILAR 测试失败: {e}", flush=True)

    print("\n" + "=" * 70, flush=True)
    if all_pass:
        print("✓ 所有轻量级 evidence 测试通过", flush=True)
    else:
        print("✗ 部分测试失败", flush=True)
    print("=" * 70, flush=True)

    sys.exit(0 if all_pass else 1)
