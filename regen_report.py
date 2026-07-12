#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重跑 TECH_BID_SIMILAR — 复用已下载的 PDF (batch_downloads/75689/)
报告将保留 12 小时（API 清理已改为 43200 秒）
"""
import sys
import json
import time
import requests

sys.stdout.reconfigure(line_buffering=True)

API = "http://127.0.0.1:8001/api/v1/collusive-check/items/analyze"

# 使用 batch_id=75689 复用已下载的 PDF
# batch_downloads/75689/701_测试公司A.pdf (投标文件-雀翼0828.pdf, 469页)
# batch_downloads/75689/702_测试公司B.pdf (投标文件.pdf, 615页)
COMPANIES = [
    {
        "companyRecordId": 701,
        "registrationCompanyId": 301,
        "sectionId": 11,
        "bidderName": "测试公司A",
        "bidFileUrl": "http://127.0.0.1:18080/placeholder_a.pdf",
    },
    {
        "companyRecordId": 702,
        "registrationCompanyId": 302,
        "sectionId": 11,
        "bidderName": "测试公司B",
        "bidFileUrl": "http://127.0.0.1:18080/placeholder_b.pdf",
    },
]


def main():
    print("=" * 70, flush=True)
    print("重跑 TECH_BID_SIMILAR (batch_id=75689, 复用已下载 PDF)", flush=True)
    print("=" * 70, flush=True)

    # 健康检查
    print("\n>>> 健康检查...", flush=True)
    for attempt in range(30):
        try:
            resp = requests.get("http://127.0.0.1:8001/api/v1/collusive-check/health", timeout=5)
            if resp.status_code == 200:
                print(f"<<< API 就绪: {resp.json()}", flush=True)
                break
        except Exception:
            time.sleep(3)
    else:
        print("✗ API 服务未就绪", flush=True)
        sys.exit(1)

    # 发送请求
    payload = {
        "batchId": 75689,
        "projectId": 1,
        "checkMode": "SAME_SECTION",
        "itemCode": "TECH_BID_SIMILAR",
        "companies": COMPANIES,
    }

    print(f"\n>>> 发送 TECH_BID_SIMILAR 请求 (batch=75689)...", flush=True)
    print(f"    公司: 701(测试公司A), 702(测试公司B)", flush=True)
    print(f"    PDF已缓存: batch_downloads/75689/", flush=True)
    print(f"    预计耗时: 30-35 分钟 (OCR + SBERT + 匹配)", flush=True)

    t0 = time.time()
    try:
        resp = requests.post(API, json=payload, timeout=3600)
        dt = time.time() - t0
        print(f"\n<<< HTTP {resp.status_code} 耗时 {dt:.1f}s ({dt/60:.1f}分钟)", flush=True)

        if resp.status_code != 200:
            print(f"错误响应: {resp.text[:500]}", flush=True)
            sys.exit(1)

        data = resp.json()
        print(f"\nitemCode: {data.get('itemCode')}", flush=True)
        print(f"itemName: {data.get('itemName')}", flush=True)

        # 保存完整响应
        response_path = "tech_bid_response_75689.json"
        with open(response_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n完整响应已保存: {response_path}", flush=True)

        # 打印结果摘要
        print(f"\n{'='*70}", flush=True)
        print("公司级结果摘要", flush=True)
        print(f"{'='*70}", flush=True)
        for r in data.get("results", []):
            cid = r["companyRecordId"]
            status = r["status"]
            summary = r.get("summary", "")
            ev = r.get("evidence", {})
            print(f"\n公司 {cid}: {status}", flush=True)
            print(f"  summary: {summary}", flush=True)

            if status == "FAILED":
                print(f"  similarCompanyRecordIds: {ev.get('similarCompanyRecordIds', [])}", flush=True)
                sp = ev.get("similarParagraphs", [])
                print(f"  similarParagraphs: {len(sp)} 条", flush=True)
                for p in sp[:3]:
                    print(f"    -> 公司 {p.get('companyRecordId')}, sim={p.get('similarity',0):.4f}, matches={len(p.get('paragraphMatches',[]))}", flush=True)
                si = ev.get("similarImages", [])
                print(f"  similarImages: {len(si)} 条", flush=True)
                for im in si[:3]:
                    print(f"    -> 公司 {im.get('companyRecordId')}, count={im.get('imageMatchCount',0)}, pairs={len(im.get('similarImages',[]))}", flush=True)
            else:
                print(f"  evidence: {{}}", flush=True)

        print(f"\n{'='*70}", flush=True)
        print(f"总耗时: {dt:.1f}s ({dt/60:.1f}分钟)", flush=True)
        print(f"{'='*70}", flush=True)

    except requests.exceptions.Timeout:
        dt = time.time() - t0
        print(f"\n✗ 请求超时 ({dt:.1f}s)", flush=True)
        sys.exit(1)
    except Exception as e:
        dt = time.time() - t0
        print(f"\n✗ 请求失败 ({dt:.1f}s): {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
