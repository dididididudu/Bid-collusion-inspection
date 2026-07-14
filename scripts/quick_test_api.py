"""
快速 API 测试脚本 — 零依赖，直接匹配本地 PDF 文件名，无需 HTTP 文件服务

原理:
  API 先检查 batch_downloads/{batchId}/{recordId}_{bidderName}.pdf 是否存在，
  存在则跳过下载，bidFileUrl 只是一个不会被用到的占位字段。
  因此只要 batchId + recordId + bidderName 与已有文件名匹配，无需任何外部服务。

用法:
    # 测试单个维度
    python scripts/quick_test_api.py --item FILE_CODE_SIMILAR

    # 测试所有轻量项
    python scripts/quick_test_api.py --lightweight

    # 测试所有维度（含重型）
    python scripts/quick_test_api.py --all

    # 自定义 PDF 目录
    python scripts/quick_test_api.py --pdf-dir batch_downloads/75689 --all
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

ITEM_NAMES = {
    "FILE_CODE_SIMILAR": "文件码雷同",
    "EDITOR_SIGNER_SIMILAR": "编辑经办人雷同",
    "DOC_AUTHOR_SIMILAR": "文档作者雷同",
    "SAME_BID_CONTACT_SIMILAR": "人名雷同",
    "SAME_bidderName_SIMILAR": "公司名雷同",
    "TECH_BID_SIMILAR": "技术标雷同",
    "Business_BID_SIMILAR": "商务标雷同",
}

LIGHTWEIGHT = [
    "FILE_CODE_SIMILAR", "DOC_AUTHOR_SIMILAR", "EDITOR_SIGNER_SIMILAR",
    "SAME_BID_CONTACT_SIMILAR", "SAME_bidderName_SIMILAR",
]
HEAVY = ["TECH_BID_SIMILAR", "Business_BID_SIMILAR"]


def parse_pdf_dir(pdf_dir: Path):
    """扫描 PDF 目录，从文件名解析 recordId 和 bidderName

    文件名格式: {recordId}_{bidderName}.pdf
    例如: 501_A公司.pdf → companyRecordId=501, bidderName="A公司"
    """
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if len(pdfs) < 2:
        raise SystemExit(f"PDF 目录至少需要 2 个文件: {pdf_dir}")

    companies = []
    for pdf in pdfs:
        stem = pdf.stem
        # 尝试从文件名解析: digits_name
        m = re.match(r'^(\d+)_(.+)$', stem)
        if m:
            record_id = int(m.group(1))
            bidder_name = m.group(2)
        else:
            # 无法解析就用 hash 生成 ID
            record_id = abs(hash(stem)) % 900 + 100
            bidder_name = stem

        companies.append({
            "companyRecordId": record_id,
            "registrationCompanyId": record_id + 100,
            "sectionId": 11,
            "bidderName": bidder_name,
            "bidFileUrl": f"http://localhost/placeholder.pdf",  # 不会被用到
        })

    return companies


def check_api_health(base_url: str) -> bool:
    """检查 API 服务是否在线"""
    try:
        resp = requests.get(f"{base_url}/api/v1/collusive-check/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def run_test(base_url: str, item_code: str, companies: list, batch_id: int, timeout: int):
    """执行单个 itemCode 检测"""
    item_name = ITEM_NAMES.get(item_code, item_code)
    payload = {
        "batchId": batch_id,
        "projectId": 10001,
        "checkMode": "SAME_SECTION",
        "itemCode": item_code,
        "companies": companies,
    }

    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"[{item_code}] {item_name}")
    print(f"[{item_code}] 发起请求... (timeout={timeout}s)")
    sys.stdout.flush()

    try:
        resp = requests.post(
            f"{base_url}/api/v1/collusive-check/items/analyze",
            json=payload,
            timeout=timeout,
        )
        elapsed = time.time() - t0
        data = resp.json()

        results = data.get("results", [])
        failed = sum(1 for r in results if r["status"] == "FAILED")
        success = sum(1 for r in results if r["status"] == "SUCCESS")
        error = sum(1 for r in results if r["status"] == "ERROR")

        print(f"[{item_code}] 耗时 {elapsed:.1f}s | "
              f"FAILED={failed} SUCCESS={success} ERROR={error}")

        # 输出每个公司的结果
        for r in results:
            cid = r["companyRecordId"]
            st = r["status"]
            summary = r["summary"]
            ev = r.get("evidence", {})
            # 精简 evidence 输出
            ev_brief = {}
            if "fileId" in ev:
                ev_brief["fileId"] = ev["fileId"][:16] + "..."
            if "similarCompanyRecordIds" in ev:
                ev_brief["similarIds"] = ev["similarCompanyRecordIds"]
            if "commonMobiles" in ev:
                ev_brief["commonMobiles"] = len(ev["commonMobiles"])
            if "commonPersons" in ev:
                ev_brief["commonPersons"] = len(ev["commonPersons"])
            if "foundCompanyNames" in ev:
                ev_brief["foundCompanyNames"] = ev["foundCompanyNames"]
            print(f"  [{st:7s}] company={cid}  {summary}")
            if ev_brief:
                print(f"          {json.dumps(ev_brief, ensure_ascii=False)}")

        return {"itemCode": item_code, "elapsed": elapsed,
                "failed": failed, "success": success, "error": error}

    except requests.exceptions.Timeout:
        elapsed = time.time() - t0
        print(f"[{item_code}] 超时 ({elapsed:.0f}s > {timeout}s)")
        return {"itemCode": item_code, "elapsed": elapsed,
                "failed": 0, "success": 0, "error": 4, "timeout": True}
    except Exception as e:
        print(f"[{item_code}] 异常: {e}")
        return {"itemCode": item_code, "elapsed": time.time() - t0,
                "failed": 0, "success": 0, "error": 4}


def main():
    parser = argparse.ArgumentParser(description="快速 API 测试 — 零依赖本地模式")
    parser.add_argument("--api", default="http://127.0.0.1:8001")
    parser.add_argument("--pdf-dir", default="batch_downloads/75689")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="单个 itemCode 超时秒数 (默认 1800，即 30 分钟)")
    parser.add_argument("--item", help="测试单个 itemCode")
    parser.add_argument("--lightweight", action="store_true",
                        help="测试全部 5 个轻量项")
    parser.add_argument("--heavy", action="store_true",
                        help="测试全部 2 个重型项")
    parser.add_argument("--all", action="store_true",
                        help="测试全部 7 个维度")
    args = parser.parse_args()

    if not args.item and not args.lightweight and not args.heavy and not args.all:
        parser.print_help()
        print("\n示例:")
        print("  python scripts/quick_test_api.py --item FILE_CODE_SIMILAR")
        print("  python scripts/quick_test_api.py --lightweight")
        print("  python scripts/quick_test_api.py --all")
        return

    base_url = args.api.rstrip("/")

    # 1. 检查 API
    if not check_api_health(base_url):
        print(f"[ERROR] API 服务未启动: {base_url}")
        print(f"  请先运行: python collusive_check_api.py")
        return

    # 2. 扫描 PDF 目录，自动解析文件名 → companyRecordId + bidderName
    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        print(f"[ERROR] PDF 目录不存在: {pdf_dir}")
        return

    companies = parse_pdf_dir(pdf_dir)
    # 自动从目录名提取 batchId
    batch_id = int(pdf_dir.name) if pdf_dir.name.isdigit() else 75689

    print(f"PDF 目录: {pdf_dir}")
    print(f"batchId: {batch_id}")
    print(f"解析到 {len(companies)} 家公司:")
    for c in companies:
        print(f"  - companyRecordId={c['companyRecordId']}  bidderName={c['bidderName']}")

    # 3. 确定要测试的 itemCode 列表
    if args.all:
        items = LIGHTWEIGHT + HEAVY
    elif args.heavy:
        items = HEAVY
    elif args.lightweight:
        items = LIGHTWEIGHT
    else:
        items = [args.item]

    print(f"\n待测试: {len(items)} 个维度")
    print(f"API: {base_url}")
    print("模式: 本地文件（无需 HTTP 文件服务）")

    # 4. 按顺序执行
    summary = []
    for item_code in items:
        result = run_test(base_url, item_code, companies, batch_id, args.timeout)
        summary.append(result)

    # 6. 汇总
    print(f"\n{'='*60}")
    print("汇总:")
    print(f"{'itemCode':35s} {'耗时':>8s}  FAILED  SUCCESS  ERROR")
    print("-" * 75)
    for s in summary:
        flags = " ⚠超时" if s.get("timeout") else ""
        print(f"{s['itemCode']:35s} {s['elapsed']:>6.1f}s  "
              f"{s['failed']:>6d}  {s['success']:>7d}  {s['error']:>5d}{flags}")

    total_time = sum(s["elapsed"] for s in summary)
    print(f"\n总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    main()
