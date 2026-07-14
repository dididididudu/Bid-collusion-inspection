"""
一键测试脚本 — 单命令完成全部测试，无需手动启动文件服务、生成 JSON、拼 curl。

用法:
    python scripts/test.py                          # 测试全部 7 项
    python scripts/test.py --lightweight            # 仅 5 个轻量项
    python scripts/test.py --pdf-dir batch_downloads/75689
    python scripts/test.py --item TECH_BID_SIMILAR  # 单项
"""

import argparse
import json
import re
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

import requests

ITEMS = [
    ("FILE_CODE_SIMILAR",        "文件码雷同"),
    ("DOC_AUTHOR_SIMILAR",       "文档作者雷同"),
    ("EDITOR_SIGNER_SIMILAR",    "编辑经办人雷同"),
    ("SAME_BID_CONTACT_SIMILAR", "人名雷同"),
    ("SAME_bidderName_SIMILAR",  "公司名雷同"),
    ("TECH_BID_SIMILAR",         "技术标雷同"),
    ("Business_BID_SIMILAR",     "商务标雷同"),
]

LIGHTWEIGHT = [code for code, _ in ITEMS[:5]]
HEAVY = [code for code, _ in ITEMS[5:]]


def _start_file_server(directory: Path, host: str, port: int):
    """后台启动 HTTP 文件服务"""
    handler = lambda *args: SimpleHTTPRequestHandler(*args, directory=str(directory))
    server = ThreadingHTTPServer((host, port), handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


def _parse_pdfs(pdf_dir: Path, file_host: str, file_port: int):
    """扫描 PDF 目录，从文件名解析公司信息（直接使用原始文件，不创建副本）"""
    pdfs = sorted(p for p in pdf_dir.glob("*.pdf") if not p.stem.endswith("_test"))
    if len(pdfs) < 2:
        raise SystemExit(f"PDF 不足（至少 2 个），当前 {len(pdfs)} 个: {pdf_dir}")

    companies = []
    for pdf in pdfs:
        m = re.match(r'^(\d+)_(.+)$', pdf.stem)
        if not m:
            print(f"  ⚠ 跳过: {pdf.name}")
            continue
        record_id = int(m.group(1))
        bidder_name = m.group(2)
        companies.append({
            "companyRecordId": record_id,
            "registrationCompanyId": record_id + 100,
            "sectionId": 11,
            "bidderName": bidder_name,
            "bidFileUrl": f"http://{file_host}:{file_port}/{quote(pdf.name)}",
        })
    return companies


def main():
    parser = argparse.ArgumentParser(description="围标串标 AI 服务一键测试")
    parser.add_argument("--pdf-dir", default="batch_downloads/75689")
    parser.add_argument("--api", default="http://127.0.0.1:8001")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--item", help="测试单个 itemCode")
    parser.add_argument("--lightweight", action="store_true", help="仅轻量项")
    parser.add_argument("--heavy", action="store_true", help="仅重型项")
    parser.add_argument("--all", action="store_true", help="全部 7 项（默认）")
    parser.add_argument("--no-response", action="store_true", help="不输出响应 JSON")
    args = parser.parse_args()

    # 确定测试范围
    if args.item:
        items = [(args.item, {c: n for c, n in ITEMS}.get(args.item, args.item))]
    elif args.lightweight:
        items = [(c, n) for c, n in ITEMS[:5]]
    elif args.heavy:
        items = [(c, n) for c, n in ITEMS[5:]]
    else:
        items = list(ITEMS)

    pdf_dir = Path(args.pdf_dir).resolve()
    if not pdf_dir.is_dir():
        raise SystemExit(f"目录不存在: {pdf_dir}")
    batch_id = int(pdf_dir.name) if pdf_dir.name.isdigit() else 75689

    # 1. 解析 PDF → companies
    companies = _parse_pdfs(pdf_dir, args.host, args.port)
    print(f"batchId={batch_id} 公司数={len(companies)}")
    for c in companies:
        print(f"  [{c['companyRecordId']}] {c['bidderName']}")

    # 2. 启动文件服务
    server = _start_file_server(pdf_dir, args.host, args.port)
    print(f"文件服务: http://{args.host}:{args.port}")

    # 3. 按顺序测试
    results = []
    for item_code, item_name in items:
        payload = {
            "batchId": batch_id, "projectId": 10001,
            "checkMode": "SAME_SECTION", "itemCode": item_code,
            "companies": companies,
        }

        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{args.api.rstrip('/')}/api/v1/collusive-check/items/analyze",
                json=payload, timeout=args.timeout,
            )
            elapsed = time.perf_counter() - t0
            data = resp.json()
            rlist = data.get("results", [])
            failed = sum(1 for r in rlist if r["status"] == "FAILED")
            success = sum(1 for r in rlist if r["status"] == "SUCCESS")
            errors = sum(1 for r in rlist if r["status"] == "ERROR")
            print(f"  [{item_code}] {elapsed:.1f}s  FAILED={failed} SUCCESS={success} ERROR={errors}")
            if not args.no_response:
                print(json.dumps(data, ensure_ascii=False, indent=2))
            results.append({"code": item_code, "elapsed": elapsed, "failed": failed, "success": success, "error": errors})
        except requests.exceptions.Timeout:
            elapsed = time.perf_counter() - t0
            print(f"  [{item_code}] 超时 ({elapsed:.0f}s)")
            results.append({"code": item_code, "elapsed": elapsed, "failed": 0, "success": 0, "error": len(companies), "timeout": True})
        except requests.exceptions.ConnectionError:
            server.shutdown()
            raise SystemExit(f"API 未启动: {args.api}\n请先执行: python collusive_check_api.py")

    server.shutdown()

    # 4. 汇总
    print(f"\n{'itemCode':35s} {'耗时':>8s}  FAILED SUCCESS ERROR")
    print("-" * 65)
    for r in results:
        flag = " ⚠" if r.get("timeout") else ""
        print(f"{r['code']:35s} {r['elapsed']:>6.1f}s  {r['failed']:>6d} {r['success']:>7d} {r['error']:>5d}{flag}")
    total = sum(r["elapsed"] for r in results)
    print(f"\n总耗时: {total:.1f}s ({total/60:.1f}min)")



if __name__ == "__main__":
    main()
