"""
从 PDF 目录自动生成 JSON 请求文件和 curl 命令，简化 Java 后端接口测试。

用法:
    python scripts/gen_test_payloads.py batch_downloads/75689
    python scripts/gen_test_payloads.py batch_downloads/75689 --port 18081 --host 127.0.0.1
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

# Git Bash / Windows 终端 UTF-8 输出
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ITEMS = [
    ("FILE_CODE_SIMILAR",        "文件码雷同（轻量）"),
    ("DOC_AUTHOR_SIMILAR",       "文档作者雷同（轻量）"),
    ("EDITOR_SIGNER_SIMILAR",    "编辑经办人雷同（轻量）"),
    ("SAME_BID_CONTACT_SIMILAR", "人名雷同（轻量）"),
    ("SAME_bidderName_SIMILAR",  "公司名雷同（轻量）"),
    ("TECH_BID_SIMILAR",         "技术标雷同（重型，含 OCR+SBERT）"),
    ("Business_BID_SIMILAR",     "商务标雷同（重型，含 OCR+SBERT）"),
]


def main():
    parser = argparse.ArgumentParser(description="生成 API 测试用 JSON 请求文件")
    parser.add_argument("pdf_dir", help="PDF 文件目录，如 batch_downloads/75689")
    parser.add_argument("--port", type=int, default=18081, help="HTTP 文件服务端口 (默认 18081)")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP 文件服务地址 (默认 127.0.0.1)")
    parser.add_argument("--api", default="http://127.0.0.1:8001", help="API 地址")
    parser.add_argument("--check-mode", default="SAME_SECTION", help="checkMode")
    parser.add_argument("--project-id", type=int, default=10001, help="projectId")
    parser.add_argument("--no-ascii", action="store_true", help="不创建 ASCII 副本（Linux/Mac）")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir).resolve()
    if not pdf_dir.is_dir():
        raise SystemExit(f"目录不存在: {pdf_dir}")

    # 自动从目录名提取 batchId
    batch_id = int(pdf_dir.name) if pdf_dir.name.isdigit() else None
    if batch_id is None:
        raise SystemExit(f"目录名必须是纯数字 (batchId)，当前: {pdf_dir.name}")

    # 扫描 PDF 文件，解析 {recordId}_{bidderName}.pdf
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if len(pdfs) < 2:
        raise SystemExit(f"至少需要 2 个 PDF 文件，当前: {len(pdfs)} 个")

    # 排除已有的 ASCII 副本（*_test.pdf）
    original_pdfs = [p for p in pdfs if not p.stem.endswith("_test")]
    if not original_pdfs:
        original_pdfs = pdfs  # 如果全是 _test 后缀，就全用

    companies = []
    ascii_copies = []  # (src, dst) for Windows ASCII copies

    for pdf in original_pdfs:
        stem = pdf.stem
        # 排除 _test 后缀的副本
        if stem.endswith("_test"):
            continue
        m = re.match(r'^(\d+)_(.+)$', stem)
        if not m:
            print(f"  ⚠ 跳过（无法解析）: {pdf.name}")
            continue
        record_id = int(m.group(1))
        bidder_name = m.group(2)

        # Windows: 创建 ASCII 副本; Linux/Mac: 直接用原名
        if args.no_ascii or sys.platform != "win32":
            url_name = pdf.name
        else:
            ascii_name = f"{record_id}_test.pdf"
            ascii_path = pdf_dir / ascii_name
            url_name = ascii_name
            if not ascii_path.exists():
                ascii_copies.append((pdf, ascii_path))

        companies.append({
            "companyRecordId": record_id,
            "registrationCompanyId": record_id + 100,
            "sectionId": 11,
            "bidderName": bidder_name,
            "bidFileUrl": f"http://{args.host}:{args.port}/{url_name}",
        })

    if len(companies) < 2:
        raise SystemExit(f"至少需要 2 个可解析的 PDF，当前: {len(companies)} 个")

    # 创建 ASCII 副本
    if ascii_copies:
        print(f"\n{'='*50}")
        print(f"Windows: 创建 {len(ascii_copies)} 个 ASCII 文件副本 ...")
        for src, dst in ascii_copies:
            shutil.copy2(src, dst)
            print(f"  ✓ {dst.name}")

    # 生成 JSON 请求文件
    os.makedirs(pdf_dir, exist_ok=True)
    for item_code, item_name in ITEMS:
        payload = {
            "batchId": batch_id,
            "projectId": args.project_id,
            "checkMode": args.check_mode,
            "itemCode": item_code,
            "companies": companies,
        }
        path = pdf_dir / f"req_{item_code}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    # ── 输出摘要 ──
    file_base = f"http://{args.host}:{args.port}"
    print(f"\n{'='*50}")
    print(f"batchId: {batch_id}")
    print(f"公司数: {len(companies)}")
    for c in companies:
        print(f"  - [{c['companyRecordId']}] {c['bidderName']}")
    print(f"\nJSON 请求文件 → {pdf_dir}/req_*.json ({len(ITEMS)} 个)")

    # ── 打印操作步骤 ──
    print(f"\n{'='*50}")
    print("接下来执行:")
    print(f"\n  [1] 启动文件服务（新终端）:")
    print(f"      cd {pdf_dir}")
    print(f"      python -m http.server {args.port}")
    print(f"\n  [2] 运行测试:")

    # 轻量项
    for item_code, item_name in ITEMS[:5]:
        print(f"  # {item_name}")
        print(f"  curl -s -X POST \"{args.api}/api/v1/collusive-check/items/analyze\" \\")
        print(f"    -H \"Content-Type: application/json\" \\")
        print(f"    -d @{pdf_dir / f'req_{item_code}.json'} | python -m json.tool")
        print()

    # 重型项
    for item_code, item_name in ITEMS[5:]:
        print(f"  # {item_name}")
        print(f"  curl -s -X POST \"{args.api}/api/v1/collusive-check/items/analyze\" \\")
        print(f"    -H \"Content-Type: application/json\" \\")
        print(f"    -d @{pdf_dir / f'req_{item_code}.json'} \\")
        print(f"    --max-time 1800 | python -m json.tool")
        print()


if __name__ == "__main__":
    main()
