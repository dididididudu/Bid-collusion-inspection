"""Local performance smoke test for collusive-check API.

Usage:
    python scripts/perf_test_collusive_api.py --pdf-dir batch_downloads/75689 --api http://127.0.0.1:8001

The API expects downloadable URLs, so this script serves local PDFs over a
temporary HTTP server and posts a normal analyze request.
"""

import argparse
import functools
import json
import os
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests


def _start_file_server(pdf_dir: Path, host: str, port: int):
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(pdf_dir))
    server = ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _build_companies(pdf_dir: Path, file_base_url: str):
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if len(pdfs) < 2:
        raise SystemExit(f"Need at least 2 PDFs in {pdf_dir}")

    companies = []
    for idx, pdf in enumerate(pdfs, start=1):
        record_id = 1000 + idx
        companies.append({
            "companyRecordId": record_id,
            "registrationCompanyId": 2000 + idx,
            "sectionId": 1,
            "bidderName": pdf.stem[:40] or f"company_{idx}",
            "bidFileUrl": f"{file_base_url}/{requests.utils.quote(pdf.name)}",
        })
    return companies


def _post_item(api: str, batch_id: int, item_code: str, companies: list):
    payload = {
        "batchId": batch_id,
        "projectId": 999001,
        "checkMode": "SAME_SECTION",
        "itemCode": item_code,
        "companies": companies,
    }
    started = time.perf_counter()
    resp = requests.post(
        f"{api.rstrip('/')}/api/v1/collusive-check/items/analyze",
        json=payload,
        timeout=3600,
    )
    elapsed = time.perf_counter() - started
    print(f"\n[{item_code}] status={resp.status_code} elapsed={elapsed:.2f}s")
    try:
        data = resp.json()
        failed = sum(1 for r in data.get("results", []) if r.get("status") == "FAILED")
        errors = sum(1 for r in data.get("results", []) if r.get("status") == "ERROR")
        print(f"results={len(data.get('results', []))}, failed={failed}, errors={errors}")
        print(json.dumps(data, ensure_ascii=False, indent=2)[:4000])
    except Exception:
        print(resp.text[:4000])
    resp.raise_for_status()
    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", required=True, help="Directory containing local PDF files")
    parser.add_argument("--api", default="http://127.0.0.1:8001", help="Collusive-check API base URL")
    parser.add_argument("--host", default="127.0.0.1", help="Temporary file server host")
    parser.add_argument("--port", type=int, default=18081, help="Temporary file server port")
    parser.add_argument("--batch-id", type=int, default=int(time.time()), help="Batch id for API cache reuse")
    parser.add_argument(
        "--heavy",
        action="store_true",
        help="Test heavy text+image/OCR items: TECH_BID_SIMILAR and BID_COMPANY_NAME_ABNORMAL",
    )
    parser.add_argument(
        "--items",
        nargs="+",
        default=None,
        help="Item codes to test in order",
    )
    args = parser.parse_args()
    if args.items is None:
        args.items = (
            ["TECH_BID_SIMILAR", "BID_COMPANY_NAME_ABNORMAL"]
            if args.heavy
            else ["FILE_CODE_SIMILAR", "DOC_AUTHOR_SIMILAR", "EDITOR_SIGNER_SIMILAR"]
        )

    pdf_dir = Path(args.pdf_dir).resolve()
    if not pdf_dir.is_dir():
        raise SystemExit(f"PDF directory does not exist: {pdf_dir}")

    server = _start_file_server(pdf_dir, args.host, args.port)
    file_base_url = f"http://{args.host}:{args.port}"
    companies = _build_companies(pdf_dir, file_base_url)

    print(f"Serving {pdf_dir} at {file_base_url}")
    print(f"API: {args.api}")
    print(f"batchId: {args.batch_id}")
    print(f"companies: {len(companies)}")
    print("Tip: use --heavy to test text+image/OCR heavy items.")

    timings = {}
    try:
        for item in args.items:
            timings[item] = _post_item(args.api, args.batch_id, item, companies)
    finally:
        server.shutdown()

    print("\nTiming summary:")
    for item, elapsed in timings.items():
        print(f"  {item}: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
