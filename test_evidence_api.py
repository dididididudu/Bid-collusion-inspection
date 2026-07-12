#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
围标串标检查 API — evidence 字段验证测试

验证各检测维度的 evidence 是否包含详细信息：
1. 文件码/文档作者：展示值 + 公司ID
2. 文本相似：相似段落内容 + 公司ID
3. 图片相似：图片引用 + 公司ID

用法:
    python test_evidence_api.py                          # 运行所有测试
    python test_evidence_api.py --host 127.0.0.1 --port 8001
"""

import os
import sys
import json
import time
import http.server
import socketserver
import threading
from typing import Optional, List, Dict, Any

try:
    import requests
except ImportError:
    print("缺少 requests 库，请运行: pip install requests")
    sys.exit(1)


# ============================================================
# 测试配置
# ============================================================

class TestConfig:
    def __init__(self):
        self.host = os.environ.get("BID_TEST_HOST", "127.0.0.1")
        self.port = int(os.environ.get("BID_TEST_PORT", "8001"))
        self.base_url = f"http://{self.host}:{self.port}"
        self.data_dir = os.environ.get(
            "BID_TEST_DATA_DIR",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_input"),
        )
        self.timeout = 1800
        self.file_server_port = int(os.environ.get("BID_FILE_SERVER_PORT", "18080"))


# ============================================================
# 本地文件服务器
# ============================================================

class FileServer:
    def __init__(self, data_dir: str, port: int):
        self.data_dir = data_dir
        self.port = port
        self.httpd = None
        self.thread = None

    def start(self):
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"测试数据目录不存在: {self.data_dir}")
        os.chdir(self.data_dir)
        handler = http.server.SimpleHTTPRequestHandler
        self.httpd = socketserver.TCPServer(("127.0.0.1", self.port), handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        print(f"[FileServer] 启动: http://127.0.0.1:{self.port}/")

    def stop(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()

    def get_url(self, filename: str) -> str:
        return f"http://127.0.0.1:{self.port}/{filename}"


# ============================================================
# 测试用例
# ============================================================

class EvidenceTestSuite:
    """验证各维度 evidence 字段结构"""

    def __init__(self, config: TestConfig, file_server: FileServer):
        self.config = config
        self.file_server = file_server
        self.results = []

    def _get_companies(self) -> List[Dict]:
        """获取测试公司列表"""
        pdfs = [f for f in os.listdir(self.file_server.data_dir) if f.lower().endswith('.pdf')]
        pdfs.sort()
        companies = []
        for i, pdf_name in enumerate(pdfs[:3], start=501):
            companies.append({
                "companyRecordId": i,
                "registrationCompanyId": 100 + i,
                "sectionId": 11,
                "bidderName": f"测试公司{i}",
                "bidFileUrl": self.file_server.get_url(pdf_name),
            })
        return companies

    def _post_analyze(self, item_code: str, companies: List[Dict]) -> Dict:
        """发送 POST /analyze 请求"""
        resp = requests.post(
            f"{self.config.base_url}/api/v1/collusive-check/items/analyze",
            json={
                "batchId": int(time.time()) % 100000,
                "projectId": 1,
                "checkMode": "SAME_SECTION",
                "itemCode": item_code,
                "companies": companies,
            },
            timeout=self.config.timeout,
        )
        assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:300]}"
        return resp.json()

    def _log(self, passed: bool, test_name: str, detail: str = ""):
        status = "✓ PASS" if passed else "✗ FAIL"
        self.results.append((test_name, passed, detail))
        print(f"  {status} {test_name}")
        if not passed and detail:
            print(f"         {detail[:200]}")

    # ── 测试1: 文件码雷同 evidence ──
    def test_file_code_evidence(self):
        print("\n[测试1] FILE_CODE_SIMILAR evidence 结构验证")
        companies = self._get_companies()
        try:
            data = self._post_analyze("FILE_CODE_SIMILAR", companies)
        except Exception as e:
            self._log(False, "file_code_request", str(e))
            return

        self._log(True, "file_code_request")

        for r in data.get("results", []):
            if r["status"] == "FAILED":
                ev = r.get("evidence", {})
                has_file_id = "fileId" in ev
                has_similar = "similarCompanyRecordIds" in ev
                self._log(has_file_id, f"file_code_has_fileId (company={r['companyRecordId']})",
                          f"evidence: {ev}")
                self._log(has_similar, f"file_code_has_similarIds (company={r['companyRecordId']})",
                          f"evidence: {ev}")
                if has_similar:
                    self._log(
                        len(ev["similarCompanyRecordIds"]) > 0,
                        f"file_code_similarIds_nonempty (company={r['companyRecordId']})",
                    )
                break
        else:
            self._log(True, "file_code_no_failures (所有公司通过)")

    # ── 测试2: 文档作者雷同 evidence ──
    def test_doc_author_evidence(self):
        print("\n[测试2] DOC_AUTHOR_SIMILAR evidence 结构验证")
        companies = self._get_companies()
        try:
            data = self._post_analyze("DOC_AUTHOR_SIMILAR", companies)
        except Exception as e:
            self._log(False, "author_request", str(e))
            return

        self._log(True, "author_request")

        for r in data.get("results", []):
            if r["status"] == "FAILED":
                ev = r.get("evidence", {})
                has_author = "author" in ev
                has_similar = "similarCompanyRecordIds" in ev
                self._log(has_author, f"author_has_author (company={r['companyRecordId']})",
                          f"evidence: {ev}")
                self._log(has_similar, f"author_has_similarIds (company={r['companyRecordId']})",
                          f"evidence: {ev}")
                if has_author:
                    self._log(
                        len(ev["author"]) > 0,
                        f"author_name_nonempty (company={r['companyRecordId']})",
                    )
                break
        else:
            self._log(True, "author_no_failures (所有公司通过)")

    # ── 测试3: 技术标雷同 evidence（含段落内容）──
    def test_tech_bid_evidence(self):
        print("\n[测试3] TECH_BID_SIMILAR evidence 结构验证（含段落内容+图片引用）")
        companies = self._get_companies()
        if len(companies) < 2:
            self._log(False, "tech_bid_skip", "需要至少2家公司")
            return

        try:
            data = self._post_analyze("TECH_BID_SIMILAR", companies)
        except Exception as e:
            self._log(False, "tech_bid_request", str(e))
            return

        self._log(True, "tech_bid_request")

        for r in data.get("results", []):
            if r["status"] == "FAILED":
                ev = r.get("evidence", {})
                has_similar = "similarCompanyRecordIds" in ev
                has_paragraphs = "similarParagraphs" in ev
                has_images = "similarImages" in ev

                self._log(has_similar, f"tech_has_similarIds (company={r['companyRecordId']})",
                          f"evidence keys: {list(ev.keys())}")
                self._log(has_paragraphs, f"tech_has_similarParagraphs (company={r['companyRecordId']})",
                          f"evidence keys: {list(ev.keys())}")
                self._log(has_images, f"tech_has_similarImages (company={r['companyRecordId']})",
                          f"evidence keys: {list(ev.keys())}")

                # 验证段落匹配结构
                if has_paragraphs and ev["similarParagraphs"]:
                    sp = ev["similarParagraphs"][0]
                    has_cid = "companyRecordId" in sp
                    has_sim = "similarity" in sp
                    has_matches = "paragraphMatches" in sp
                    self._log(has_cid, f"tech_para_has_companyRecordId (company={r['companyRecordId']})")
                    self._log(has_sim, f"tech_para_has_similarity (company={r['companyRecordId']})")
                    self._log(has_matches, f"tech_para_has_matches (company={r['companyRecordId']})")

                    if has_matches and sp["paragraphMatches"]:
                        pm = sp["paragraphMatches"][0]
                        has_pa = "paragraph_a" in pm
                        has_pb = "paragraph_b" in pm
                        has_psim = "similarity" in pm
                        self._log(has_pa, f"tech_pm_has_paragraph_a (company={r['companyRecordId']})")
                        self._log(has_pb, f"tech_pm_has_paragraph_b (company={r['companyRecordId']})")
                        self._log(has_psim, f"tech_pm_has_similarity (company={r['companyRecordId']})")

                        # 验证段落内容非空
                        if has_pa and has_pb:
                            self._log(
                                len(pm["paragraph_a"]) > 0 and len(pm["paragraph_b"]) > 0,
                                f"tech_pm_content_nonempty (company={r['companyRecordId']})",
                            )

                # 验证图片匹配结构
                if has_images and ev["similarImages"]:
                    si = ev["similarImages"][0]
                    has_cid = "companyRecordId" in si
                    has_count = "imageMatchCount" in si
                    has_imgs = "similarImages" in si
                    self._log(has_cid, f"tech_img_has_companyRecordId (company={r['companyRecordId']})")
                    self._log(has_count, f"tech_img_has_count (company={r['companyRecordId']})")
                    self._log(has_imgs, f"tech_img_has_similarImages (company={r['companyRecordId']})")

                    if has_imgs and si["similarImages"]:
                        img = si["similarImages"][0]
                        has_sa = "source_a" in img
                        has_sb = "source_b" in img
                        has_conf = "confidence" in img
                        self._log(has_sa, f"tech_imgpair_has_source_a (company={r['companyRecordId']})")
                        self._log(has_sb, f"tech_imgpair_has_source_b (company={r['companyRecordId']})")
                        self._log(has_conf, f"tech_imgpair_has_confidence (company={r['companyRecordId']})")

                break
        else:
            self._log(True, "tech_bid_no_failures (所有公司通过)")

    # ── 测试4: SUCCESS 结果 evidence 为空 ──
    def test_success_empty_evidence(self):
        print("\n[测试4] SUCCESS 结果 evidence 为空验证")
        companies = self._get_companies()
        try:
            data = self._post_analyze("FILE_CODE_SIMILAR", companies)
        except Exception as e:
            self._log(False, "success_request", str(e))
            return

        for r in data.get("results", []):
            if r["status"] == "SUCCESS":
                ev = r.get("evidence", {})
                self._log(
                    len(ev) == 0,
                    f"success_empty_evidence (company={r['companyRecordId']})",
                    f"evidence: {ev}",
                )
                break
        else:
            self._log(True, "success_no_success_results (所有公司异常)")

    # ── 运行所有测试 ──
    def run_all(self):
        print("=" * 70)
        print("evidence 字段验证测试套件")
        print("=" * 70)

        self.test_file_code_evidence()
        self.test_doc_author_evidence()
        self.test_tech_bid_evidence()
        self.test_success_empty_evidence()

        # 汇总
        total = len(self.results)
        passed = sum(1 for _, p, _ in self.results if p)
        failed = total - passed

        print("\n" + "=" * 70)
        print(f"测试结果汇总: {passed}/{total} 通过, {failed} 失败")
        print("=" * 70)

        for name, p, detail in self.results:
            status = "✓" if p else "✗"
            print(f"  {status} {name}")
            if not p and detail:
                print(f"      {detail[:150]}")

        print("=" * 70)
        return failed == 0


# ============================================================
# 主程序
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="evidence 字段验证测试")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--file-port", type=int, default=18080)
    args = parser.parse_args()

    config = TestConfig()
    config.host = args.host
    config.port = args.port
    config.file_server_port = args.file_port

    file_server = FileServer(config.data_dir, config.file_server_port)
    try:
        file_server.start()
    except FileNotFoundError as e:
        print(f"错误: {e}")
        sys.exit(1)

    suite = EvidenceTestSuite(config, file_server)
    try:
        success = suite.run_all()
    finally:
        file_server.stop()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
