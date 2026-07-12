#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
围标串标检查 API — 同步接口测试框架

用法:
    python test_api.py                              # 运行所有测试
    python test_api.py --module health              # 只运行健康检查模块
    python test_api.py --module lightweight         # 只运行轻量维度测试
    python test_api.py --module validation          # 只运行参数校验测试
    python test_api.py --module heavy               # 只运行重量维度测试（耗时）
    python test_api.py --test test_health_ok        # 只运行单个测试
    python test_api.py --list                       # 列出所有测试
    python test_api.py --host 127.0.0.1 --port 8001 # 指定服务地址
    python test_api.py --skip-heavy                 # 跳过耗时测试

测试模块:
    health       — 健康检查 + itemCode 列表
    validation   — 参数校验（未知 itemCode、空 companies 等）
    lightweight  — 5 个轻量维度（文件码/作者/编辑/联系人/公司名）
    heavy        — 2 个重量维度（技术标/商务标雷同，耗时较长）

依赖:
    pip install requests
"""

import os
import sys
import time
import json
import http.server
import threading
import socketserver
import traceback
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

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
        self.timeout = 1800  # 同步接口最长 30 分钟
        # 本地文件服务器端口（用于提供测试 PDF 的 bidFileUrl）
        self.file_server_port = int(os.environ.get("BID_FILE_SERVER_PORT", "18080"))


# ============================================================
# 本地文件服务器（提供 bidFileUrl）
# ============================================================

class FileServerManager:
    """启动一个简单的 HTTP 服务器来提供测试 PDF 文件"""

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
        print(f"[FileServer] 文件服务器启动: http://127.0.0.1:{self.port}/")
        print(f"[FileServer] 数据目录: {self.data_dir}")

    def stop(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
            print("[FileServer] 文件服务器已停止")

    def get_file_url(self, filename: str) -> str:
        return f"http://127.0.0.1:{self.port}/{filename}"


# ============================================================
# 测试数据管理
# ============================================================

class TestDataManager:
    """管理测试用的 PDF 文件和公司信息"""

    def __init__(self, file_server: FileServerManager):
        self.file_server = file_server

    def get_pdf_files(self) -> List[str]:
        """获取测试数据目录中的 PDF 文件名"""
        data_dir = self.file_server.data_dir
        pdfs = []
        for f in os.listdir(data_dir):
            if f.lower().endswith(".pdf"):
                pdfs.append(f)
        return sorted(pdfs)

    def get_test_companies(self) -> List[Dict]:
        """构建测试公司列表（基于 test_input 目录的 PDF 文件）"""
        pdfs = self.get_pdf_files()
        companies = []
        for i, pdf_name in enumerate(pdfs[:3], start=501):
            companies.append({
                "companyRecordId": i,
                "registrationCompanyId": 100 + i,
                "sectionId": 11,
                "bidderName": f"测试公司{i}",
                "bidFileUrl": self.file_server.get_file_url(pdf_name),
            })
        return companies

    def get_invalid_url_companies(self) -> List[Dict]:
        """构建含无效 URL 的公司列表（用于异常测试）"""
        return [
            {
                "companyRecordId": 601,
                "registrationCompanyId": 201,
                "sectionId": 11,
                "bidderName": "正常公司",
                "bidFileUrl": self.file_server.get_file_url(self.get_pdf_files()[0]),
            },
            {
                "companyRecordId": 602,
                "registrationCompanyId": 202,
                "sectionId": 11,
                "bidderName": "无效URL公司",
                "bidFileUrl": "http://127.0.0.1:99999/nonexistent.pdf",
            },
        ]


# ============================================================
# 测试结果报告
# ============================================================

class TestReporter:
    def __init__(self):
        self.results: List[Dict] = []

    def add(self, module: str, test_name: str, passed: bool,
            duration: float, detail: str = "", response: Any = None):
        self.results.append({
            "module": module,
            "test": test_name,
            "passed": passed,
            "duration": round(duration, 2),
            "detail": detail,
            "response": response,
        })

    def print_summary(self):
        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        failed = total - passed
        print("\n" + "=" * 70)
        print(f"测试结果汇总: {passed}/{total} 通过, {failed} 失败")
        print("=" * 70)

        for r in self.results:
            status = "✓ PASS" if r["passed"] else "✗ FAIL"
            print(f"  {status} [{r['module']}] {r['test']} ({r['duration']}s)")
            if not r["passed"] and r["detail"]:
                print(f"         {r['detail'][:200]}")

        print("=" * 70)
        return failed == 0

    def generate_json(self) -> str:
        report = {
            "generated_at": datetime.now().isoformat(),
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r["passed"]),
            "failed": sum(1 for r in self.results if not r["passed"]),
            "results": self.results,
        }
        output_path = os.path.join(os.path.dirname(__file__), "test_report.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"JSON 报告已生成: {output_path}")
        return output_path


# ============================================================
# 测试基类
# ============================================================

class BaseTest:
    def __init__(self, config: TestConfig, reporter: TestReporter,
                 data_manager: TestDataManager):
        self.config = config
        self.reporter = reporter
        self.data = data_manager

    def base_url(self) -> str:
        return self.config.base_url

    def run_test(self, test_name: str, test_func) -> bool:
        t0 = time.time()
        try:
            result = test_func()
            elapsed = time.time() - t0
            passed = result is True or result is None
            self.reporter.add(
                self.module_name, test_name, passed, elapsed,
                detail="" if passed else str(result),
            )
            return passed
        except Exception as e:
            elapsed = time.time() - t0
            self.reporter.add(
                self.module_name, test_name, False, elapsed,
                detail=f"{type(e).__name__}: {e}",
            )
            return False

    @property
    def module_name(self) -> str:
        return self.__class__.__name__.replace("Test", "").lower()

    def _post_analyze(self, payload: dict) -> requests.Response:
        """发送 POST /analyze 请求"""
        return requests.post(
            f"{self.base_url()}/api/v1/collusive-check/items/analyze",
            json=payload,
            timeout=self.config.timeout,
        )


# ============================================================
# 测试模块 1: 健康检查
# ============================================================

class HealthTest(BaseTest):
    def run_all(self):
        self.run_test("test_health_ok", self.test_health_ok)
        self.run_test("test_item_codes", self.test_item_codes)

    def test_health_ok(self):
        resp = requests.get(f"{self.base_url()}/api/v1/collusive-check/health",
                            timeout=10)
        assert resp.status_code == 200, f"期望 200, 实际 {resp.status_code}"
        data = resp.json()
        assert data["status"] == "ok", f"期望 status=ok, 实际 {data.get('status')}"
        assert "supported_items" in data, "缺少 supported_items 字段"
        return True

    def test_item_codes(self):
        resp = requests.get(f"{self.base_url()}/api/v1/collusive-check/item-codes",
                            timeout=10)
        assert resp.status_code == 200, f"期望 200, 实际 {resp.status_code}"
        data = resp.json()
        items = data.get("items", [])
        assert len(items) == 7, f"期望 7 个 itemCode, 实际 {len(items)}"

        codes = {item["code"] for item in items}
        expected = {
            "FILE_CODE_SIMILAR", "EDITOR_SIGNER_SIMILAR", "DOC_AUTHOR_SIMILAR",
            "SAME_BID_CONTACT_SIMILAR", "SAME_bidderName_SIMILAR",
            "TECH_BID_SIMILAR", "BID_COMPANY_NAME_ABNORMAL",
        }
        assert codes == expected, f"itemCode 不匹配\n期望: {expected}\n实际: {codes}"

        assert "COM_BID_SIMILAR" not in codes, "不应包含 COM_BID_SIMILAR"
        assert "SAME_bidderName_SIMILAR" in codes, "应包含 SAME_bidderName_SIMILAR"
        return True


# ============================================================
# 测试模块 2: 参数校验
# ============================================================

class ValidationTest(BaseTest):
    def run_all(self):
        self.run_test("test_unknown_itemcode", self.test_unknown_itemcode)
        self.run_test("test_empty_companies", self.test_empty_companies)
        self.run_test("test_heavy_single_company", self.test_heavy_single_company)

    def test_unknown_itemcode(self):
        companies = self.data.get_test_companies()
        resp = self._post_analyze({
            "batchId": 999, "projectId": 1, "checkMode": "SAME_SECTION",
            "itemCode": "UNKNOWN_CODE", "companies": companies,
        })
        assert resp.status_code == 400, f"期望 400, 实际 {resp.status_code}"
        assert "未知 itemCode" in resp.text, f"响应内容: {resp.text[:200]}"
        return True

    def test_empty_companies(self):
        resp = self._post_analyze({
            "batchId": 999, "projectId": 1, "checkMode": "SAME_SECTION",
            "itemCode": "FILE_CODE_SIMILAR", "companies": [],
        })
        assert resp.status_code == 400, f"期望 400, 实际 {resp.status_code}"
        assert "不能为空" in resp.text, f"响应内容: {resp.text[:200]}"
        return True

    def test_heavy_single_company(self):
        companies = self.data.get_test_companies()[:1]
        resp = self._post_analyze({
            "batchId": 999, "projectId": 1, "checkMode": "SAME_SECTION",
            "itemCode": "TECH_BID_SIMILAR", "companies": companies,
        })
        assert resp.status_code == 400, f"期望 400, 实际 {resp.status_code}"
        assert "至少需要 2 家公司" in resp.text, f"响应内容: {resp.text[:200]}"
        return True


# ============================================================
# 测试模块 3: 轻量维度
# ============================================================

class LightweightTest(BaseTest):
    LIGHTWEIGHT_CODES = [
        "FILE_CODE_SIMILAR",
        "EDITOR_SIGNER_SIMILAR",
        "DOC_AUTHOR_SIMILAR",
        "SAME_BID_CONTACT_SIMILAR",
        "SAME_bidderName_SIMILAR",
    ]

    def run_all(self):
        for code in self.LIGHTWEIGHT_CODES:
            self.run_test(f"test_{code.lower()}", lambda c=code: self._test_item(c))

    def _test_item(self, item_code: str):
        companies = self.data.get_test_companies()
        batch_id = int(time.time()) % 100000
        resp = self._post_analyze({
            "batchId": batch_id, "projectId": 1, "checkMode": "SAME_SECTION",
            "itemCode": item_code, "companies": companies,
        })

        assert resp.status_code == 200, (
            f"{item_code}: 期望 200, 实际 {resp.status_code}, 响应: {resp.text[:300]}"
        )

        data = resp.json()
        self._validate_response_structure(data, item_code, companies)
        return True

    def _validate_response_structure(self, data: dict, item_code: str,
                                     companies: List[Dict]):
        assert data["batchId"] is not None, "缺少 batchId"
        assert data["itemCode"] == item_code, (
            f"itemCode 不匹配: 期望 {item_code}, 实际 {data.get('itemCode')}"
        )
        assert data["itemName"], "itemName 为空"
        assert len(data["results"]) == len(companies), (
            f"results 数量不匹配: 期望 {len(companies)}, 实际 {len(data['results'])}"
        )

        for i, result in enumerate(data["results"]):
            assert result["companyRecordId"] == companies[i]["companyRecordId"], (
                f"companyRecordId 不匹配: 期望 {companies[i]['companyRecordId']}, "
                f"实际 {result.get('companyRecordId')}"
            )
            assert result["status"] in ("SUCCESS", "FAILED", "ERROR"), (
                f"非法 status: {result.get('status')}"
            )
            assert isinstance(result["summary"], str), "summary 不是字符串"
            assert "evidence" in result, "缺少 evidence 字段"
            assert isinstance(result["evidence"], dict), "evidence 不是 dict"


# ============================================================
# 测试模块 4: 重量维度
# ============================================================

class HeavyTest(BaseTest):
    HEAVY_CODES = ["TECH_BID_SIMILAR", "BID_COMPANY_NAME_ABNORMAL"]

    def run_all(self):
        for code in self.HEAVY_CODES:
            self.run_test(f"test_{code.lower()}", lambda c=code: self._test_item(c))

    def _test_item(self, item_code: str):
        companies = self.data.get_test_companies()
        if len(companies) < 2:
            return f"测试数据不足（需要至少 2 个 PDF）"

        batch_id = int(time.time()) % 100000
        resp = self._post_analyze({
            "batchId": batch_id, "projectId": 1, "checkMode": "SAME_SECTION",
            "itemCode": item_code, "companies": companies,
        })

        assert resp.status_code == 200, (
            f"{item_code}: 期望 200, 实际 {resp.status_code}, 响应: {resp.text[:300]}"
        )

        data = resp.json()
        self._validate_response_structure(data, item_code, companies)

        dim_label = "技术标" if item_code == "TECH_BID_SIMILAR" else "商务标"
        failed = sum(1 for r in data["results"] if r["status"] == "FAILED")
        success = sum(1 for r in data["results"] if r["status"] == "SUCCESS")
        print(f"\n  [{item_code}] {dim_label}检测结果: {failed} FAILED, {success} SUCCESS")
        return True

    def _validate_response_structure(self, data: dict, item_code: str,
                                     companies: List[Dict]):
        assert data["itemCode"] == item_code
        assert len(data["results"]) == len(companies)
        for result in data["results"]:
            assert result["status"] in ("SUCCESS", "FAILED", "ERROR")


# ============================================================
# 测试模块 5: 异常处理
# ============================================================

class ErrorHandlingTest(BaseTest):
    def run_all(self):
        self.run_test("test_invalid_url", self.test_invalid_url)

    def test_invalid_url(self):
        companies = self.data.get_invalid_url_companies()
        batch_id = int(time.time()) % 100000
        resp = self._post_analyze({
            "batchId": batch_id, "projectId": 1, "checkMode": "SAME_SECTION",
            "itemCode": "DOC_AUTHOR_SIMILAR", "companies": companies,
        })

        assert resp.status_code == 200, f"期望 200, 实际 {resp.status_code}"
        data = resp.json()
        assert len(data["results"]) == len(companies)

        for result in data["results"]:
            assert result["status"] in ("SUCCESS", "FAILED", "ERROR"), (
                f"非法 status: {result.get('status')}"
            )
        return True


# ============================================================
# 测试注册表
# ============================================================

ALL_MODULES = {
    "health": HealthTest,
    "validation": ValidationTest,
    "lightweight": LightweightTest,
    "heavy": HeavyTest,
    "error": ErrorHandlingTest,
}


# ============================================================
# 主入口
# ============================================================

def list_tests():
    print("\n可用测试模块:")
    for name, cls in ALL_MODULES.items():
        print(f"  {name:15s} — {cls.__doc__ or cls.__name__}")
    print("\n用法: python test_api.py --module <module_name>")
    print("      python test_api.py --module health lightweight")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="围标串标检查 API 测试框架")
    parser.add_argument("--host", default="127.0.0.1", help="服务地址")
    parser.add_argument("--port", type=int, default=8001, help="服务端口")
    parser.add_argument("--module", nargs="+", help="指定测试模块")
    parser.add_argument("--test", help="只运行单个测试")
    parser.add_argument("--list", action="store_true", help="列出所有测试")
    parser.add_argument("--skip-heavy", action="store_true", help="跳过重量维度测试")
    parser.add_argument("--report", choices=["json", "both"], help="生成报告")
    args = parser.parse_args()

    if args.list:
        list_tests()
        return

    config = TestConfig()
    config.host = args.host
    config.port = args.port
    config.base_url = f"http://{args.host}:{args.port}"

    reporter = TestReporter()

    file_server = FileServerManager(config.data_dir, config.file_server_port)
    try:
        file_server.start()
    except Exception as e:
        print(f"文件服务器启动失败: {e}")
        print(f"请确保测试数据目录存在: {config.data_dir}")
        return

    data_manager = TestDataManager(file_server)

    pdfs = data_manager.get_pdf_files()
    if not pdfs:
        print(f"错误: 测试数据目录 {config.data_dir} 中没有 PDF 文件")
        file_server.stop()
        return
    print(f"[TestData] 找到 {len(pdfs)} 个测试 PDF: {', '.join(pdfs)}")
    print(f"[TestData] 测试公司数: {len(data_manager.get_test_companies())}")

    modules_to_run = args.module or list(ALL_MODULES.keys())
    if args.skip_heavy and "heavy" in modules_to_run:
        modules_to_run = [m for m in modules_to_run if m != "heavy"]
        print("[INFO] 已跳过重量维度测试")

    print(f"\n开始测试: {', '.join(modules_to_run)}")
    print(f"服务地址: {config.base_url}")
    print("=" * 70)

    t0 = time.time()
    try:
        for module_name in modules_to_run:
            if module_name not in ALL_MODULES:
                print(f"[WARN] 未知模块: {module_name}, 跳过")
                continue

            test_cls = ALL_MODULES[module_name]
            test_instance = test_cls(config, reporter, data_manager)

            print(f"\n--- 运行模块: {module_name} ---")
            test_instance.run_all()
    finally:
        file_server.stop()

    total_time = time.time() - t0
    all_passed = reporter.print_summary()
    print(f"总耗时: {total_time:.1f}s")

    if args.report in ("json", "both"):
        reporter.generate_json()

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
