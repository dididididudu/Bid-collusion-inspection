#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
围标串标检测 — 综合测试框架

用法:
    python test_api.py                              # 运行所有测试（API + 内部模块）
    python test_api.py --module health              # 只运行健康检查模块
    python test_api.py --module lightweight         # 只运行轻量维度测试
    python test_api.py --module validation          # 只运行参数校验测试
    python test_api.py --module heavy               # 只运行重量维度测试（耗时）
    python test_api.py --module internal            # 只运行内部模块单元测试（无需 API）
    python test_api.py --test test_health_ok        # 只运行单个测试
    python test_api.py --list                       # 列出所有测试
    python test_api.py --host 127.0.0.1 --port 8001 # 指定服务地址
    python test_api.py --skip-heavy                 # 跳过耗时测试

测试模块:
    health       — 健康检查 + itemCode 列表（需 API）
    validation   — 参数校验（未知 itemCode、空 companies 等，需 API）
    lightweight  — 5 个轻量维度（文件码/作者/编辑/联系人/公司名，需 API）
    heavy        — 2 个重量维度（技术标/商务标雷同，需 API）
    internal     — 内部模块单元测试（动态生成 PDF 测试核心算法，无需 API）

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
import random
import shutil
import tempfile
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
            "TECH_BID_SIMILAR", "Business_BID_SIMILAR",
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
    HEAVY_CODES = ["TECH_BID_SIMILAR", "Business_BID_SIMILAR"]

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
# 测试模块 6: 内部模块单元测试（无需 API 服务）
# ============================================================

class InternalUnitTest:
    """内部模块单元测试 — 直接测试提取/匹配逻辑，无需启动 API

    通过动态生成 PDF 来验证各检测维度的核心算法。
    """

    def __init__(self, reporter: TestReporter):
        self.reporter = reporter
        self.tmpdir = tempfile.mkdtemp(prefix='bid_internal_test_')

    def run_all(self):
        for name, fn in [("文本相似度", self._test_text_sim),
                         ("文件码雷同", self._test_file_id),
                         ("文档作者", self._test_author),
                         ("编辑经办人", self._test_editor),
                         ("联系人雷同", self._test_contact),
                         ("公司名雷同", self._test_company),
                         ("信用代码", self._test_credit)]:
            self._run_module(name, fn)
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        print("  临时文件已清理")

    def _run_module(self, name, fn):
        print(f"\n  ▶ {name}")
        t0 = time.time()
        try:
            fn()
            elapsed = time.time() - t0
            self.reporter.add("internal", name, True, elapsed)
        except Exception as e:
            elapsed = time.time() - t0
            self.reporter.add("internal", name, False, elapsed,
                              detail=f"{type(e).__name__}: {e}")
            traceback.print_exc()

    # ------------------------------------------------------------------ #
    #  PDF 生成工具
    # ------------------------------------------------------------------ #
    @staticmethod
    def _make_pdf(path, pages, meta=None):
        import fitz
        doc = fitz.open()
        for txt in pages:
            page = doc.new_page(width=595, height=842)
            y = 72
            for raw_line in txt.strip().split('\n'):
                line = raw_line.strip()
                if not line:
                    y += 12
                    continue
                is_heading = line.startswith('# ')
                if is_heading:
                    line = line[2:]
                font_size = 14 if is_heading else 10
                for start in range(0, len(line), 42):
                    page.insert_text(
                        (72, y), line[start:start + 42],
                        fontsize=font_size, fontname="china-s",
                    )
                    y += 22 if is_heading else 16
                    if y > 780:
                        page = doc.new_page(width=595, height=842)
                        y = 72
        if meta:
            doc.set_metadata({k: v for k, v in meta.items() if v})
        doc.save(path)
        doc.close()

    def _make_3page_pdf(self, path, company, contact, cv,
                        author='', creator='', producer=''):
        intro = cv.get('intro', '公司简介。')
        about = cv.get('about', '业务介绍。')
        qual = cv.get('qualification', '资质认证。')
        tech = cv.get('tech_approach', '技术方案。')
        detail = cv.get('tech_detail', '技术细节。')
        qa = cv.get('quality_assurance', '质量保证。')
        ccl = cv.get('credit_code_line', '')
        extra = cv.get('extra', '')
        pages = [
            f"# {company} 投标文件\n\n项目名称：XX智慧政务平台\n\n{intro}\n\n{about}\n\n{qual}",
            f"# 技术方案\n\n{tech}\n\n{detail}\n\n{qa}",
            f"# 联系方式\n\n联系人：{contact.get('name', '')}\n联系电话：{contact.get('phone', '')}\n电子邮箱：{contact.get('email', '')}\n\n# 公司信息\n\n公司名称：{company}\n{ccl}\n{extra}",
        ]
        self._make_pdf(path, pages,
                       {'author': author, 'creator': creator, 'producer': producer} or None)

    # ------------------------------------------------------------------ #
    #  文本相似度
    # ------------------------------------------------------------------ #
    TEXT_SIM_CASES = [
        ("标书模板完全相同",
         ("A公司", {"name": "张", "phone": "13800000001", "email": "a@a.com"},
          {"intro": "我公司成立于2010年注册资本5000万元。", "about": "专注于政务信息化建设十五年。",
           "qualification": "拥有ISO9001和CMMI5等资质认证。",
           "tech_approach": "采用微服务架构支持高并发低延迟。", "tech_detail": "系统基于SpringCloud框架。",
           "quality_assurance": "提供7x24小时技术支持2小时响应。"}),
         ("B公司", {"name": "李", "phone": "13900000002", "email": "b@b.com"},
          {"intro": "我公司成立于2010年注册资本5000万元。", "about": "专注于政务信息化建设十五年。",
           "qualification": "拥有ISO9001和CMMI5等资质认证。",
           "tech_approach": "采用微服务架构支持高并发低延迟。", "tech_detail": "系统基于SpringCloud框架。",
           "quality_assurance": "提供7x24小时技术支持2小时响应。"}),
         True, "完全相同内容应匹配"),
        ("部分段落重叠",
         ("C公司", {"name": "王", "phone": "13700000003", "email": "c@c.com"},
          {"intro": "我公司专注于政务信息化领域具有丰富经验。", "about": "已完成多个大型项目。",
           "qualification": "拥有ISO9001和系统集成资质。",
           "tech_approach": "采用微服务架构支持高并发低延迟。", "tech_detail": "系统基于SpringCloud框架构建。",
           "quality_assurance": "提供5x8小时技术支持服务。"}),
         ("D公司", {"name": "赵", "phone": "13600000004", "email": "d@d.com"},
          {"intro": "我们专注于大数据分析领域。", "about": "核心产品是数据分析平台。",
           "qualification": "拥有ISO9001资质。",
           "tech_approach": "采用微服务架构支持高并发低延迟。", "tech_detail": "系统基于SpringCloud框架构建。",
           "quality_assurance": "提供7x24小时技术支持服务。"}),
         True, "部分重叠应匹配"),
        ("完全不同行业不应匹配",
         ("E公司", {"name": "钱", "phone": "13500000005", "email": "e@e.com"},
          {"intro": "本公司专注于人工智能领域核心产品为智能客服机器人。", "about": "拥有多项AI软件著作权。",
           "qualification": "已通过ISO9001认证。",
           "tech_approach": "采用深度学习框架PyTorch训练模型。", "tech_detail": "模型部署在GPU集群上推理延迟低于50毫秒。",
           "quality_assurance": "提供7x24小时在线文档和技术支持。"}),
         ("F公司", {"name": "孙", "phone": "13400000006", "email": "f@f.com"},
          {"intro": "本公司专注于生物医药研发核心产品为创新药物。", "about": "拥有多项药品发明专利。",
           "qualification": "已通过GMP药品质量管理认证。",
           "tech_approach": "采用基因编辑技术CRISPRCas9进行药物开发。", "tech_detail": "实验数据表明药效提升百分之三十。",
           "quality_assurance": "严格按照GLP药物非临床研究规范执行实验。"}),
         False, "完全不同行业不应匹配"),
        ("共用招标模板",
         ("G公司", {"name": "周", "phone": "13300000007", "email": "g@g.com"},
          {"intro": "我公司成立于2015年。具有丰富的行业经验。", "about": "拥有相关资质认证。",
           "qualification": "详见资质证书附件。",
           "tech_approach": "技术方案详见投标文件。", "tech_detail": "具体参数见技术规格书。",
           "quality_assurance": "提供售后服务。"}),
         ("H公司", {"name": "吴", "phone": "13200000008", "email": "h@h.com"},
          {"intro": "我公司成立于2016年。具有丰富的行业经验。", "about": "拥有相关资质认证。",
           "qualification": "详见资质证书附件。",
           "tech_approach": "技术方案详见投标文件。", "tech_detail": "具体参数见技术规格书。",
           "quality_assurance": "提供售后服务。"}),
         True, "共有模板语应匹配"),
        ("长文本部分重复",
         ("I公司", {"name": "郑", "phone": "13100000009", "email": "i@i.com"},
          {"intro": "我公司成立于2012年注册资本1亿元是国家级高新技术企业。多年来深耕智慧城市领域。",
           "about": "拥有员工500余人其中研发人员占比60%以上。通过了CMMI5认证。",
           "qualification": "通过ISO9001/ISO14001/ISO45001三体系认证。",
           "tech_approach": "采用业界领先的云原生技术架构基于Kubernetes容器编排平台实现弹性伸缩。",
           "tech_detail": "前端使用React框架后端使用SpringBoot微服务架构。",
           "quality_assurance": "提供7x24小时技术支持服务2小时响应4小时到场8小时解决问题。"}),
         ("J公司", {"name": "冯", "phone": "13000000010", "email": "j@j.com"},
          {"intro": "我公司成立于2013年注册资本8000万元是国家级高新技术企业。多年来深耕金融科技领域。",
           "about": "拥有员工300余人其中研发人员占比50%以上。通过了CMMI3认证。",
           "qualification": "通过ISO9001和ISO27001认证。",
           "tech_approach": "采用业界领先的云原生技术架构基于Kubernetes容器编排平台实现弹性伸缩。",
           "tech_detail": "前端使用Vue框架后端使用SpringBoot微服务架构。",
           "quality_assurance": "提供7x24小时技术支持服务2小时响应4小时到场8小时解决问题。"}),
         True, "长文本部分重复应匹配"),
        ("极短内容相同",
         ("超短A", {"name": "甲", "phone": "1", "email": "a@b"},
          {"intro": "本公司投标。", "about": "。", "qualification": "。",
           "tech_approach": "。", "tech_detail": "。", "quality_assurance": "。"}),
         ("超短B", {"name": "乙", "phone": "2", "email": "b@c"},
          {"intro": "本公司投标。", "about": "。", "qualification": "。",
           "tech_approach": "。", "tech_detail": "。", "quality_assurance": "。"}),
         True, "极短但相同应匹配"),
        ("含编码的相似模板",
         ("K公司", {"name": "张", "phone": "13800000011", "email": "k@k.com"},
          {"intro": "项目编号GCHG2024001项目预算580万元。", "about": "合同编号HT2024001。",
           "qualification": "证书编号ZJ2024001。",
           "tech_approach": "标准编号GBT222392019。", "tech_detail": "设备型号Huawei2288HV7。",
           "quality_assurance": "报修编号BX2024001。"}),
         ("L公司", {"name": "李", "phone": "13900000012", "email": "l@l.com"},
          {"intro": "项目编号GCHG2024002项目预算620万元。", "about": "合同编号HT2024002。",
           "qualification": "证书编号ZJ2024002。",
           "tech_approach": "标准编号GBT222392019。", "tech_detail": "设备型号DellR750xs。",
           "quality_assurance": "报修编号BX2024002。"}),
         True, "含编码模板应匹配"),
        ("中英文混合",
         ("M公司", {"name": "Tom", "phone": "861088888888", "email": "tom@m.com"},
          {"intro": "We are a leading IT company founded in 2010。我们专注于软件开发。",
           "about": "Our team has 200 engineers。核心团队来自知名互联网企业。",
           "qualification": "拥有CMMILevel5和ISO认证。",
           "tech_approach": "采用Agile开发方法Scrum团队协作。", "tech_detail": "技术栈包括JavaPythonGo。",
           "quality_assurance": "提供7x24技术支持。"}),
         ("N公司", {"name": "Jerry", "phone": "862199999999", "email": "jerry@n.com"},
          {"intro": "We are a leading IT company founded in 2012。我们专注于人工智能。",
           "about": "Our team has 150 engineers。核心团队来自知名AI企业。",
           "qualification": "拥有多项AI专利和软件著作权。",
           "tech_approach": "采用Agile开发方法Scrum团队协作。", "tech_detail": "技术栈包括PythonTensorFlow。",
           "quality_assurance": "提供5x8技术支持。"}),
         True, "中英文混合相似应匹配"),
        ("全复用标书模板",
         ("投标人X", {"name": "张", "phone": "13800000013", "email": "x@x.com"},
          {"intro": "我方完全理解并积极响应本次招标文件所有要求。我公司郑重承诺提供优质产品和服务。",
           "about": "我方保证所提供产品均为原装正品享受厂家标准质保服务。",
           "qualification": "我方具有履行合同所必需的设备和专业技术能力。",
           "tech_approach": "我方承诺按招标文件要求的技术规范实施。", "tech_detail": "我方接受招标文件所有商务条款。",
           "quality_assurance": "我方承诺提供不少于三年的免费质保期。"}),
         ("投标人Y", {"name": "李", "phone": "13900000014", "email": "y@y.com"},
          {"intro": "我方完全理解并积极响应本次招标文件所有要求。我公司郑重承诺提供优质产品和服务。",
           "about": "我方保证所提供产品均为原装正品享受厂家标准质保服务。",
           "qualification": "我方具有履行合同所必需的设备和专业技术能力。",
           "tech_approach": "我方承诺按招标文件要求的技术规范实施。", "tech_detail": "我方接受招标文件所有商务条款。",
           "quality_assurance": "我方承诺提供不少于三年的免费质保期。"}),
         True, "全复用模板应匹配"),
    ]

    AUTHOR_CASES = [
        ("相同作者", "张三", "张三", True),
        ("不同作者", "张三", "李四", False),
        ("空作者vs有作者", "", "张三", False),
        ("三字名相同", "张三丰", "张三丰", True),
        ("含空格姓名", "张 三", "张三", False),
        ("含短横姓名", "张-三", "张-三", True),
    ]

    EDITOR_CASES = [
        ("相同creator+producer", "Word", "Microsoft Word", "Word", "Microsoft Word", True),
        ("仅creator相同", "WPS Office", "PDFCreator", "WPS Office", "Acrobat", True),
        ("仅producer相同", "Word", "Acrobat", "LibreOffice", "Acrobat", True),
        ("版本号不同", "Microsoft Word 2021", "Acrobat", "Microsoft Word 2019", "Acrobat", True),
        ("不同软件", "Microsoft Word", "Microsoft Word", "Adobe Acrobat", "Adobe Acrobat", False),
    ]

    CONTACT_CASES = [
        ("相同手机号", ("张", "13800138000", "a@a.com"), ("李", "13800138000", "b@b.com"), True),
        ("相同邮箱", ("张", "13800000001", "same@test.com"), ("李", "13800000002", "same@test.com"), True),
        ("同名+不同手机", ("张三", "13800138000", "a@a.com"), ("张三", "13900139000", "b@b.com"), True),
        ("仅手机相同", ("", "13800000001", ""), ("", "13800000001", ""), True),
        ("仅邮箱相同", ("", "", "only@test.com"), ("", "", "only@test.com"), True),
        ("邮箱大小写", ("", "", "Test@Example.com"), ("", "", "test@example.com"), True),
    ]

    COMPANY_CASES = [
        ("完全相同", "北京华软科技有限公司", "北京华软科技有限公司", True),
        ("不同公司", "北京华软科技有限公司", "上海智联信息技术有限公司", False),
        ("核心名同地域不同", "北京华软科技有限公司", "上海华软科技有限公司", False),
        ("含后缀不同", "华软科技有限公司", "华软科技股份有限公司", False),
        ("空vs有", "", "北京华软科技有限公司", False),
    ]

    CREDIT_CASES = [
        ("完全相同", "91110108MA01XXXXX1", "91110108MA01XXXXX1", True),
        ("不同代码", "91110108MA01XXXXX1", "91110108MA01XXXXX2", False),
        ("空代码", "", "", False),
        ("空vs有", "", "91110108MA01XXXXX1", False),
    ]

    # ------------------------------------------------------------------ #
    #  1. 文本相似度（含位置随机化测试）
    # ------------------------------------------------------------------ #
    def _test_text_sim(self):
        from config import DetectionConfig
        from extraction.pdf_extractor import PyMuPDFExtractor
        from extraction.text_processor import ChunkedTextProcessor
        from extraction.feature_cache import DocumentCache
        from matching.paragraph_matcher import ParagraphMatcher
        cfg = DetectionConfig()
        cfg.CHUNK_PAGE_SIZE = 50
        d = os.path.join(self.tmpdir, "text_sim")
        os.makedirs(d, exist_ok=True)
        ex = PyMuPDFExtractor(cfg)
        tp = ChunkedTextProcessor(cfg)

        # 位置随机化测试
        random.seed(42)
        shared_pool = [
            '本公司承诺完全响应招标文件的所有技术要求提供不少于三年的免费质保期服务。',
            '验收标准应符合招标文件第四章相关规定包括功能验收性能验收安全验收三个阶段。',
            '项目经理须持有高级工程师职称证书并具有不少于十年同类项目管理经验。',
            '系统架构采用分布式微服务设计支持水平扩展和故障自动转移。',
        ]
        uni_a = ['公司A成立于2010年注册资本5000万元。', '公司A已完成15个同类项目。']
        uni_b = ['公司B成立于2012年注册资本8000万元。', '公司B已完成20个同类项目。']
        with_dt = ''
        for trial in range(10):
            t0 = time.time()
            try:
                a_paras = uni_a.copy()
                b_paras = uni_b.copy()
                for sp in shared_pool:
                    a_paras.insert(random.randint(0, len(a_paras)), sp)
                    b_paras.insert(random.randint(0, len(b_paras)), sp)
                p1 = os.path.join(d, f"rt{trial}a.pdf")
                p2 = os.path.join(d, f"rt{trial}b.pdf")
                self._make_pdf(p1, ['\n'.join(a_paras[:3]), '\n'.join(a_paras[3:])])
                self._make_pdf(p2, ['\n'.join(b_paras[:3]), '\n'.join(b_paras[3:])])
                ca = DocumentCache(os.path.join(d, f"rc{trial}"), cfg)
                ids = []
                for p in [p1, p2]:
                    meta, pc, sc = ex.extract_metadata(p)
                    did = ex._generate_doc_id(p)
                    fn = os.path.basename(p)
                    fs = os.path.getsize(p)
                    chs = []
                    [ca.store_chunk(c) or chs.append(c) for c in ex.extract_chunks(p, 50, 0)]
                    feat = tp.aggregate_chunks(
                        doc_id=did, filename=fn, file_size=fs,
                        chunks=chs, metadata=meta, is_scanned=False, page_count=pc
                    )
                    ca.store_document(feat)
                    ids.append(did)
                da = ca.load_document(ids[0])
                db = ca.load_document(ids[1])
                pm = ParagraphMatcher(cfg)
                pa = ca.load_all_paragraphs_full(ids[0])
                pb = ca.load_all_paragraphs_full(ids[1])
                ms = pm.match(da, db, ca, para_full_a=pa, para_full_b=pb)
                ok = len(ms) >= len(shared_pool)
                with_dt += f"\n      位置随机化{trial}: {'PASS' if ok else 'FAIL'} ({len(ms)}/{len(shared_pool)}段)"
                elapsed = time.time() - t0
                self.reporter.add("internal", f"文本相似-位置随机化{trial}", ok, elapsed,
                                  detail=f"共享{len(shared_pool)}段,匹配{len(ms)}段")
            except Exception as e:
                elapsed = time.time() - t0
                self.reporter.add("internal", f"文本相似-位置随机化{trial}", False, elapsed,
                                  detail=f"{type(e).__name__}: {e}")

        # 标准测试用例
        for i, (name, co1, co2, expect, desc) in enumerate(self.TEXT_SIM_CASES):
            t0 = time.time()
            try:
                p1 = os.path.join(d, f"t{i}a.pdf")
                p2 = os.path.join(d, f"t{i}b.pdf")
                self._make_3page_pdf(p1, co1[0], co1[1], co1[2], author='T')
                self._make_3page_pdf(p2, co2[0], co2[1], co2[2], author='T')
                ca = DocumentCache(os.path.join(d, f"c{i}"), cfg)
                ids = []
                for p in [p1, p2]:
                    meta, pc, sc = ex.extract_metadata(p)
                    did = ex._generate_doc_id(p)
                    fn = os.path.basename(p)
                    fs = os.path.getsize(p)
                    chs = []
                    [ca.store_chunk(c) or chs.append(c) for c in ex.extract_chunks(p, 50, 0)]
                    feat = tp.aggregate_chunks(
                        doc_id=did, filename=fn, file_size=fs,
                        chunks=chs, metadata=meta, is_scanned=False, page_count=pc
                    )
                    ca.store_document(feat)
                    ids.append(did)
                da = ca.load_document(ids[0])
                db = ca.load_document(ids[1])
                pm = ParagraphMatcher(cfg)
                if not da.doc_minhash or not db.doc_minhash:
                    ok = False
                    dt = "无minhash"
                else:
                    pa = ca.load_all_paragraphs_full(ids[0])
                    pb = ca.load_all_paragraphs_full(ids[1])
                    ms = pm.match(da, db, ca, para_full_a=pa, para_full_b=pb)
                    ok = len(ms) > 0
                    dt = f"{len(ms)}对匹配最高{max((m.get('similarity', 0) for m in ms), default=0):.3f}"
                ca.close()
                elapsed = time.time() - t0
                self.reporter.add("internal", f"文本相似-{name}", ok == expect, elapsed,
                                  detail=f"{desc}|{dt}")
            except Exception as e:
                elapsed = time.time() - t0
                self.reporter.add("internal", f"文本相似-{name}", False, elapsed,
                                  detail=f"{type(e).__name__}: {e}")

    # ------------------------------------------------------------------ #
    #  2. 文件码雷同
    # ------------------------------------------------------------------ #
    def _test_file_id(self):
        from config import DetectionConfig
        from extraction.pdf_extractor import PyMuPDFExtractor
        cfg = DetectionConfig()
        ex = PyMuPDFExtractor(cfg)
        d = os.path.join(self.tmpdir, "fid")
        os.makedirs(d, exist_ok=True)
        bc = {"intro": "测试文档内容。", "about": "测试。", "qualification": "。",
              "tech_approach": "。", "tech_detail": "。", "quality_assurance": "。"}
        bp = {"name": "", "phone": "", "email": ""}
        a = os.path.join(d, "f0.pdf")
        b = os.path.join(d, "f0b.pdf")
        self._make_3page_pdf(a, "F0", bp, bc)
        shutil.copy2(a, b)
        m1, _, _ = ex.extract_metadata(a)
        m2, _, _ = ex.extract_metadata(b)
        ok = bool(m1.file_id and m1.file_id == m2.file_id)
        self.reporter.add("internal", "文件码-同一副本", ok, 0,
                          detail=f"fid1={m1.file_id} fid2={m2.file_id}")
        for i in range(7):
            p = os.path.join(d, f"u{i}.pdf")
            self._make_3page_pdf(p, f"U{i}", bp, bc)
            me, _, _ = ex.extract_metadata(p)
            ok2 = bool(me.file_id)
            self.reporter.add("internal", f"文件码-独立文件{i}", ok2, 0,
                              detail=f"fid={me.file_id}")

    # ------------------------------------------------------------------ #
    #  3. 文档作者雷同
    # ------------------------------------------------------------------ #
    def _test_author(self):
        from config import DetectionConfig
        from extraction.pdf_extractor import PyMuPDFExtractor
        cfg = DetectionConfig()
        ex = PyMuPDFExtractor(cfg)
        d = os.path.join(self.tmpdir, "auth")
        os.makedirs(d, exist_ok=True)
        bc = {"intro": "测试文档。", "about": "用于作者检测。", "qualification": "。",
              "tech_approach": "。", "tech_detail": "。", "quality_assurance": "。"}
        bp = {"name": "T", "phone": "1", "email": "t@t.com"}
        for i, (name, a1, a2, expect) in enumerate(self.AUTHOR_CASES):
            t0 = time.time()
            try:
                p1 = os.path.join(d, f"a{i}.pdf")
                p2 = os.path.join(d, f"b{i}.pdf")
                self._make_3page_pdf(p1, "A" + str(i), bp, bc, author=a1)
                self._make_3page_pdf(p2, "B" + str(i), bp, bc, author=a2)
                m1, _, _ = ex.extract_metadata(p1)
                m2, _, _ = ex.extract_metadata(p2)
                v1 = (m1.author or '').strip().lower()
                v2 = (m2.author or '').strip().lower()
                ok = bool(v1 and v2 and v1 == v2)
                elapsed = time.time() - t0
                self.reporter.add("internal", f"作者-{name}", ok == expect, elapsed,
                                  detail=f"a1={m1.author} a2={m2.author}")
            except Exception as e:
                elapsed = time.time() - t0
                self.reporter.add("internal", f"作者-{name}", False, elapsed,
                                  detail=f"{type(e).__name__}: {e}")

    # ------------------------------------------------------------------ #
    #  4. 编辑经办人雷同
    # ------------------------------------------------------------------ #
    def _test_editor(self):
        from config import DetectionConfig
        from extraction.pdf_extractor import PyMuPDFExtractor
        cfg = DetectionConfig()
        ex = PyMuPDFExtractor(cfg)
        d = os.path.join(self.tmpdir, "edit")
        os.makedirs(d, exist_ok=True)
        bc = {"intro": "测试。", "about": "用于经办人检测。", "qualification": "。",
              "tech_approach": "。", "tech_detail": "。", "quality_assurance": "。"}
        bp = {"name": "T", "phone": "1", "email": "t@t.com"}
        for i, (name, cr1, pr1, cr2, pr2, expect) in enumerate(self.EDITOR_CASES):
            t0 = time.time()
            try:
                p1 = os.path.join(d, f"e{i}.pdf")
                p2 = os.path.join(d, f"f{i}.pdf")
                self._make_3page_pdf(p1, "E" + str(i), bp, bc, creator=cr1, producer=pr1)
                self._make_3page_pdf(p2, "F" + str(i), bp, bc, creator=cr2, producer=pr2)
                m1, _, _ = ex.extract_metadata(p1)
                m2, _, _ = ex.extract_metadata(p2)
                flds = ['creator', 'producer', 'software_fingerprint']
                matched = [f for f in flds if (
                    (getattr(m1, f, '') or '').lower().strip() == (getattr(m2, f, '') or '').lower().strip()
                    and getattr(m1, f, '') and getattr(m2, f, ''))]
                ok = len(matched) > 0
                elapsed = time.time() - t0
                self.reporter.add("internal", f"经办人-{name}", ok == expect, elapsed,
                                  detail=f"cr1={m1.creator} pr1={m1.producer} vs cr2={m2.creator} pr2={m2.producer}")
            except Exception as e:
                elapsed = time.time() - t0
                self.reporter.add("internal", f"经办人-{name}", False, elapsed,
                                  detail=f"{type(e).__name__}: {e}")

    # ------------------------------------------------------------------ #
    #  5. 联系人雷同
    # ------------------------------------------------------------------ #
    def _test_contact(self):
        from extraction.contact_extractor import extract_contacts_from_text
        d = os.path.join(self.tmpdir, "cont")
        os.makedirs(d, exist_ok=True)
        bc = {"intro": "公司介绍内容。", "about": "关于我们。", "qualification": "。",
              "tech_approach": "。", "tech_detail": "。", "quality_assurance": "。"}
        for i, (name, c1, c2, expect) in enumerate(self.CONTACT_CASES):
            t0 = time.time()
            try:
                p1 = os.path.join(d, f"c{i}a.pdf")
                p2 = os.path.join(d, f"c{i}b.pdf")
                self._make_3page_pdf(p1, f"C{i}A", {"name": c1[0], "phone": c1[1], "email": c1[2]}, bc)
                self._make_3page_pdf(p2, f"C{i}B", {"name": c2[0], "phone": c2[1], "email": c2[2]}, bc)
                import fitz
                t1 = "".join(p.get_text("text") for p in fitz.open(p1))
                t2 = "".join(p.get_text("text") for p in fitz.open(p2))
                fp1 = extract_contacts_from_text(t1)
                fp2 = extract_contacts_from_text(t2)
                s1 = set(fp1.mobile_phones + fp1.emails + [n.strip() for n in fp1.contact_names])
                s2 = set(fp2.mobile_phones + fp2.emails + [n.strip() for n in fp2.contact_names])
                ok = len(s1 & s2) > 0
                elapsed = time.time() - t0
                self.reporter.add("internal", f"联系人-{name}", ok == expect, elapsed,
                                  detail=f"A:手机={fp1.mobile_phones}邮箱={fp1.emails}姓名={fp1.contact_names}|"
                                         f"B:手机={fp2.mobile_phones}邮箱={fp2.emails}姓名={fp2.contact_names}")
            except Exception as e:
                elapsed = time.time() - t0
                self.reporter.add("internal", f"联系人-{name}", False, elapsed,
                                  detail=f"{type(e).__name__}: {e}")

    # ------------------------------------------------------------------ #
    #  6. 公司名雷同
    # ------------------------------------------------------------------ #
    def _test_company(self):
        from extraction.contact_extractor import extract_contacts_from_text
        d = os.path.join(self.tmpdir, "comp")
        os.makedirs(d, exist_ok=True)
        bp = {"name": "T", "phone": "1", "email": "t@t.com"}
        for i, (name, co1, co2, expect) in enumerate(self.COMPANY_CASES):
            t0 = time.time()
            try:
                p1 = os.path.join(d, f"cp{i}a.pdf")
                p2 = os.path.join(d, f"cp{i}b.pdf")
                self._make_3page_pdf(p1, co1, bp,
                                     {"intro": "公司简介。", "about": "。", "qualification": "。",
                                      "tech_approach": "。", "tech_detail": "。", "quality_assurance": "。"})
                self._make_3page_pdf(p2, co2, bp,
                                     {"intro": "公司简介。", "about": "。", "qualification": "。",
                                      "tech_approach": "。", "tech_detail": "。", "quality_assurance": "。"})
                import fitz
                t1 = "".join(p.get_text("text") for p in fitz.open(p1))
                t2 = "".join(p.get_text("text") for p in fitz.open(p2))
                fp1 = extract_contacts_from_text(t1)
                fp2 = extract_contacts_from_text(t2)
                ok = len(set(fp1.company_names) & set(fp2.company_names)) > 0
                elapsed = time.time() - t0
                self.reporter.add("internal", f"公司名-{name}", ok == expect, elapsed,
                                  detail=f"A公司名={fp1.company_names} B公司名={fp2.company_names}")
            except Exception as e:
                elapsed = time.time() - t0
                self.reporter.add("internal", f"公司名-{name}", False, elapsed,
                                  detail=f"{type(e).__name__}: {e}")

    # ------------------------------------------------------------------ #
    #  7. 信用代码雷同
    # ------------------------------------------------------------------ #
    def _test_credit(self):
        from extraction.contact_extractor import extract_contacts_from_text
        d = os.path.join(self.tmpdir, "cred")
        os.makedirs(d, exist_ok=True)
        bp = {"name": "T", "phone": "1", "email": "t@t.com"}
        for i, (name, cc1, cc2, expect) in enumerate(self.CREDIT_CASES):
            t0 = time.time()
            try:
                p1 = os.path.join(d, f"cr{i}a.pdf")
                p2 = os.path.join(d, f"cr{i}b.pdf")
                bc1 = {"intro": "", "about": "", "qualification": "", "tech_approach": "",
                       "tech_detail": "", "quality_assurance": "", "credit_code_line": f"统一社会信用代码：{cc1}"}
                self._make_3page_pdf(p1, "CR" + str(i), bp, bc1)
                bc2 = dict(bc1)
                bc2["credit_code_line"] = f"统一社会信用代码：{cc2}"
                self._make_3page_pdf(p2, "CR" + str(i) + "b", bp, bc2)
                import fitz
                t1 = "".join(p.get_text("text") for p in fitz.open(p1))
                t2 = "".join(p.get_text("text") for p in fitz.open(p2))
                fp1 = extract_contacts_from_text(t1)
                fp2 = extract_contacts_from_text(t2)
                ok = len(set(fp1.credit_codes) & set(fp2.credit_codes)) > 0
                elapsed = time.time() - t0
                self.reporter.add("internal", f"信用代码-{name}", ok == expect, elapsed,
                                  detail=f"A={fp1.credit_codes} B={fp2.credit_codes}")
            except Exception as e:
                elapsed = time.time() - t0
                self.reporter.add("internal", f"信用代码-{name}", False, elapsed,
                                  detail=f"{type(e).__name__}: {e}")


# ============================================================
# 测试注册表
# ============================================================

ALL_MODULES = {
    "health": HealthTest,
    "validation": ValidationTest,
    "lightweight": LightweightTest,
    "heavy": HeavyTest,
    "error": ErrorHandlingTest,
    "internal": InternalUnitTest,
}


# ============================================================
# 主入口
# ============================================================

def list_tests():
    print("\n可用测试模块:")
    for name, cls in ALL_MODULES.items():
        via = "（需 API 服务）" if name != "internal" else "（无需 API，直接测试内部模块）"
        print(f"  {name:15s} — {cls.__doc__ or cls.__name__} {via}")
    print("\n用法: python test_api.py --module <module_name>")
    print("      python test_api.py --module internal")


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

    has_api_tests = any(m != "internal" for m in modules_to_run)
    has_internal = "internal" in modules_to_run

    if has_api_tests:
        print(f"[INFO] API 测试需要服务地址: {config.base_url}")
        file_server.start()
        data_manager = TestDataManager(file_server)
        pdfs = data_manager.get_pdf_files()
        if not pdfs:
            print(f"错误: 测试数据目录 {config.data_dir} 中没有 PDF 文件")
            file_server.stop()
            return
        print(f"[TestData] 找到 {len(pdfs)} 个测试 PDF: {', '.join(pdfs)}")
        print(f"[TestData] 测试公司数: {len(data_manager.get_test_companies())}")
    else:
        data_manager = None  # internal 模块不需要

    print(f"\n开始测试: {', '.join(modules_to_run)}")
    if has_api_tests:
        print(f"服务地址: {config.base_url}")
    print("=" * 70)

    t0 = time.time()
    try:
        for module_name in modules_to_run:
            if module_name not in ALL_MODULES:
                print(f"[WARN] 未知模块: {module_name}, 跳过")
                continue

            test_cls = ALL_MODULES[module_name]

            if module_name == "internal":
                test_instance = test_cls(reporter)
            else:
                test_instance = test_cls(config, reporter, data_manager)

            print(f"\n--- 运行模块: {module_name} ---")
            test_instance.run_all()
    finally:
        if has_api_tests:
            file_server.stop()

    total_time = time.time() - t0
    all_passed = reporter.print_summary()
    print(f"总耗时: {total_time:.1f}s")

    if args.report in ("json", "both"):
        reporter.generate_json()

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
