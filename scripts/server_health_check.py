"""
围标串标 API 服务健康检查脚本
用于验证服务器部署后服务是否正常运行

用法:
  # 默认连接本地 http://localhost:8001
  python scripts/server_health_check.py

  # 指定远程服务器
  python scripts/server_health_check.py --api-url http://192.168.1.100:8001

  # 指定 PDF 目录（自动预制到 batch_downloads 跳过下载）
  python scripts/server_health_check.py --pdf-dir ./input

  # 只做轻量检查（不测试重量级管线）
  python scripts/server_health_check.py --lightweight-only

  # 静默模式（只输出 JSON 结果摘要）
  python scripts/server_health_check.py --quiet
"""
import os
import sys
import json
import time
import argparse
import shutil
import urllib.parse
from pathlib import Path
from datetime import datetime

import re

try:
    import requests
except ImportError:
    print("[FAIL] 需要 requests 库: pip install requests")
    sys.exit(1)


# ============================================================
# 配置
# ============================================================

# 所有检查项（与 API 一致）
ITEM_CODES = {
    "FILE_CODE_SIMILAR": "文件码雷同",
    "EDITOR_SIGNER_SIMILAR": "编辑经办人雷同",
    "DOC_AUTHOR_SIMILAR": "文档作者雷同",
    "SAME_BID_CONTACT_SIMILAR": "人名雷同",
    "SAME_bidderName_SIMILAR": "公司名雷同",
    "TECH_BID_SIMILAR": "技术标雷同",
    "BID_COMPANY_NAME_ABNORMAL": "商务标雷同",
}

LIGHTWEIGHT_ITEMS = {
    "FILE_CODE_SIMILAR", "EDITOR_SIGNER_SIMILAR", "DOC_AUTHOR_SIMILAR",
    "SAME_BID_CONTACT_SIMILAR", "SAME_bidderName_SIMILAR",
}

HEAVY_ITEMS = {
    "TECH_BID_SIMILAR", "BID_COMPANY_NAME_ABNORMAL",
}

DEFAULT_API_URL = "http://localhost:8001"
POLL_INTERVAL = 2        # 轮询间隔（秒）
MAX_POLL_TIME = 120      # 最大等待时间（秒）


# ============================================================
# 工具函数
# ============================================================

def log(msg: str, end="\n"):
    print(msg, end=end, flush=True)


def divider(title: str):
    pad = (60 - len(title) - 2) // 2
    log(f"\n{'=' * 60}")
    log(f"{'=' * pad} {title} {'=' * pad}")
    log(f"{'=' * 60}\n")


def ok(msg: str):
    log(f"  [OK] {msg}")


def fail(msg: str):
    log(f"  [FAIL] {msg}")


def warn(msg: str):
    log(f"  [WARN] {msg}")


def pretty_json(obj):
    """格式化 JSON 输出"""
    return json.dumps(obj, ensure_ascii=False, indent=2)


def sanitize_filename(name: str) -> str:
    """与 API 中 _sanitize_filename 一致"""
    name = re.sub(r'[\\/*?:"<>|]', "_", name)
    return name.strip().replace(" ", "_")


# ============================================================
# 测试核心
# ============================================================

class ServerTester:
    def __init__(self, api_url: str, pdf_dir: str = None,
                 lightweight_only: bool = False, quiet: bool = False):
        self.api_url = api_url.rstrip("/")
        self.pdf_dir = Path(pdf_dir) if pdf_dir else None
        self.lightweight_only = lightweight_only
        self.quiet = quiet
        self.results = {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "details": [],
        }

        # 自动检测项目根目录
        self._project_root = Path(__file__).resolve().parent.parent
        self._batch_downloads = self._project_root / "batch_downloads"

    def _req(self, method: str, path: str, **kwargs) -> requests.Response:
        """发送 HTTP 请求"""
        url = f"{self.api_url}{path}"
        timeout = kwargs.pop("timeout", 30)
        try:
            return requests.request(method, url, timeout=timeout, **kwargs)
        except requests.exceptions.ConnectionError:
            log(f"\n  [NETWORK] 无法连接: {url}")
            log(f"  请确认服务已启动且地址正确")
            return None
        except Exception as e:
            log(f"\n  [NETWORK] 请求异常: {e}")
            return None

    # ── 文件预制 ──────────────────────────────────────────

    def prepare_pdfs(self, batch_id: int, companies: list) -> dict:
        """将 PDF 预制到 batch_downloads/{batch_id}/ 使 API 跳过下载"""
        if not self.pdf_dir:
            return {}

        batch_dir = self._batch_downloads / str(batch_id)
        batch_dir.mkdir(parents=True, exist_ok=True)

        prepared = {}
        for c in companies:
            src = self.pdf_dir / c["src"] if isinstance(c["src"], str) else c["src"]
            if not src.exists():
                warn(f"源文件不存在: {src}")
                continue
            dest_name = f"{c['recordId']}_{sanitize_filename(c['name'])}.pdf"
            dest = batch_dir / dest_name
            shutil.copy2(str(src), str(dest))
            prepared[c["recordId"]] = str(dest)
            ok(f"预制 PDF: {dest_name} ({dest.stat().st_size / 1024:.0f} KB)")

        if prepared:
            ok(f"共预制 {len(prepared)} 个文件到 {batch_dir}")
        return prepared

    # ── 单项测试 ──────────────────────────────────────────

    def test_health(self):
        """测试健康检查端点"""
        divider("健康检查")
        r = self._req("GET", "/api/v1/collusive-check/health")
        if r is None:
            return False
        if r.status_code != 200:
            fail(f"状态码: {r.status_code}")
            return False
        data = r.json()
        ok(f"状态: {data.get('status')}")
        ok(f"活跃任务: {data.get('active_tasks', 0)}")
        ok(f"时间戳: {data.get('timestamp', 'N/A')}")
        if not self.quiet:
            log(f"\n返回数据:\n{pretty_json(data)}")
        return True

    def test_item_codes(self):
        """测试检查项列表"""
        divider("检查项列表")
        r = self._req("GET", "/api/v1/collusive-check/item-codes")
        if r is None:
            return False
        if r.status_code != 200:
            fail(f"状态码: {r.status_code}")
            return False
        data = r.json()
        items = data.get("items", [])
        ok(f"共 {len(items)} 个检查项")
        for item in items:
            log(f"    {item['code']:30s} {item['name']}")
        if not self.quiet:
            log(f"\n返回数据:\n{pretty_json(data)}")
        return True

    def test_analyze(self, label: str, item_code: str, companies: list,
                     batch_id: int = None, expect_error: int = None) -> dict:
        """测试单项检查（同步接口，直接返回结果）"""
        if not self.quiet:
            divider(f"{label} ({item_code})")
        else:
            log(f"  [{item_code}] ", end="")

        bid = batch_id or int(time.time())
        payload = {
            "batchId": bid,
            "projectId": 10001,
            "checkMode": "SAME_SECTION",
            "itemCode": item_code,
            "companies": companies,
        }

        # 同步接口：POST 直接返回 AnalyzeResponse（HTTP 200）
        # 重量维度可能耗时数分钟，超时设为 1800s
        timeout = 1800 if item_code in HEAVY_ITEMS else 60
        r = self._req("POST", "/api/v1/collusive-check/items/analyze",
                      json=payload, timeout=timeout)
        if r is None:
            if not self.quiet:
                fail("无法连接 API")
            return {"passed": False, "data": None, "error": "connection_error"}

        if expect_error:
            if r.status_code == expect_error:
                data = r.json()
                if not self.quiet:
                    ok(f"期望错误 {expect_error}: {data.get('detail', data)}")
                return {"passed": True, "data": data, "error": None}
            else:
                if not self.quiet:
                    fail(f"期望状态码 {expect_error}，实际 {r.status_code}")
                return {"passed": False, "data": None, "error": f"status_{r.status_code}"}

        if r.status_code != 200:
            if not self.quiet:
                fail(f"检查失败 (HTTP {r.status_code}): {r.text[:200]}")
            return {"passed": False, "data": None, "error": f"http_{r.status_code}"}

        result = r.json()
        company_results = result.get("results", [])
        if not self.quiet:
            log(f"\n结果摘要:")
            for cr in company_results:
                icon = "PASS" if cr["status"] == "SUCCESS" else "FAIL"
                log(f"  [{icon}] ID={cr['companyRecordId']:>4d}  {cr['status']:7s}  {cr['summary']}")

        # 同步接口只要有 results 即为成功
        has_results = len(company_results) > 0
        return {"passed": has_results, "data": result, "error": None}

    # ── 完整测试套件 ──────────────────────────────────────

    def run_all(self):
        """运行所有测试"""
        log(f"\n{'=' * 60}")
        log(f"  围标串标 API 服务健康检查")
        log(f"  目标: {self.api_url}")
        log(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if self.pdf_dir:
            log(f"  PDF 目录: {self.pdf_dir}")
        if self.lightweight_only:
            log(f"  模式: 仅轻量检查")
        log(f"{'=' * 60}\n")

        ok("健康检查")
        passed = self.test_health()
        if not passed:
            fail("服务不可用，终止测试")
            return False
        self.results["passed"] += 1

        ok("检查项列表")
        passed = self.test_item_codes()
        if passed:
            self.results["passed"] += 1
        else:
            self.results["failed"] += 1

        # 错误场景测试
        divider("错误场景测试")
        result = self.test_analyze("无效 itemCode", "INVALID_CODE", [
            {"companyRecordId": 1, "registrationCompanyId": 1,
             "sectionId": 1, "bidderName": "测试", "bidFileUrl": "http://x.com/a.pdf"}
        ], expect_error=400)
        if result["passed"]:
            self.results["passed"] += 1
        else:
            self.results["failed"] += 1

        result = self.test_analyze("空 companies", "DOC_AUTHOR_SIMILAR", [], expect_error=400)
        if result["passed"]:
            self.results["passed"] += 1
        else:
            self.results["failed"] += 1

        # 轻量检查项
        if self.pdf_dir:
            pdf_files = sorted(self.pdf_dir.glob("*.pdf"))
            if len(pdf_files) < 2:
                warn(f"PDF 文件不足 2 个（找到 {len(pdf_files)} 个），跳过检查项测试")
            else:
                test_pdfs = pdf_files[:min(3, len(pdf_files))]
                companies = []
                for i, pdf_path in enumerate(test_pdfs):
                    companies.append({
                        "companyRecordId": 500 + i + 1,
                        "registrationCompanyId": 100 + i + 1,
                        "sectionId": 11,
                        "bidderName": f"测试公司{chr(65+i)}",
                        "bidFileUrl": f"http://localhost/{urllib.parse.quote(pdf_path.name)}",
                    })

                batch_id = int(time.time())
                ok(f"准备测试 PDF（batchId={batch_id}）")
                for i, c in enumerate(companies):
                    pdf_path = test_pdfs[i]
                    sanitized_name = sanitize_filename(c["bidderName"])
                    dest_dir = self._batch_downloads / str(batch_id)
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest_path = dest_dir / f"{c['companyRecordId']}_{sanitized_name}.pdf"
                    shutil.copy2(str(pdf_path), str(dest_path))

                test_items = [
                    ("DOC_AUTHOR_SIMILAR", "文档作者雷同"),
                    ("EDITOR_SIGNER_SIMILAR", "编辑经办人雷同"),
                    ("FILE_CODE_SIMILAR", "文件码雷同"),
                    ("BID_COMPANY_NAME_ABNORMAL", "投标文件公司名称异常"),
                    ("SAME_BID_CONTACT_SIMILAR", "同标段单位联系人雷同"),
                ]
                for item_code, item_name in test_items:
                    ok(f"{item_name}")
                    result = self.test_analyze(item_name, item_code, companies, batch_id=batch_id)
                    if result["passed"]:
                        self.results["passed"] += 1
                    else:
                        self.results["failed"] += 1
        else:
            warn("未指定 --pdf-dir，跳过 PDF 相关检查项测试")

        # 重量检查项（可选）
        if not self.lightweight_only and self.pdf_dir and len(pdf_files) >= 2:
            batch_id = int(time.time()) + 1
            companies_heavy = []
            for i, pdf_path in enumerate(test_pdfs[:2]):
                companies_heavy.append({
                    "companyRecordId": 600 + i + 1,
                    "registrationCompanyId": 200 + i + 1,
                    "sectionId": 11,
                    "bidderName": f"重检公司{chr(65+i)}",
                    "bidFileUrl": f"http://localhost/{urllib.parse.quote(pdf_path.name)}",
                })
                sanitized_name = sanitize_filename(companies_heavy[-1]["bidderName"])
                dest_dir = self._batch_downloads / str(batch_id)
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(pdf_path), str(dest_dir / f"{companies_heavy[-1]['companyRecordId']}_{sanitized_name}.pdf"))

            for item_code, item_name in [("TECH_BID_SIMILAR", "技术标雷同"),
                                         ("BID_COMPANY_NAME_ABNORMAL", "商务标雷同")]:
                ok(f"{item_name}（重量级，首次运行耗时较长）")
                result = self.test_analyze(item_name, item_code, companies_heavy, batch_id=batch_id)
                if result["passed"]:
                    self.results["passed"] += 1
                else:
                    self.results["failed"] += 1
        else:
            if self.lightweight_only:
                warn("跳过重量级检查项（--lightweight-only）")

        # 汇总
        elapsed = time.time() - time.time()  # placeholder
        total = self.results["passed"] + self.results["failed"]
        divider("测试汇总")
        log(f"  通过: {self.results['passed']} / {total}")
        log(f"  失败: {self.results['failed']} / {total}")

        return self.results["failed"] == 0


def main():
    parser = argparse.ArgumentParser(description="围标串标 API 服务健康检查")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help=f"API 地址")
    parser.add_argument("--pdf-dir", type=str, help="PDF 输入目录")
    parser.add_argument("--lightweight-only", action="store_true",
                        help="仅测试轻量检查项")
    parser.add_argument("--quiet", "-q", action="store_true", help="简洁输出模式")

    args = parser.parse_args()

    pdf_dir = None
    if args.pdf_dir:
        pdf_dir = Path(args.pdf_dir)
        if not pdf_dir.exists():
            log(f"[FAIL] PDF 目录不存在: {pdf_dir}")
            sys.exit(1)
        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            log(f"[FAIL] PDF 目录中没有 .pdf 文件: {pdf_dir}")
            sys.exit(1)
        log(f"[OK] 找到 {len(pdf_files)} 个 PDF 文件")

    tester = ServerTester(api_url=args.api_url, pdf_dir=pdf_dir,
                          lightweight_only=args.lightweight_only, quiet=args.quiet)
    success = tester.run_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
