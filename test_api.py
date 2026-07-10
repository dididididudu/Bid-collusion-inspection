"""
围标串标检测 API 测试脚本

用法:
    python test_api.py                          # 使用默认测试文件
    python test_api.py file1.pdf file2.pdf      # 指定测试文件
    python test_api.py --host 192.168.1.100 --port 8000 file1.pdf file2.pdf

功能:
    1. 提交检测任务（上传 PDF 文件）
    2. 轮询任务状态直到完成
    3. 下载检测报告
    4. 打印检测结果摘要
"""

import os
import sys
import time
import json
import requests
from datetime import datetime

API_HOST = os.environ.get("BID_TEST_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("BID_TEST_PORT", "8000"))
BASE_URL = f"http://{API_HOST}:{API_PORT}"

POLL_INTERVAL = 3  # 轮询间隔（秒）
MAX_WAIT = 600     # 最大等待时间（秒）


def print_banner(text):
    print("=" * 60)
    print(f" {text}")
    print("=" * 60)


def test_health():
    """测试健康检查端点"""
    print_banner("[1/4] 健康检查")
    try:
        resp = requests.get(f"{BASE_URL}/api/health", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            print(f"  ✅ 服务正常: {data}")
            return True
        else:
            print(f"  ❌ 健康检查失败: HTTP {resp.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"  ❌ 无法连接到 {BASE_URL}")
        print(f"  请确认服务已启动: python api_server.py")
        return False
    except Exception as e:
        print(f"  ❌ 异常: {e}")
        return False


def test_dimensions():
    """测试维度查询端点"""
    print_banner("[2/4] 查询检测维度")
    try:
        resp = requests.get(f"{BASE_URL}/api/dimensions", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            print(f"  ✅ 可用维度:")
            for dim in data.get("dimensions", []):
                status = "启用" if dim.get("default") else "禁用"
                print(f"     - {dim['id']}: {dim['name']} ({status})")
            return True
        else:
            print(f"  ❌ 查询失败: HTTP {resp.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ 异常: {e}")
        return False


def submit_detection(pdf_files):
    """提交检测任务"""
    print_banner("[3/4] 提交检测任务")
    print(f"  上传文件: {', '.join(os.path.basename(f) for f in pdf_files)}")

    files = []
    opened_files = []
    try:
        for fp in pdf_files:
            if not os.path.exists(fp):
                print(f"  ❌ 文件不存在: {fp}")
                return None
            f = open(fp, "rb")
            opened_files.append(f)
            files.append(("files", (os.path.basename(fp), f, "application/pdf")))

        data = {
            "content_similarity": "true",
            "use_gpu": "false",
            "ocr_engine": "",
        }

        print(f"  正在上传...")
        upload_start = time.time()
        resp = requests.post(
            f"{BASE_URL}/api/detect",
            files=files,
            data=data,
            timeout=120,
        )
        upload_time = time.time() - upload_start

        if resp.status_code == 202:
            result = resp.json()
            task_id = result["task_id"]
            print(f"  ✅ 任务已提交 (上传耗时 {upload_time:.1f}s)")
            print(f"     任务 ID: {task_id}")
            print(f"     消息: {result.get('message', '')}")
            return task_id
        else:
            print(f"  ❌ 提交失败: HTTP {resp.status_code}")
            try:
                print(f"     错误: {resp.json()}")
            except Exception:
                print(f"     响应: {resp.text[:500]}")
            return None
    except Exception as e:
        print(f"  ❌ 异常: {e}")
        return None
    finally:
        for f in opened_files:
            f.close()


def poll_task(task_id):
    """轮询任务状态"""
    print_banner("[4/4] 等待检测完成")
    print(f"  轮询间隔: {POLL_INTERVAL}s, 最大等待: {MAX_WAIT}s")

    start_time = time.time()
    last_progress = ""

    while True:
        elapsed = time.time() - start_time
        if elapsed > MAX_WAIT:
            print(f"\n  ❌ 超时（{MAX_WAIT}s）")
            return None

        try:
            resp = requests.get(f"{BASE_URL}/api/detect/{task_id}", timeout=30)
            if resp.status_code != 200:
                print(f"\n  ❌ 查询失败: HTTP {resp.status_code}")
                return None

            data = resp.json()
            status = data.get("status", "unknown")
            progress = data.get("progress", "")
            elapsed_str = f"[{elapsed:.0f}s]"

            if status == "completed":
                print(f"\n  ✅ 检测完成!")
                print(f"     耗时: {data.get('elapsed_seconds', '?')}s")
                return data
            elif status == "failed":
                print(f"\n  ❌ 检测失败: {data.get('error', '未知错误')}")
                return data
            elif status == "processing":
                progress_str = f"{progress}"
                if progress_str != last_progress:
                    print(f"\n  {elapsed_str} 状态: {progress_str}")
                    last_progress = progress_str
                else:
                    print(f"  {elapsed_str} 状态: {progress_str}", end="\r")
            else:
                print(f"  {elapsed_str} 状态: {status} - {progress}")

        except Exception as e:
            print(f"\n  ⚠️ 查询异常: {e}")

        time.sleep(POLL_INTERVAL)


def print_results(result):
    """打印检测结果摘要"""
    print_banner("检测结果摘要")

    result_data = result.get("result", {})
    if not result_data:
        print("  ⚠️ 无结果数据")
        return

    print(f"  报告 ID: {result_data.get('report_id', 'N/A')}")
    print(f"  文件总数: {result_data.get('total_files', 'N/A')}")
    print(f"  文档对数: {result_data.get('total_pairs', 'N/A')}")
    print(f"  可疑对数: {result_data.get('suspicious_pairs', 'N/A')}")

    dims = result_data.get("dimensions", {})
    if dims:
        print(f"\n  维度命中:")
        for dim_key, dim_info in dims.items():
            enabled = dim_info.get("enabled", False)
            hit = dim_info.get("hit", False)
            name = dim_info.get("name", dim_key)
            if enabled:
                mark = "🔴" if hit else "🟢"
                print(f"    {mark} {name}: {'命中' if hit else '未命中'}")

    pairs = result_data.get("pairwise_results", [])
    if pairs:
        print(f"\n  文档对详情 (前 5 对):")
        for i, pair in enumerate(pairs[:5]):
            sim = pair.get("text_similarity", 0)
            matches = pair.get("text_match_count", 0)
            print(f"    [{i+1}] 相似度: {sim:.3f}, 匹配段落: {matches}")

    report_url = result.get("report_url", "")
    if report_url:
        print(f"\n  📄 报告下载: {BASE_URL}{report_url}")


def download_report(task_id, output_dir="."):
    """下载检测报告"""
    print_banner("下载报告")
    try:
        resp = requests.get(f"{BASE_URL}/api/detect/{task_id}/report", timeout=60)
        if resp.status_code == 200:
            content_type = resp.headers.get("content-type", "")
            if "pdf" in content_type:
                filename = f"report_{task_id[:8]}.pdf"
            else:
                filename = f"report_{task_id[:8]}.json"

            filepath = os.path.join(output_dir, filename)
            with open(filepath, "wb") as f:
                f.write(resp.content)
            print(f"  ✅ 报告已下载: {filepath} ({len(resp.content)} bytes)")
        else:
            print(f"  ⚠️ 下载失败: HTTP {resp.status_code}")
    except Exception as e:
        print(f"  ⚠️ 下载异常: {e}")


def main():
    global API_HOST, API_PORT, BASE_URL

    print_banner("围标串标检测 API 测试脚本")
    print(f"  服务地址: {BASE_URL}")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    args = sys.argv[1:]
    pdf_files = []

    i = 0
    while i < len(args):
        if args[i] == "--host" and i + 1 < len(args):
            API_HOST = args[i + 1]
            BASE_URL = f"http://{API_HOST}:{API_PORT}"
            i += 2
        elif args[i] == "--port" and i + 1 < len(args):
            API_PORT = int(args[i + 1])
            BASE_URL = f"http://{API_HOST}:{API_PORT}"
            i += 2
        else:
            pdf_files.append(args[i])
            i += 1

    if not pdf_files:
        test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_input")
        if os.path.isdir(test_dir):
            pdf_files = [
                os.path.join(test_dir, f)
                for f in sorted(os.listdir(test_dir))
                if f.lower().endswith(".pdf")
            ]

        if len(pdf_files) < 2:
            print(f"\n  ❌ 需要至少 2 个 PDF 文件")
            print(f"  用法: python test_api.py file1.pdf file2.pdf")
            print(f"  或将 PDF 文件放入 test_input/ 目录")
            sys.exit(1)

        print(f"  使用 test_input 目录中的文件: {len(pdf_files)} 个")

    print(f"  测试文件: {len(pdf_files)} 个")
    print()

    if not test_health():
        sys.exit(1)
    print()

    if not test_dimensions():
        print("  ⚠️ 维度查询失败，继续执行...")
    print()

    task_id = submit_detection(pdf_files)
    if not task_id:
        sys.exit(1)
    print()

    result = poll_task(task_id)
    if not result:
        sys.exit(1)
    print()

    if result.get("status") == "completed":
        print_results(result)
        print()
        download_report(task_id)

    print_banner("测试完成")


if __name__ == "__main__":
    main()
