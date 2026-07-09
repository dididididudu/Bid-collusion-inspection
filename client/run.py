"""
投标串标检测客户端 — 调用服务端 API 进行检测

1. 修改 config.json 中的 server 地址
2. 把 PDF 放入 input/ 文件夹
3. 运行: python run.py

结果保存到 output/（带时间戳，永不覆盖）
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
INPUT_DIR = SCRIPT_DIR / "input"
OUTPUT_DIR = SCRIPT_DIR / "output"
CONFIG_PATH = SCRIPT_DIR / "config.json"

# ── 默认配置（被 config.json 覆盖）──
DEFAULT_CONFIG = {
    "server": "http://127.0.0.1:8000",
    "content_similarity": True,
    "use_gpu": False,
    "poll_interval": 5,
    "timeout": 3600,
}


def load_config() -> dict:
    """读取 config.json，缺失字段使用默认值"""
    config = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                user_config = json.load(f)
            config.update(user_config)
            logger.info(f"已读取配置文件: {CONFIG_PATH}")
        except Exception as e:
            logger.warning(f"配置文件读取失败 ({e})，使用默认配置")
    else:
        logger.info(f"配置文件不存在 ({CONFIG_PATH})，使用默认配置")
        # 写出默认配置供用户修改
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=4)
            logger.info(f"已生成默认配置文件: {CONFIG_PATH}，请修改 server 地址后重新运行")
        except Exception:
            pass
    return config


def find_pdfs(directory: Path) -> list:
    """扫描 input 目录下的所有 PDF 文件"""
    if not directory.exists():
        logger.error(f"input 目录不存在: {directory}")
        logger.info(f"请创建目录 {directory} 并将 PDF 文件放入其中")
        return []

    pdf_files = sorted(directory.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"在 {directory} 中未找到 PDF 文件")
        return []

    total_mb = sum(f.stat().st_size for f in pdf_files) / (1024 * 1024)
    logger.info(f"找到 {len(pdf_files)} 个 PDF 文件（总计 {total_mb:.1f} MB）:")
    for f in pdf_files:
        size_kb = f.stat().st_size / 1024
        logger.info(f"   {f.name} ({size_kb:.0f} KB)")
    return pdf_files


def submit_task(server: str, pdf_files: list, content_similarity: bool, use_gpu: bool) -> str:
    """上传 PDF 并提交检测任务，返回 task_id"""
    url = f"{server.rstrip('/')}/api/detect"
    files = []
    for f in pdf_files:
        files.append(("files", (f.name, open(f, "rb"), "application/pdf")))

    data = {
        "content_similarity": str(content_similarity).lower(),
        "use_gpu": str(use_gpu).lower(),
    }

    logger.info(f"正在上传 {len(pdf_files)} 个文件到 {url} ...")
    try:
        resp = requests.post(url, files=files, data=data, timeout=120)
    finally:
        for _, fobj in files:
            fobj[1].close()

    if resp.status_code != 202:
        logger.error(f"提交失败 (HTTP {resp.status_code}): {resp.text}")
        sys.exit(1)

    result = resp.json()
    task_id = result["task_id"]
    logger.info(f"任务已提交! task_id={task_id}")
    return task_id


def poll_task(server: str, task_id: str, poll_interval: int, max_time: int) -> dict:
    """轮询等待任务完成，返回完整结果"""
    url = f"{server.rstrip('/')}/api/detect/{task_id}"
    start_time = time.time()

    logger.info(f"等待检测完成 (轮询间隔 {poll_interval}s, 超时 {max_time}s) ...")

    while True:
        elapsed = time.time() - start_time
        if elapsed > max_time:
            logger.error(f"等待超时（{max_time}s），任务可能仍在运行")
            logger.info(f"稍后可通过 {url} 查询结果")
            sys.exit(1)

        try:
            resp = requests.get(url, timeout=30)
        except requests.RequestException as e:
            logger.warning(f"查询失败: {e}，{poll_interval}s 后重试...")
            time.sleep(poll_interval)
            continue

        if resp.status_code != 200:
            logger.warning(f"查询返回 HTTP {resp.status_code}，{poll_interval}s 后重试...")
            time.sleep(poll_interval)
            continue

        data = resp.json()
        status = data.get("status")

        if status == "completed":
            logger.info(f"检测完成! 耗时 {data.get('elapsed_seconds', '?')} 秒")
            result_data = data.get("result", {})
            logger.info(f"   文件数: {result_data.get('total_files', '?')}")
            logger.info(f"   比对对数: {result_data.get('total_pairs', '?')}")
            logger.info(f"   可疑对数: {result_data.get('suspicious_pairs', '?')}")
            if result_data.get("dimensions"):
                hits = [k for k, v in result_data["dimensions"].items()
                        if v.get("hit")]
                logger.info(f"   命中维度: {hits or '无'}")
            report_url = data.get("report_url")
            if report_url:
                logger.info(f"   PDF 报告: {server.rstrip('/')}{report_url}")
            return data

        elif status == "failed":
            logger.error(f"检测失败: {data.get('error', '未知错误')}")
            sys.exit(1)

        else:
            prog = data.get("progress", {})
            phase = prog.get("phase", "处理中")
            current = prog.get("current", 0)
            total = prog.get("total", 0)
            done = len(data.get("partial_results", []))
            logger.info(
                f"   [{elapsed:.0f}s] 状态: {phase} "
                f"({current}/{total})  已完成 {done} 对分析"
            )
            time.sleep(poll_interval)


def save_results(data: dict, server: str) -> Path:
    """保存结果到 output/（带时间戳，永不覆盖）"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    json_path = OUTPUT_DIR / f"result_{timestamp}.json"
    counter = 1
    while json_path.exists():
        json_path = OUTPUT_DIR / f"result_{timestamp}_{counter}.json"
        counter += 1

    save_data = {
        "query_time": now.isoformat(),
        "server": server,
        "result": data.get("result"),
        "status": data.get("status"),
        "elapsed_seconds": data.get("elapsed_seconds"),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    logger.info(f"结果已保存: {json_path}")

    # 下载 PDF 报告
    report_url = data.get("report_url")
    if report_url:
        try:
            full_url = f"{server.rstrip('/')}{report_url}"
            resp = requests.get(full_url, timeout=60)
            if resp.status_code == 200:
                pdf_path = json_path.with_suffix(".pdf")
                with open(pdf_path, "wb") as f:
                    f.write(resp.content)
                logger.info(f"PDF 报告已下载: {pdf_path}")
        except Exception as e:
            logger.warning(f"PDF 报告下载失败: {e}")

    return json_path


def main():
    config = load_config()

    server = config["server"]
    content_similarity = config["content_similarity"]
    use_gpu = config["use_gpu"]
    poll_interval = config["poll_interval"]
    timeout = config["timeout"]

    logger.info(f"服务器地址: {server}")
    logger.info(f"内容相似度: {'启用' if content_similarity else '关闭'}")
    logger.info(f"GPU 加速: {'启用' if use_gpu else '关闭'}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = find_pdfs(INPUT_DIR)
    if not pdf_files:
        sys.exit(1)

    total_mb = sum(f.stat().st_size for f in pdf_files) / (1024 * 1024)
    if total_mb > 200:
        logger.warning(f"文件总量 {total_mb:.0f} MB 较大，检测可能需要较长时间")

    task_id = submit_task(server, pdf_files, content_similarity, use_gpu)
    result = poll_task(server, task_id, poll_interval, timeout)
    save_results(result, server)


if __name__ == "__main__":
    main()
