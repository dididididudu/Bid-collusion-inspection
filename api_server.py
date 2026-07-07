"""
FastAPI 服务 — 投标文件串标围标检测

用法:
    python api_server.py
    # 或: uvicorn api_server:app --host 0.0.0.0 --port 8000

端点:
    POST /api/detect          提交检测任务（返回 task_id）
    GET  /api/detect/{task_id} 查询任务状态和结果
    GET  /api/dimensions      查询可用检测维度
"""
import os
import sys
import json
import uuid
import time
import shutil
import logging
import threading
import multiprocessing
from datetime import datetime
from typing import Optional, List
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# 项目根目录
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

from config import DetectionConfig
from pipeline.orchestrator import BidDetectionOrchestrator
from report import ReportGenerator

logger = logging.getLogger(__name__)

# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(
    title="投标文件串标围标检测 API",
    description="上传 PDF 文档，进行串标围标多维检测",
    version="1.0.0",
)


@app.on_event("startup")
async def preload_models():
    """服务启动时预加载所有模型，避免首次请求卡住"""
    logger.info("=" * 50)
    logger.info("正在预加载模型...")

    logger.info("[1/2] 加载 SBERT 语义模型...")
    try:
        model = _get_global_sbert()
        logger.info(f"  SBERT {'加载完成' if model else '加载失败'}")
    except Exception as e:
        logger.warning(f"  SBERT 加载异常: {e}")

    logger.info("[2/2] 加载 OCR 引擎 (EasyOCR)...")
    try:
        from image_analysis.image_ocr import ImageOCREngine
        engine = ImageOCREngine(use_gpu=False, engine='easyocr', offline=True)
        ok = engine.is_available
        logger.info(f"  OCR {'加载完成' if ok else '不可用'}")
    except Exception as e:
        logger.warning(f"  OCR 加载异常: {e}")

    logger.info("模型预加载完成，服务就绪")
    logger.info("=" * 50)


# ============================================================
# 任务存储 + 并发控制
# ============================================================

_tasks = {}  # task_id -> TaskRecord
_lock = threading.Lock()

# 最大并发检测任务数，超过则排队等待
MAX_CONCURRENT_TASKS = 2
_concurrency_semaphore = threading.Semaphore(MAX_CONCURRENT_TASKS)

# 全局共享的 SBERT 模型（只读，线程安全）
_global_sbert_model = None
_sbert_lock = threading.Lock()

# 临时工作目录
WORK_DIR = os.environ.get("BID_WORK_DIR", os.path.join(_project_root, "workdir"))
os.makedirs(WORK_DIR, exist_ok=True)


# ============================================================
# 数据模型
# ============================================================

class TaskRecord:
    """单个检测任务的记录"""
    def __init__(self, task_id: str, content_similarity: bool, options: dict):
        self.task_id = task_id
        self.status = "pending"  # pending / processing / completed / failed
        self.progress = {"phase": "等待中", "current": 0, "total": 0}
        self.result = None
        self.error = None
        self.content_similarity = content_similarity
        self.options = options
        self.created_at = datetime.now().isoformat()
        self.completed_at = None
        self.elapsed_seconds = 0
        self.work_dir = os.path.join(WORK_DIR, task_id)

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "status": self.status,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "elapsed_seconds": self.elapsed_seconds,
            "content_similarity": self.content_similarity,
        }


# ============================================================
# 任务运行函数（后台线程）
# ============================================================

def _get_global_sbert():
    """全局共享 SBERT 模型（避免每个任务加载一次，~500MB）"""
    global _global_sbert_model
    if _global_sbert_model is None:
        with _sbert_lock:
            if _global_sbert_model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                    logger.info("Loading global SBERT model...")
                    _global_sbert_model = SentenceTransformer(
                        'paraphrase-multilingual-MiniLM-L12-v2', device='cpu'
                    )
                    logger.info("Global SBERT model loaded")
                except Exception as e:
                    logger.warning(f"SBERT load failed: {e}")
    return _global_sbert_model


def _run_detection(task_id: str):
    """后台运行检测（受并发数限制，超出的排队等待）"""
    _concurrency_semaphore.acquire(blocking=True)
    try:
        _run_detection_impl(task_id)
    finally:
        _concurrency_semaphore.release()


def _run_detection_impl(task_id: str):
    """实际的检测逻辑"""
    record = _tasks.get(task_id)
    if not record:
        return

    try:
        record.status = "processing"

        config = DetectionConfig()
        config.CACHE_DIR = os.path.join(record.work_dir, "cache")
        config.CHECKPOINT_DIR = os.path.join(record.work_dir, "checkpoints")
        config.DISABLE_CACHE = False

        opts = record.options or {}
        if opts.get("use_gpu"):
            config.USE_GPU = True
            config._auto_detect_device()
        if opts.get("ocr_engine"):
            config.OCR_ENGINE = opts["ocr_engine"]
        if opts.get("ocr_offline"):
            config.OCR_OFFLINE_MODE = True

        config.ENABLE_OCR = True
        config.ENABLED_DIMENSIONS["content_similarity"] = record.content_similarity

        input_dir = os.path.join(record.work_dir, "input")
        output_dir = os.path.join(record.work_dir, "output")

        t0 = time.time()
        orchestrator = BidDetectionOrchestrator(config)

        if record.content_similarity and _global_sbert_model is not None:
            try:
                orchestrator.embedding_engine.model = _global_sbert_model
                orchestrator.embedding_engine.is_available = True
                if orchestrator.semantic_matcher:
                    orchestrator.semantic_matcher.model = _global_sbert_model
                    orchestrator.semantic_matcher.is_available = True
            except Exception:
                pass

        report = orchestrator.detect(input_dir, output_dir)
        elapsed = time.time() - t0

        # 序列化结果为字典
        report_gen = ReportGenerator(config)
        report_dict = report_gen._dataclass_to_dict(report)

        # 提取维度命中情况
        dims = _build_dimension_hits(report, config)

        # 构建简洁结果
        record.result = {
            "report_id": report.report_id,
            "total_files": report.total_files,
            "total_pairs": report.total_pairs,
            "suspicious_pairs": report.suspicious_pairs,
            "high_risk_pairs": report.high_risk_pairs,
            "risk_score": max((r.risk_score for r in report.pairwise_results), default=0),
            "risk_level": _max_risk_level(report),
            "dimensions": dims,
            "pairwise_results": report_dict.get("pairwise_results", []),
        }
        record.elapsed_seconds = round(elapsed, 2)
        record.status = "completed"
        record.completed_at = datetime.now().isoformat()

    except Exception as e:
        logger.exception(f"检测任务失败: {task_id}")
        record.status = "failed"
        record.error = str(e)
        record.completed_at = datetime.now().isoformat()


def _build_dimension_hits(report, config):
    """从检测结果和配置构建各维度的命中情况"""
    dims = config.ENABLED_DIMENSIONS
    result = {}

    for dim_key, dim_default in dims.items():
        if not dim_default:
            result[dim_key] = {"enabled": False, "hit": False}
            continue

        hit = False
        for pair in report.pairwise_results:
            ev = pair.evidence
            if dim_key == "content_similarity":
                if ev.text_evidence.local_similarity >= 0.3 or ev.image_evidence.image_risk_score > 0:
                    hit = True
                    break
            elif dim_key == "file_id":
                if ev.metadata_evidence.same_file_id:
                    hit = True
                    break
            elif dim_key == "author":
                if "author" in ev.metadata_evidence.matched_fields:
                    hit = True
                    break
            elif dim_key == "editor":
                if any(f in ev.metadata_evidence.matched_fields for f in ["creator", "producer", "software_fingerprint"]):
                    hit = True
                    break
            elif dim_key == "contact":
                if ev.contact_evidence.common_mobiles or ev.contact_evidence.common_emails or ev.contact_evidence.common_contacts:
                    hit = True
                    break
            elif dim_key == "company_name":
                if ev.contact_evidence.common_companies:
                    hit = True
                    break
            elif dim_key == "credit_code":
                if ev.contact_evidence.common_credit_codes:
                    hit = True
                    break
            elif dim_key == "member_id":
                if ev.contact_evidence.common_member_ids:
                    hit = True
                    break

        result[dim_key] = {"enabled": True, "hit": hit}

    return result


def _max_risk_level(report) -> str:
    levels = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
    best = "NONE"
    for pair in report.pairwise_results:
        if levels.get(pair.risk_level, 0) > levels.get(best, 0):
            best = pair.risk_level
    return best


# ============================================================
# API 端点
# ============================================================

@app.get("/api/dimensions")
async def get_dimensions():
    """返回系统支持的检测维度列表"""
    return {
        "dimensions": [
            {"id": "content_similarity", "name": "内容相似度（文本+图片）", "group": "heavy", "default": True, "description": "文本段落匹配 + 图片哈希比对，耗时较长"},
            {"id": "file_id",           "name": "文件码雷同",             "group": "metadata", "default": True},
            {"id": "author",            "name": "文档作者雷同",           "group": "metadata", "default": True},
            {"id": "editor",            "name": "编辑经办人雷同",         "group": "metadata", "default": True},
            {"id": "contact",           "name": "单位联系人雷同",         "group": "contact",  "default": True},
            {"id": "company_name",      "name": "公司名称异常",           "group": "contact",  "default": True},
            {"id": "credit_code",       "name": "信用代码雷同",           "group": "contact",  "default": True},
            {"id": "member_id",         "name": "会员号雷同",             "group": "contact",  "default": False},
        ]
    }


@app.post("/api/detect", status_code=202)
async def submit_detection(
    files: List[UploadFile] = File(..., description="PDF 文件列表"),
    content_similarity: bool = Form(True, description="是否启用内容相似度检测（文本+图片）"),
    use_gpu: bool = Form(False, description="是否启用 GPU 加速"),
    ocr_engine: str = Form("easyocr", description="OCR 引擎 (easyocr / paddleocr)"),
):
    """提交检测任务"""
    # 过滤非 PDF 文件
    pdf_files = [f for f in files if f.filename and f.filename.lower().endswith(".pdf")]
    if not pdf_files:
        raise HTTPException(status_code=400, detail="请上传至少一个 PDF 文件")
    if len(pdf_files) < 2:
        raise HTTPException(status_code=400, detail="请上传至少 2 个 PDF 文件进行比较")

    # 创建任务记录
    task_id = str(uuid.uuid4())
    options = {"use_gpu": use_gpu, "ocr_engine": ocr_engine}
    record = TaskRecord(task_id, content_similarity, options)

    with _lock:
        _tasks[task_id] = record

    # 创建工作目录
    input_dir = os.path.join(record.work_dir, "input")
    os.makedirs(input_dir, exist_ok=True)

    # 保存上传的文件（取 basename 避免路径问题）
    for f in pdf_files:
        safe_name = os.path.basename(f.filename)
        file_path = os.path.join(input_dir, safe_name)
        content = await f.read()
        with open(file_path, "wb") as f_out:
            f_out.write(content)

    # 启动后台检测线程
    thread = threading.Thread(target=_run_detection, args=(task_id,), daemon=True)
    thread.start()

    return JSONResponse(
        status_code=202,
        content={
            "task_id": task_id,
            "status": "pending",
            "message": f"检测任务已提交，{len(pdf_files)} 个文件",
        }
    )


@app.get("/api/detect/{task_id}")
async def get_detection_result(task_id: str):
    """查询检测任务状态和结果"""
    with _lock:
        record = _tasks.get(task_id)

    if not record:
        raise HTTPException(status_code=404, detail="任务不存在")

    return record.to_dict()


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


# ============================================================
# 启动（直接运行 python api_server.py 时）
# ============================================================

if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("BID_HOST", "0.0.0.0")
    port = int(os.environ.get("BID_PORT", "8000"))
    log_level = os.environ.get("BID_LOG_LEVEL", "info").lower()
    print(f"🚀 投标串标检测 API 服务启动: http://{host}:{port}")
    print(f"📁 工作目录: {WORK_DIR}")
    print(f"📖 API 文档: http://{host}:{port}/docs")
    uvicorn.run(app, host=host, port=port, log_level=log_level)
