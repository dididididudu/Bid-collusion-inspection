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

# ── 一劳永逸解决 __pycache__ 陈旧问题 ──
# ProcessPoolExecutor 子进程可能加载旧 .pyc 导致代码更新不生效。
# 设置环境变量后子进程继承该设置，不会写入/读取 .pyc。
import os as _os
_os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys as _sys
_sys.dont_write_bytecode = True
# 启动时清理一次项目目录下已有的 __pycache__
for _root, _dirs, _files in _os.walk(_os.path.dirname(_os.path.abspath(__file__))):
    if '__pycache__' in _dirs:
        import shutil as _shutil
        _shutil.rmtree(_os.path.join(_root, '__pycache__'), ignore_errors=True)

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pkg_resources')
warnings.filterwarnings('ignore', category=UserWarning, module='jieba')
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
from fastapi.responses import JSONResponse, FileResponse
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

# API 日志轮转（10MB × 3 备份）
_log_handler = logging.handlers.RotatingFileHandler(
    'api_server.log', maxBytes=10 * 1024 * 1024, backupCount=3,
    encoding='utf-8',
)
_log_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logging.getLogger().addHandler(_log_handler)

# 根日志器级别设为 INFO，让正常运行信息也能写入日志
logging.getLogger().setLevel(logging.INFO)

# 第三方库保持 WARNING，避免刷屏
for _lib in ['jieba', 'paddleocr', 'paddle', 'paddlex', 'easyocr',
             'torch', 'cv2', 'matplotlib', 'urllib3',
             'transformers', 'sentence_transformers', 'sklearn', 'PIL']:
    logging.getLogger(_lib).setLevel(logging.WARNING)

# 内部模块设为 INFO，让轨道、阶段、进度日志可见
for _mod in ['pipeline.checkpoint', 'pipeline.streaming_context',
             'pipeline.parallel_workers', 'pipeline.ocr_helpers',
             'extraction.pdf_extractor', 'extraction.feature_cache',
             'extraction.text_processor', 'matching.paragraph_matcher',
             'matching.semantic_matcher', 'matching.lsh_index',
             'matching.selector', 'embedding.embedding_engine',
             'image_analysis.image_hasher', 'image_analysis.image_matcher',
             'image_analysis.image_ocr']:
    logging.getLogger(_mod).setLevel(logging.INFO)

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

    logger.info("[1/2] 加载 SBERT 语义模型（优先本地缓存）...")
    try:
        # 先试离线（模型已下载过则瞬间完成）
        from sentence_transformers import SentenceTransformer
        _ = SentenceTransformer(
            'paraphrase-multilingual-MiniLM-L12-v2',
            device='cpu', cache_folder='./models',
            trust_remote_code=True, local_files_only=True,
        )
        logger.info("  SBERT 离线加载完成")
        _global_sbert_model = _
    except Exception as e:
        logger.warning(f"  SBERT 本地加载失败（首次运行需要联网下载）: {e}")
        logger.info("  SBERT 将在 Phase 1.5 按需加载")

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

# 后台定时清理：每小时删除超过 1 小时的旧任务目录
def _cleanup_worker():
    while True:
        time.sleep(3600)
        now = time.time()
        for name in os.listdir(WORK_DIR):
            path = os.path.join(WORK_DIR, name)
            if os.path.isdir(path) and len(name) == 36:
                if (now - os.path.getmtime(path)) / 3600 > 1:
                    shutil.rmtree(path, ignore_errors=True)

_cleanup_thread = threading.Thread(target=_cleanup_worker, daemon=True)
_cleanup_thread.start()


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
        self.output_dir = None  # 报告输出目录（PDF下载用）
        # 增量结果支持
        self.partial_results = []
        self._lock = threading.Lock()

    def add_partial_result(self, data: dict):
        """线程安全地添加一条增量结果"""
        with self._lock:
            self.partial_results.append(data)

    def update_progress(self, phase: str, current: int, total: int):
        """线程安全地更新进度"""
        with self._lock:
            self.progress = {"phase": phase, "current": current, "total": total}

    def to_dict(self):
        base = {
            "task_id": self.task_id,
            "status": self.status,
            "progress": self.progress,
            "error": self.error,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "elapsed_seconds": self.elapsed_seconds,
            "content_similarity": self.content_similarity,
        }
        if self.status == "completed":
            base["result"] = self.result
            base["report_url"] = f"/api/detect/{self.task_id}/report"
        if self.status == "processing":
            # 处理中：返回已完成的部分结果
            with self._lock:
                base["partial_results"] = list(self.partial_results)
        return base


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

        # 打印任务配置摘要
        file_list = os.listdir(input_dir) if os.path.isdir(input_dir) else []
        file_sizes = {}
        for fname in file_list:
            fpath = os.path.join(input_dir, fname)
            file_sizes[fname] = os.path.getsize(fpath) if os.path.isfile(fpath) else 0
        total_mb = sum(file_sizes.values()) / (1024 * 1024)
        logger.info(f"任务 {task_id[:8]} 启动 — {len(file_list)} 个文件 ({total_mb:.1f} MB), "
                    f"content_similarity={record.content_similarity}, "
                    f"gpu={opts.get('use_gpu', False)}, ocr={opts.get('ocr_engine', 'easyocr')}")
        logger.info(f"任务 {task_id[:8]} 文件列表: {', '.join(file_list)}")

        t0 = time.time()

        # 进度回调：Phase 3 每分析完一对就推送到 TaskRecord
        _pair_count = [0]
        def _on_progress(data: dict):
            if data.get('phase_start'):
                # Phase 3 启动通知：记录总对数
                record.update_progress("Phase 3: 段落分析", 0, data.get('total_pairs', 0))
                return
            if data.get('update_progress'):
                # 纯进度更新（串行路径）
                record.update_progress("Phase 3: 段落分析", data['current'], data['total'])
                return
            # 真实的配对完成结果
            _pair_count[0] += 1
            record.add_partial_result(data)

        orchestrator = BidDetectionOrchestrator(config, progress_callback=_on_progress)

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
            "dimensions": dims,
            "metadata_groups": report_dict.get("metadata_groups", []),  # ★ 聚合组
            "pairwise_results": report_dict.get("pairwise_results", []),
        }
        record.elapsed_seconds = round(elapsed, 2)
        record.status = "completed"
        record.completed_at = datetime.now().isoformat()
        record.output_dir = output_dir

        # 任务完成摘要
        dim_hits = [k for k, v in dims.items() if v.get("hit")]
        logger.info(f"任务 {task_id[:8]} 完成 ─ "
                    f"{report.total_files} 个文件, {report.total_pairs} 对, "
                    f"{report.suspicious_pairs} 对可疑, "
                    f"耗时 {record.elapsed_seconds:.1f}s, "
                    f"命中维度: {dim_hits or '无'}")

        # 清理缓存和检查点目录（保留 output 目录供 PDF 下载）
        for subdir in ['cache', 'checkpoints', 'input']:
            path = os.path.join(record.work_dir, subdir)
            if os.path.exists(path):
                import shutil
                shutil.rmtree(path, ignore_errors=True)

    except Exception as e:
        logger.exception(f"任务 {task_id[:8]} 失败: {e}")
        record.status = "failed"
        record.error = str(e)
        record.completed_at = datetime.now().isoformat()
    finally:
        # 不再清理 work_dir，留给定时任务清理（1小时后自动删除）
        pass


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
                if ev.text_evidence.local_similarity >= 0.3 or ev.image_evidence.common_image_count > 0 or ev.image_evidence.text_identical_count > 0:
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


@app.get("/api/detect/{task_id}/report")
async def download_report(task_id: str):
    """下载检测报告 PDF"""
    from fastapi.responses import FileResponse

    with _lock:
        record = _tasks.get(task_id)

    if not record:
        raise HTTPException(status_code=404, detail="任务不存在")
    if record.status != "completed":
        raise HTTPException(status_code=400, detail="任务尚未完成")
    if not record.output_dir:
        raise HTTPException(status_code=404, detail="报告文件未找到")

    pdf_path = os.path.join(record.output_dir, "detection_report.pdf")
    json_path = os.path.join(record.output_dir, "detection_report.json")

    # 默认返回 PDF，支持 ?format=json
    fmt = os.environ.get('REQUEST_QUERY', '')  # 会被 query param 覆盖
    if not fmt:
        fmt = ''  # 由 FastAPI 的 query 参数接管

    # 用路径判断：PDF 存在则优先返回 PDF
    if os.path.exists(pdf_path):
        return FileResponse(
            pdf_path,
            media_type='application/pdf',
            filename='detection_report.pdf',
        )
    elif os.path.exists(json_path):
        return FileResponse(
            json_path,
            media_type='application/json',
            filename='detection_report.json',
        )
    else:
        raise HTTPException(status_code=404, detail="报告文件不存在")


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
