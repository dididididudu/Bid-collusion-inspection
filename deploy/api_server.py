"""
投标文件串标围标检测 — FastAPI 服务

模型在启动时加载一次，常驻内存。收到请求时直接检测，无冷启动延迟。

启动:
    python deploy/api_server.py

测试:
    curl -X POST http://localhost:8000/detect \
      -F "files=@a.pdf" -F "files=@b.pdf"

查看状态:
    curl http://localhost:8000/status
"""

import os
import shutil
import tempfile
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")


# ============================================================
# 模型常驻：启动时加载，关闭时释放
# ============================================================

_model_server = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期：启动加载模型，关闭释放"""
    global _model_server
    from deploy.model_server import ModelServer

    logger.info("正在启动检测服务...")
    _model_server = ModelServer()
    _model_server.warmup()
    logger.info("服务就绪")
    yield
    _model_server.shutdown()
    logger.info("服务已关闭")


app = FastAPI(
    title="投标文件串标围标检测",
    description="上传 PDF 文件，检测是否存在串标围标嫌疑",
    version="3.0",
    lifespan=lifespan,
)


@app.get("/status")
async def status():
    """查看模型状态"""
    if _model_server is None:
        return {"status": "未初始化"}
    return _model_server.status


@app.post("/detect")
async def detect(files: list[UploadFile] = File(...)):
    """检测上传的 PDF 文件

    Args:
        files: PDF 文件列表

    Returns:
        检测报告 JSON
    """
    if not files:
        return JSONResponse({"error": "请上传至少一个 PDF 文件"}, status_code=400)

    # 创建临时目录
    tmp_input = tempfile.mkdtemp(prefix="bid_input_")
    tmp_output = tempfile.mkdtemp(prefix="bid_output_")

    try:
        # 保存上传文件
        for f in files:
            if not f.filename.lower().endswith('.pdf'):
                return JSONResponse(
                    {"error": f"仅支持 PDF 文件: {f.filename}"},
                    status_code=400,
                )
            path = os.path.join(tmp_input, f.filename)
            with open(path, 'wb') as out:
                content = await f.read()
                out.write(content)

        # 执行检测（模型已在内存中，无需加载）
        logger.info(f"开始检测: {len(files)} 个文件")
        report = _model_server.detect(tmp_input, tmp_output)

        # 返回摘要
        return {
            "status": "完成",
            "total_files": report.total_files,
            "suspicious_pairs": report.suspicious_pairs,
            "high_risk_pairs": report.high_risk_pairs,
            "report_id": report.report_id,
            "details": [
                {
                    "pair_id": r.pair_id,
                    "risk_level": r.risk_level,
                    "risk_score": r.risk_score,
                    "text_similarity": r.similarity_scores.get('text_local', 0),
                    "risk_factors": r.risk_factors[:5],
                }
                for r in report.pairwise_results
                if r.risk_level != "NONE"
            ],
        }

    finally:
        # 清理临时文件
        shutil.rmtree(tmp_input, ignore_errors=True)
        shutil.rmtree(tmp_output, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
