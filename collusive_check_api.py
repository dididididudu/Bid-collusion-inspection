"""
围标串标检查 API — Java 后端异步调用接口

端点:
  POST /api/v1/collusive-check/items/analyze   提交单项检查（返回 taskId）
  GET  /api/v1/collusive-check/items/{taskId}  轮询检查结果

请求格式:
  POST {"batchId", "projectId", "checkMode", "itemCode", "companies"}

工作流:
  1. 下载 PDF（按 batchId 缓存到本地）
  2. 对轻量检查（元数据/联系方式）做定向提取
  3. 对重量检查（文本/图片）运行完整管线，结果缓存复用
  4. 利用 page_classifications 区分技术标/商务标

用法:
    python collusive_check_api.py          # 独立运行（默认 8001 端口）
"""

import os
import re
import sys
import json
import time
import uuid
import asyncio
import logging
import hashlib
import threading
import shutil
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── 项目路径 ──
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logger = logging.getLogger(__name__)

# ============================================================
# 常量
# ============================================================

# 下载目录（按 batchId 组织）
DOWNLOAD_BASE = os.path.join(_project_root, "batch_downloads")
# 工作目录（管线全量检测用）
WORK_DIR = os.path.join(_project_root, "collusive_workdir")
# 缓存 TTL（秒）
CACHE_TTL = 3600
# PDF 下载超时（秒）
DOWNLOAD_TIMEOUT = 60
# 下载重试次数
DOWNLOAD_RETRIES = 3

os.makedirs(DOWNLOAD_BASE, exist_ok=True)
os.makedirs(WORK_DIR, exist_ok=True)

# 全局 SBERT 模型（只读，线程安全，startup 时加载，供 _run_full_pipeline 注入）
_global_sbert_model = None

# ============================================================
# itemCode → 检查项名称映射
# ============================================================

ITEM_CODE_NAMES: Dict[str, str] = {
    "FILE_CODE_SIMILAR": "文件码雷同",
    "EDITOR_SIGNER_SIMILAR": "编辑经办人雷同",
    "DOC_AUTHOR_SIMILAR": "文档作者雷同",
    "SAME_BID_CONTACT_SIMILAR": "人名雷同",
    "SAME_bidderName_SIMILAR": "公司名雷同",
    "TECH_BID_SIMILAR": "技术标雷同",
    "Business_BID_SIMILAR": "商务标雷同",
}

# 轻量检查（无需全量管线，秒级返回）
LIGHTWEIGHT_ITEMS = {
    "FILE_CODE_SIMILAR", "EDITOR_SIGNER_SIMILAR", "DOC_AUTHOR_SIMILAR",
    "SAME_BID_CONTACT_SIMILAR", "SAME_bidderName_SIMILAR",
}

# 重量检查（技术标 + 商务标，各包含文本+图片，需走完整管道）
TECH_BID_ITEMS = {"TECH_BID_SIMILAR"}
COMMERCIAL_BID_ITEMS = {"Business_BID_SIMILAR"}
HEAVY_ITEMS = TECH_BID_ITEMS | COMMERCIAL_BID_ITEMS

# ============================================================
# Pydantic 数据模型
# ============================================================

class CompanyInfo(BaseModel):
    companyRecordId: int
    registrationCompanyId: int
    sectionId: int
    bidderName: str
    bidFileUrl: str

class AnalyzeRequest(BaseModel):
    batchId: int
    projectId: int
    checkMode: str = "SAME_SECTION"
    itemCode: str
    companies: List[CompanyInfo]

class CompanyResult(BaseModel):
    companyRecordId: int
    registrationCompanyId: int
    sectionId: int
    status: str          # SUCCESS / FAILED / ERROR
    summary: str
    evidence: Dict[str, Any] = Field(default_factory=dict)

class AnalyzeResponse(BaseModel):
    batchId: int
    itemCode: str
    itemName: str
    results: List[CompanyResult]

class BatchCache:
    """批次级管线结果缓存（heavy items 复用，线程安全）

    同 batch + 同 dimension 的管线结果缓存 CACHE_TTL 秒，
    避免技术标/商务标分别调用时重复跑全量管道。
    """

    def __init__(self):
        self._cache: Dict[str, dict] = {}
        self._cache_time: Dict[str, float] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            cached = self._cache.get(key)
            ts = self._cache_time.get(key, 0)
            if cached and (time.time() - ts) < CACHE_TTL:
                return cached
        return None

    def set(self, key: str, data: dict):
        with self._lock:
            self._cache[key] = data
            self._cache_time[key] = time.time()

    def invalidate(self, key: str):
        with self._lock:
            self._cache.pop(key, None)
            self._cache_time.pop(key, None)


batch_cache = BatchCache()

# ============================================================
# PDF 下载器
# ============================================================

def _sanitize_filename(name: str) -> str:
    """清理文件名中的非法字符"""
    return re.sub(r'[<>:"/\\|?*]', '_', name)[:80]


def download_batch_pdfs(batch_id: int, companies: List[CompanyInfo]) -> Dict[int, str]:
    """下载一批 PDF 到本地

    Args:
        batch_id: 批次 ID
        companies: 公司信息列表

    Returns:
        {companyRecordId: local_pdf_path, ...}

    Raises:
        RuntimeError: 如果全部下载失败
    """
    batch_dir = os.path.join(DOWNLOAD_BASE, str(batch_id))
    os.makedirs(batch_dir, exist_ok=True)

    def _download_one(company: CompanyInfo):
        record_id = company.companyRecordId
        filename = f"{record_id}_{_sanitize_filename(company.bidderName)}.pdf"
        local_path = os.path.join(batch_dir, filename)

        # 如果已存在且文件有效，跳过下载
        if os.path.exists(local_path) and os.path.getsize(local_path) > 100:
            return record_id, local_path, None

        # 下载（带重试）
        for attempt in range(1, DOWNLOAD_RETRIES + 1):
            try:
                logger.info(f"下载 PDF [batch={batch_id}, company={company.bidderName}] "
                            f"尝试 {attempt}/{DOWNLOAD_RETRIES} ...")
                resp = requests.get(
                    company.bidFileUrl,
                    timeout=DOWNLOAD_TIMEOUT,
                    allow_redirects=True,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                                      "Chrome/120.0.0.0 Safari/537.36",
                    },
                    verify=True,
                )
                resp.raise_for_status()

                # 检查内容类型
                content_type = resp.headers.get("Content-Type", "")
                if "text/html" in content_type and not resp.url.endswith(".pdf"):
                    logger.warning(f"  URL 返回 HTML（可能需鉴权），保存原始响应检查")

                with open(local_path, "wb") as f:
                    f.write(resp.content)

                if len(resp.content) > 100:
                    logger.info(f"  ✅ 下载完成: {local_path} ({len(resp.content) / 1024:.0f}KB)")
                    return record_id, local_path, None
                else:
                    logger.warning(f"  文件过小 ({len(resp.content)} bytes)，重试...")

            except requests.exceptions.Timeout:
                logger.warning(f"  超时 (attempt {attempt})")
            except requests.exceptions.SSLError as e:
                logger.warning(f"  SSL 错误 (attempt {attempt}): {e}")
                try:
                    resp = requests.get(
                        company.bidFileUrl, timeout=DOWNLOAD_TIMEOUT,
                        verify=False, allow_redirects=True,
                        headers={"User-Agent": "Mozilla/5.0 ..."},
                    )
                    resp.raise_for_status()
                    with open(local_path, "wb") as f:
                        f.write(resp.content)
                    if len(resp.content) > 100:
                        logger.info(f"  ✅ 下载完成（跳过 SSL）")
                        return record_id, local_path, None
                except Exception as e2:
                    logger.warning(f"  SSL 跳过后仍失败: {e2}")
            except requests.exceptions.RequestException as e:
                logger.warning(f"  请求失败 (attempt {attempt}): {e}")

            if attempt < DOWNLOAD_RETRIES:
                time.sleep(2 * attempt)

        err_msg = f"公司 {company.bidderName}({record_id}) PDF 下载失败"
        logger.error(err_msg)
        return record_id, "", err_msg

    result = {}
    errors = []
    download_workers = min(
        max(1, int(os.environ.get("PDF_DOWNLOAD_WORKERS", "4"))),
        len(companies),
    )
    logger.info(f"[batch={batch_id}] PDF 下载并发: workers={download_workers}, files={len(companies)}")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=download_workers) as executor:
        futures = [executor.submit(_download_one, company) for company in companies]
        for future in as_completed(futures):
            record_id, local_path, err = future.result()
            if err:
                errors.append(err)
            elif local_path:
                result[record_id] = local_path

    if not result:
        raise RuntimeError(f"所有 PDF 下载失败: {'; '.join(errors)}")
    if errors:
        logger.warning(f"部分 PDF 下载失败: {'; '.join(errors)}")

    return result


# ============================================================
# 轻量检查处理器（无需全量管线）
# ============================================================

def _extract_pdf_metadata(pdf_path: str) -> dict:
    """提取 PDF 元数据"""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        meta = doc.metadata or {}
        doc.close()
        return {
            "author": meta.get("author", ""),
            "creator": meta.get("creator", ""),
            "producer": meta.get("producer", ""),
            "title": meta.get("title", ""),
            "subject": meta.get("subject", ""),
        }
    except Exception as e:
        logger.warning(f"元数据提取失败 {pdf_path}: {e}")
        return {}


def _extract_pdf_file_id(pdf_path: str) -> str:
    """提取 PDF 文件唯一标识

    - 优先读取 PDF trailer 中的 /ID 条目（符合 PDF 规范的文件标识符）
    - 回退：取文件偏移 0x200~0x1200（跳过头部注释）共 4KB 的 MD5
    """
    try:
        import fitz
        doc = fitz.open(pdf_path)

        file_id = ""
        try:
            trailer = doc.pdf_trailer()
            if trailer and isinstance(trailer, dict):
                pdf_id = trailer.get("/ID")
                if pdf_id and isinstance(pdf_id, list) and len(pdf_id) > 0:
                    raw = pdf_id[0]
                    if isinstance(raw, bytes):
                        file_id = hashlib.md5(raw).hexdigest()
                    elif isinstance(raw, str):
                        file_id = hashlib.md5(raw.encode()).hexdigest()
        except Exception:
            pass

        doc.close()

        if file_id:
            return file_id

        # 回退：取文件中部位置 4KB 的 MD5
        with open(pdf_path, "rb") as f:
            f.seek(0, 2)
            file_size = f.tell()
            offset = max(0x200, file_size // 4)
            f.seek(offset)
            sample = f.read(4096)
            if len(sample) == 0:
                f.seek(0)
                sample = f.read()
        return hashlib.md5(sample).hexdigest()
    except Exception as e:
        logger.warning(f"文件 ID 提取失败 {pdf_path}: {e}")
        return ""


def _extract_text_preview(pdf_path: str, max_chars: int = 50000) -> str:
    """快速提取 PDF 文本内容"""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
            if len(text) >= max_chars:
                break
        doc.close()
        return text[:max_chars]
    except Exception as e:
        logger.warning(f"文本提取失败 {pdf_path}: {e}")
        return ""


def _get_pdf_page_count(pdf_path: str) -> int:
    """获取 PDF 页数"""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        count = doc.page_count
        doc.close()
        return count
    except Exception:
        return 0


def _get_batch_work_dirs(batch_id: int, dimension: str = "") -> Dict[str, str]:
    """返回同 batch 共享的工作目录。

    cache/input/output 按 batch 共享，checkpoint 按任务维度隔离，避免技术标
    和商务标的 Phase 3 进度互相覆盖。
    """
    stable_workdir = os.environ.get("COLLUSIVE_STABLE_WORKDIR", "1").lower() in ("1", "true", "yes")
    suffix = "" if stable_workdir else f"_{uuid.uuid4().hex[:8]}"
    work_dir = os.path.join(WORK_DIR, f"batch_{batch_id}{suffix}")
    ckpt_name = f"checkpoints_{dimension or 'text'}"
    return {
        "work_dir": work_dir,
        "input_dir": os.path.join(work_dir, "input"),
        "output_dir": os.path.join(work_dir, "output"),
        "cache_dir": os.path.join(work_dir, "cache"),
        "ckpt_dir": os.path.join(work_dir, ckpt_name),
    }


def _copy_batch_inputs(
    batch_id: int,
    companies: List[CompanyInfo],
    pdf_paths: Dict[int, str],
    input_dir: str,
) -> Dict[int, str]:
    """复制 batch PDF 到稳定 input 目录，返回 companyRecordId -> doc_id。"""
    os.makedirs(input_dir, exist_ok=True)
    doc_id_map = {}
    for c in companies:
        src = pdf_paths.get(c.companyRecordId)
        if src and os.path.exists(src):
            safe_name = f"{c.companyRecordId}_{_sanitize_filename(c.bidderName)}.pdf"
            dst = os.path.join(input_dir, safe_name)
            if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
                shutil.copy2(src, dst)
            doc_id_map[c.companyRecordId] = hashlib.md5(dst.encode('utf-8')).hexdigest()[:16]
    return doc_id_map


def _build_pipeline_config(cache_dir: str, ckpt_dir: str, dimension: str = "all"):
    from config import DetectionConfig

    config = DetectionConfig()
    config.CACHE_DIR = cache_dir
    config.CHECKPOINT_DIR = ckpt_dir
    config.TEXT_EMBEDDING_CACHE_DIR = os.environ.get(
        "TEXT_EMBEDDING_CACHE_DIR",
        os.path.join(WORK_DIR, "shared_embedding_cache"),
    )
    config.ENABLE_OCR = os.environ.get("COLLUSIVE_ENABLE_OCR", "1").lower() in ("1", "true", "yes")
    config.ENABLE_IMAGE_ANALYSIS = os.environ.get("COLLUSIVE_ENABLE_IMAGE_ANALYSIS", "1").lower() in ("1", "true", "yes")
    config.OCR_ENGINE = "paddleocr"
    config.USE_GPU = os.environ.get("USE_GPU", str(config.USE_GPU)).lower() in ("1", "true", "yes")
    env_sbert_device = os.environ.get("SBERT_DEVICE", config.SBERT_DEVICE).lower()
    if env_sbert_device in ("cpu", "cuda", "mps", "auto"):
        config.SBERT_DEVICE = env_sbert_device
    config.SBERT_BATCH_SIZE = int(os.environ.get("SBERT_BATCH_SIZE", config.SBERT_BATCH_SIZE))
    config.PHASE1_WORKERS = int(os.environ.get("PHASE1_WORKERS", config.PHASE1_WORKERS))
    config.PHASE3_WORKERS = int(os.environ.get("PHASE3_WORKERS", config.PHASE3_WORKERS))
    config.TOC_FILTER_ENABLED = True
    config.ANALYSIS_DIMENSION = dimension
    return config


def _warm_text_cache(
    batch_id: int,
    companies: List[CompanyInfo],
    pdf_paths: Dict[int, str],
) -> Dict[int, str]:
    """为联系人/公司名等内容轻量项预热 batch 级文本缓存。

    只执行 Phase 0/1 文本提取并写入 SQLite，不跑 OCR/SBERT/Phase 3。
    """
    from pipeline.orchestrator import BidDetectionOrchestrator

    dirs = _get_batch_work_dirs(batch_id, "text")
    os.makedirs(dirs["output_dir"], exist_ok=True)
    doc_id_map = _copy_batch_inputs(batch_id, companies, pdf_paths, dirs["input_dir"])

    config = _build_pipeline_config(dirs["cache_dir"], dirs["ckpt_dir"], "all")
    config.ENABLE_OCR = False
    config.ENABLE_IMAGE_ANALYSIS = False
    config.ENABLE_CHECKPOINT = False
    config.ENABLED_DIMENSIONS['content_similarity'] = False

    orchestrator = BidDetectionOrchestrator(config)
    try:
        file_paths = [
            os.path.join(dirs["input_dir"], f"{c.companyRecordId}_{_sanitize_filename(c.bidderName)}.pdf")
            for c in companies
            if c.companyRecordId in doc_id_map
        ]
        orchestrator._phase0_metadata(file_paths)
        for file_path in file_paths:
            orchestrator._phase1_extract_single(file_path)
        orchestrator.cache.conn.commit()
        logger.info(f"[batch={batch_id}] 文本缓存预热完成: {len(file_paths)} 个文件")
    finally:
        orchestrator.streaming.clear()
        orchestrator.cache.close()
    return doc_id_map


def handle_file_code_similar(
    companies: List[CompanyInfo],
    pdf_paths: Dict[int, str],
) -> List[CompanyResult]:
    """文件码雷同检测"""
    file_ids = {}
    for c in companies:
        path = pdf_paths.get(c.companyRecordId)
        if path:
            file_ids[c.companyRecordId] = _extract_pdf_file_id(path)

    value_to_ids: Dict[str, List[int]] = {}
    for cid, fid in file_ids.items():
        if fid:
            value_to_ids.setdefault(fid, []).append(cid)

    results = []
    for c in companies:
        cid = c.companyRecordId
        fid = file_ids.get(cid, "")
        if not fid:
            results.append(CompanyResult(
                companyRecordId=cid, registrationCompanyId=c.registrationCompanyId,
                sectionId=c.sectionId, status="SUCCESS",
                summary="未发现文件码雷同", evidence={},
            ))
            continue
        similar = [oid for oid in value_to_ids.get(fid, []) if oid != cid]
        if similar:
            results.append(CompanyResult(
                companyRecordId=cid, registrationCompanyId=c.registrationCompanyId,
                sectionId=c.sectionId, status="FAILED",
                summary=f"文件码与 {len(similar)} 家公司雷同",
                evidence={"fileId": fid, "similarCompanyRecordIds": similar},
            ))
        else:
            results.append(CompanyResult(
                companyRecordId=cid, registrationCompanyId=c.registrationCompanyId,
                sectionId=c.sectionId, status="SUCCESS",
                summary="未发现文件码雷同", evidence={},
            ))
    return results


def handle_editor_signer_similar(
    companies: List[CompanyInfo],
    pdf_paths: Dict[int, str],
) -> List[CompanyResult]:
    """编辑经办人雷同检测"""
    meta_info = {}
    for c in companies:
        path = pdf_paths.get(c.companyRecordId)
        if path:
            meta_info[c.companyRecordId] = _extract_pdf_metadata(path)

    editor_fingerprints: Dict[str, List[int]] = {}
    for cid, meta in meta_info.items():
        fp = f"{meta.get('creator','')}|{meta.get('producer','')}"
        if fp and fp != "|":
            editor_fingerprints.setdefault(fp, []).append(cid)

    results = []
    for c in companies:
        cid = c.companyRecordId
        meta = meta_info.get(cid, {})
        fp = f"{meta.get('creator','')}|{meta.get('producer','')}"
        similar = [oid for oid in editor_fingerprints.get(fp, []) if oid != cid] if fp != "|" else []
        if similar:
            editor_name = meta.get('creator', '') or meta.get('producer', '') or '未知'
            results.append(CompanyResult(
                companyRecordId=cid, registrationCompanyId=c.registrationCompanyId,
                sectionId=c.sectionId, status="FAILED",
                summary=f"编辑经办人与 {len(similar)} 家公司雷同",
                evidence={"editor": editor_name, "similarCompanyRecordIds": similar},
            ))
        else:
            results.append(CompanyResult(
                companyRecordId=cid, registrationCompanyId=c.registrationCompanyId,
                sectionId=c.sectionId, status="SUCCESS",
                summary="未发现编辑经办人雷同", evidence={},
            ))
    return results


def handle_doc_author_similar(
    companies: List[CompanyInfo],
    pdf_paths: Dict[int, str],
) -> List[CompanyResult]:
    """文档作者雷同检测"""
    meta_info = {}
    for c in companies:
        path = pdf_paths.get(c.companyRecordId)
        if path:
            meta_info[c.companyRecordId] = _extract_pdf_metadata(path)

    author_groups: Dict[str, List[int]] = {}
    for cid, meta in meta_info.items():
        author = (meta.get("author", "") or "").strip()
        if author:
            author_groups.setdefault(author, []).append(cid)

    results = []
    for c in companies:
        cid = c.companyRecordId
        meta = meta_info.get(cid, {})
        author = (meta.get("author", "") or "").strip()
        similar = [oid for oid in author_groups.get(author, []) if oid != cid] if author else []
        if similar:
            results.append(CompanyResult(
                companyRecordId=cid, registrationCompanyId=c.registrationCompanyId,
                sectionId=c.sectionId, status="FAILED",
                summary=f"文档作者与 {len(similar)} 家公司雷同",
                evidence={"author": author, "similarCompanyRecordIds": similar},
            ))
        else:
            results.append(CompanyResult(
                companyRecordId=cid, registrationCompanyId=c.registrationCompanyId,
                sectionId=c.sectionId, status="SUCCESS",
                summary="未发现文档作者雷同", evidence={},
            ))
    return results


def handle_bidder_name_similar(
    companies: List[CompanyInfo],
    pdf_paths: Dict[int, str],
    batch_id: int = 0,
) -> List[CompanyResult]:
    """公司名雷同检测 — 本公司文件中出现其他投标公司名称

    对应 itemCode: SAME_bidderName_SIMILAR
    """
    from extraction.contact_extractor import extract_contacts_from_sqlite, extract_contacts_from_text
    from extraction.feature_cache import DocumentCache

    company_name_in_docs: Dict[int, List[str]] = {}
    doc_id_map = _warm_text_cache(batch_id, companies, pdf_paths) if batch_id else {}
    cache = None
    if doc_id_map:
        dirs = _get_batch_work_dirs(batch_id, "text")
        cache = DocumentCache(dirs["cache_dir"])
    try:
        for c in companies:
            doc_id = doc_id_map.get(c.companyRecordId)
            if cache is not None and doc_id:
                ci = extract_contacts_from_sqlite(doc_id, cache)
                company_name_in_docs[c.companyRecordId] = ci.company_names
                continue
            path = pdf_paths.get(c.companyRecordId)
            if path and os.path.exists(path):
                text = _extract_text_preview(path)
                ci = extract_contacts_from_text(text)
                company_name_in_docs[c.companyRecordId] = ci.company_names
    finally:
        if cache is not None:
            cache.close()

    bidder_names: Dict[str, int] = {
        c.bidderName: c.companyRecordId for c in companies
    }

    results = []
    for c in companies:
        cid = c.companyRecordId
        names_in_doc = company_name_in_docs.get(cid, [])
        found_other_bidders = []
        for name in names_in_doc:
            other_cid = bidder_names.get(name)
            if other_cid and other_cid != cid:
                found_other_bidders.append(other_cid)

        found_other_bidders = list(set(found_other_bidders))

        if found_other_bidders:
            other_names = [
                next((cc.bidderName for cc in companies if cc.companyRecordId == oid), str(oid))
                for oid in found_other_bidders
            ]
            results.append(CompanyResult(
                companyRecordId=cid, registrationCompanyId=c.registrationCompanyId,
                sectionId=c.sectionId, status="FAILED",
                summary=f"投标文件中出现其他投标公司名称: {', '.join(other_names)}",
                evidence={"foundCompanyNames": [n for n in names_in_doc
                          if bidder_names.get(n) in found_other_bidders],
                          "similarCompanyRecordIds": found_other_bidders},
            ))
        else:
            results.append(CompanyResult(
                companyRecordId=cid, registrationCompanyId=c.registrationCompanyId,
                sectionId=c.sectionId, status="SUCCESS",
                summary="未发现公司名雷同", evidence={},
            ))
    return results


def handle_same_bid_contact_similar(
    companies: List[CompanyInfo],
    pdf_paths: Dict[int, str],
    batch_id: int = 0,
) -> List[CompanyResult]:
    """同标段单位联系人雷同 — 检查人名、手机号、邮箱跨公司雷同"""
    from extraction.contact_extractor import extract_contacts_from_sqlite, extract_contacts_from_text
    from extraction.feature_cache import DocumentCache

    contacts = {}
    doc_id_map = _warm_text_cache(batch_id, companies, pdf_paths) if batch_id else {}
    cache = None
    if doc_id_map:
        dirs = _get_batch_work_dirs(batch_id, "text")
        cache = DocumentCache(dirs["cache_dir"])
    try:
        for c in companies:
            doc_id = doc_id_map.get(c.companyRecordId)
            if cache is not None and doc_id:
                contacts[c.companyRecordId] = extract_contacts_from_sqlite(doc_id, cache)
                continue
            path = pdf_paths.get(c.companyRecordId)
            if path and os.path.exists(path):
                text = _extract_text_preview(path)
                contacts[c.companyRecordId] = extract_contacts_from_text(text)
    finally:
        if cache is not None:
            cache.close()

    mobile_index: Dict[str, List[int]] = {}
    phone_index: Dict[str, List[int]] = {}
    email_index: Dict[str, List[int]] = {}
    name_index: Dict[str, List[int]] = {}
    for cid, ci in contacts.items():
        for m in ci.mobile_phones:
            mobile_index.setdefault(m, []).append(cid)
        for p in ci.landline_phones:
            phone_index.setdefault(p, []).append(cid)
        for e in ci.emails:
            email_index.setdefault(e, []).append(cid)
        for n in set(ci.contact_names + ci.potential_names):
            name_index.setdefault(n, []).append(cid)

    results = []
    for c in companies:
        cid = c.companyRecordId
        ci = contacts.get(cid)
        if not ci:
            results.append(CompanyResult(
                companyRecordId=cid, registrationCompanyId=c.registrationCompanyId,
                sectionId=c.sectionId, status="SUCCESS",
                summary="未识别到联系人信息", evidence={},
            ))
            continue

        related_ids = set()
        shared_mobiles = set()
        shared_phones = set()
        shared_emails = set()
        shared_names = set()

        for m in ci.mobile_phones:
            for oid in mobile_index.get(m, []):
                if oid != cid:
                    related_ids.add(oid)
                    shared_mobiles.add(m)
        for p in ci.landline_phones:
            for oid in phone_index.get(p, []):
                if oid != cid:
                    related_ids.add(oid)
                    shared_phones.add(p)
        for e in ci.emails:
            for oid in email_index.get(e, []):
                if oid != cid:
                    related_ids.add(oid)
                    shared_emails.add(e)
        for n in set(ci.contact_names + ci.potential_names):
            for oid in name_index.get(n, []):
                if oid != cid:
                    related_ids.add(oid)
                    shared_names.add(n)

        if related_ids:
            evidence = {}
            if shared_mobiles:
                evidence["commonMobiles"] = sorted(shared_mobiles)
            if shared_phones:
                evidence["commonPhones"] = sorted(shared_phones)
            if shared_emails:
                evidence["commonEmails"] = sorted(shared_emails)
            if shared_names:
                evidence["commonPersons"] = sorted(shared_names)
            evidence["similarCompanyRecordIds"] = sorted(related_ids)

            summary_parts = []
            if shared_mobiles:
                summary_parts.append("手机号")
            if shared_phones:
                summary_parts.append("固话")
            if shared_emails:
                summary_parts.append("邮箱")
            if shared_names:
                summary_parts.append("人名")
            summary_str = "+".join(summary_parts) if summary_parts else "联系方式"

            results.append(CompanyResult(
                companyRecordId=cid, registrationCompanyId=c.registrationCompanyId,
                sectionId=c.sectionId, status="FAILED",
                summary=f"{summary_str}与 {len(related_ids)} 家公司雷同",
                evidence=evidence,
            ))
        else:
            results.append(CompanyResult(
                companyRecordId=cid, registrationCompanyId=c.registrationCompanyId,
                sectionId=c.sectionId, status="SUCCESS",
                summary="未发现联系人信息雷同", evidence={},
            ))
    return results


# ============================================================
# 重量检查 — 全量管线复用
# ============================================================

def _run_full_pipeline(
    batch_id: int,
    companies: List[CompanyInfo],
    pdf_paths: Dict[int, str],
    item_code: str = "TECH_BID_SIMILAR",
) -> dict:
    """运行全量串标检测管线

    按 itemCode 决定处理技术标(technical)或商务标(commercial)维度。
    同 batch + 同维度复用缓存。

    Returns:
        {
            "doc_id_map": {companyRecordId: doc_id, ...},
            "report": GlobalReport,
            "text_results": {pair_key: {tech_similarity, com_similarity, ...}}
            "image_results": {pair_key: {tech_count, com_count, ...}}
            "dimension": "technical" | "commercial"
        }
    """
    dimension = "technical" if item_code in TECH_BID_ITEMS else "commercial"
    # Business_BID_SIMILAR → commercial; TECH_BID_SIMILAR → technical
    cache_key = f"{batch_id}_{dimension}"

    cached = batch_cache.get(cache_key)
    if cached:
        logger.info(f"复用 {dimension} 管线结果 (batch {batch_id})")
        return cached

    from pipeline.orchestrator import BidDetectionOrchestrator

    dirs = _get_batch_work_dirs(batch_id, dimension)
    work_dir = dirs["work_dir"]
    input_dir = dirs["input_dir"]
    output_dir = dirs["output_dir"]
    cache_dir = dirs["cache_dir"]
    ckpt_dir = dirs["ckpt_dir"]
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    doc_id_map = _copy_batch_inputs(batch_id, companies, pdf_paths, input_dir)
    config = _build_pipeline_config(cache_dir, ckpt_dir, dimension)

    logger.info(f"运行全量管线: {len(pdf_paths)} 个文件, dimension={dimension}")
    t0 = time.time()

    orchestrator = BidDetectionOrchestrator(config)

    # 注入全局 SBERT 模型（避免每个任务重复加载 ~500MB 模型）
    if _global_sbert_model is not None:
        try:
            orchestrator.embedding_engine.set_model(_global_sbert_model)
            orchestrator.paragraph_matcher._ensure_semantic_matcher()
            orchestrator.paragraph_matcher.semantic_matcher.set_model(_global_sbert_model)
            logger.info("全局 SBERT 模型已注入 EmbeddingEngine 和 ParagraphMatcher")
        except Exception as e:
            logger.warning(f"SBERT 模型注入失败: {e}")

    report = orchestrator.detect(input_dir, output_dir)

    elapsed = time.time() - t0
    logger.info(f"管线完成: {elapsed:.1f}s, {report.total_pairs} 对, "
                f"{report.suspicious_pairs} 对可疑")

    doc_to_record: Dict[str, int] = {}
    for record_id, doc_id in doc_id_map.items():
        doc_to_record[doc_id] = record_id

    text_results = {}
    image_results = {}
    metadata_groups = report.metadata_groups

    for pair in report.pairwise_results:
        doc_a_id = pair.doc_a_id
        doc_b_id = pair.doc_b_id
        a_rec = doc_to_record.get(doc_a_id)
        b_rec = doc_to_record.get(doc_b_id)
        if not a_rec or not b_rec:
            continue

        pair_key = f"{min(a_rec, b_rec)}_{max(a_rec, b_rec)}"
        te = pair.evidence.text_evidence
        ie = pair.evidence.image_evidence

        tech_matches = []
        com_matches = []
        all_matches = []
        for pm in te.paragraph_matches:
            dim = pm.get("_dimension_tag", pm.get("dimension", "unknown"))
            if dim in ("tech_text", "tech_image"):
                tech_matches.append(pm)
            elif dim in ("com_text", "com_image"):
                com_matches.append(pm)
            else:
                tech_matches.append(pm)
            all_matches.append(pm)

        # 提取段落匹配摘要（限制数量与长度，避免响应过大）
        match_summaries = []
        for pm in all_matches[:20]:
            text_a = (pm.get("paragraph_a", "") or "")[:300]
            text_b = (pm.get("paragraph_b", "") or "")[:300]
            match_summaries.append({
                "paragraph_a": text_a,
                "paragraph_b": text_b,
                "similarity": round(pm.get("similarity", 0.0), 4),
                "paragraph_a_index": pm.get("paragraph_a_index", -1),
                "paragraph_b_index": pm.get("paragraph_b_index", -1),
                "detection_method": pm.get("detection_method", ""),
            })

        text_results[pair_key] = {
            "similarity": te.local_similarity,
            "tech_similarity": len(tech_matches),
            "com_similarity": len(com_matches),
            "has_tech_match": len(tech_matches) > 0,
            "has_com_match": len(com_matches) > 0,
            "pair_ids": {"a": a_rec, "b": b_rec},
            "paragraph_matches": match_summaries,
        }

        tech_img_count = 0
        com_img_count = 0
        image_pair_summaries = []
        for img_pair in ie.matched_image_pairs:
            dim = img_pair.get("_dimension_tag", "unknown")
            if dim in ("tech_image", "tech_image+com_image"):
                tech_img_count += 1
            if dim in ("com_image", "tech_image+com_image"):
                com_img_count += 1
            if len(image_pair_summaries) < 20:
                image_pair_summaries.append({
                    "source_a": img_pair.get("source_a", ""),
                    "source_b": img_pair.get("source_b", ""),
                    "confidence": round(img_pair.get("confidence", 0.0), 4),
                    "reasons": img_pair.get("reasons", []),
                    "ocr_text_a": (img_pair.get("ocr_text_a", "") or "")[:200],
                    "ocr_text_b": (img_pair.get("ocr_text_b", "") or "")[:200],
                })

        image_results[pair_key] = {
            "total_image_matches": ie.common_image_count,
            "tech_image_matches": tech_img_count,
            "com_image_matches": com_img_count,
            "has_tech_image": tech_img_count > 0,
            "has_com_image": com_img_count > 0,
            "pair_ids": {"a": a_rec, "b": b_rec},
            "matched_image_pairs": image_pair_summaries,
        }

    meta_index = {}
    for mg in metadata_groups:
        meta_index.setdefault(mg.group_type, []).append(mg)

    result = {
        "doc_id_map": doc_id_map,
        "doc_to_record": doc_to_record,
        "text_results": text_results,
        "image_results": image_results,
        "meta_index": meta_index,
        "report": report,
        "config": config,
        "dimension": dimension,
    }

    batch_cache.set(cache_key, result)

    if os.environ.get("COLLUSIVE_KEEP_WORKDIR", "1").lower() not in ("1", "true", "yes"):
        def _cleanup():
            time.sleep(43200)  # 12 小时后清理
            if os.path.exists(work_dir):
                shutil.rmtree(work_dir, ignore_errors=True)
        threading.Thread(target=_cleanup, daemon=True).start()

    return result


def _get_company_results_from_pipeline(
    companies: List[CompanyInfo],
    pipeline_result: dict,
    item_code: str,
) -> List[CompanyResult]:
    """从管线结果中提取指定 itemCode 的公司级结果

    管道已通过 ANALYSIS_DIMENSION 单维度过滤，无需再拆 tech/com 子计数。
    直接用 text_results / image_results 的总体匹配判定 FAILED/SUCCESS。

    TECH_BID_SIMILAR:           technical 维度
    Business_BID_SIMILAR:  commercial 维度

    evidence 字段包含详细证据：
    - 文本相似: similarParagraphs（段落内容 + 相似度 + 公司ID）
    - 图片相似: similarImages（图片引用 + 置信度 + 公司ID）
    """
    text_results = pipeline_result.get("text_results", {})
    image_results = pipeline_result.get("image_results", {})
    dimension = pipeline_result.get("dimension", "technical")
    dim_label = "技术标" if dimension == "technical" else "商务标"

    company_evidence: Dict[int, dict] = {}
    for c in companies:
        cid = c.companyRecordId
        company_evidence[cid] = {
            "similar_ids": set(),
            "text_details": [],
            "image_details": [],
        }

    for pair_key, tr in text_results.items():
        a, b = tr["pair_ids"]["a"], tr["pair_ids"]["b"]
        if tr.get("similarity", 0) >= 0.3 or tr.get("has_tech_match") or tr.get("has_com_match"):
            company_evidence[a]["similar_ids"].add(b)
            company_evidence[b]["similar_ids"].add(a)
            # 提取段落匹配详情
            para_matches = tr.get("paragraph_matches", [])
            company_evidence[a]["text_details"].append({
                "companyRecordId": b,
                "similarity": round(tr.get("similarity", 0.0), 4),
                "paragraphMatches": para_matches,
            })
            company_evidence[b]["text_details"].append({
                "companyRecordId": a,
                "similarity": round(tr.get("similarity", 0.0), 4),
                "paragraphMatches": para_matches,
            })

    for pair_key, ir in image_results.items():
        a, b = ir["pair_ids"]["a"], ir["pair_ids"]["b"]
        if ir.get("total_image_matches", 0) > 0 or ir.get("has_tech_image") or ir.get("has_com_image"):
            company_evidence[a]["similar_ids"].add(b)
            company_evidence[b]["similar_ids"].add(a)
            # 提取图片匹配详情
            img_pairs = ir.get("matched_image_pairs", [])
            company_evidence[a]["image_details"].append({
                "companyRecordId": b,
                "imageMatchCount": ir.get("total_image_matches", 0),
                "similarImages": img_pairs,
            })
            company_evidence[b]["image_details"].append({
                "companyRecordId": a,
                "imageMatchCount": ir.get("total_image_matches", 0),
                "similarImages": img_pairs,
            })

    results = []
    for c in companies:
        cid = c.companyRecordId
        ev = company_evidence.get(cid, {"similar_ids": set(), "text_details": [], "image_details": []})
        similar_ids = sorted(ev["similar_ids"])

        if similar_ids:
            similar_count = len(similar_ids)
            evidence = {
                "similarCompanyRecordIds": similar_ids,
                "similarParagraphs": ev["text_details"],
                "similarImages": ev["image_details"],
            }
            results.append(CompanyResult(
                companyRecordId=cid, registrationCompanyId=c.registrationCompanyId,
                sectionId=c.sectionId, status="FAILED",
                summary=f"{dim_label}与 {similar_count} 家公司雷同",
                evidence=evidence,
            ))
        else:
            results.append(CompanyResult(
                companyRecordId=cid, registrationCompanyId=c.registrationCompanyId,
                sectionId=c.sectionId, status="SUCCESS",
                summary=f"未发现{dim_label}雷同", evidence={},
            ))
    return results


# ============================================================
# 同步调度器
# ============================================================

def _dispatch_handler(
    item_code: str,
    companies: List[CompanyInfo],
    pdf_paths: Dict[int, str],
    batch_id: int,
) -> List[CompanyResult]:
    """根据 itemCode 路由到对应处理器

    轻量项（file_id/author/editor/contact/bidderName）：直接 fitz 提取，秒级返回
    重量项（tech/commercial）：走完整管道 + ANALYSIS_DIMENSION 过滤
    """
    if item_code == "FILE_CODE_SIMILAR":
        return handle_file_code_similar(companies, pdf_paths)
    elif item_code == "EDITOR_SIGNER_SIMILAR":
        return handle_editor_signer_similar(companies, pdf_paths)
    elif item_code == "DOC_AUTHOR_SIMILAR":
        return handle_doc_author_similar(companies, pdf_paths)
    elif item_code == "SAME_BID_CONTACT_SIMILAR":
        return handle_same_bid_contact_similar(companies, pdf_paths, batch_id)
    elif item_code == "SAME_bidderName_SIMILAR":
        return handle_bidder_name_similar(companies, pdf_paths, batch_id)
    elif item_code in HEAVY_ITEMS:
        pipeline_result = _run_full_pipeline(
            batch_id, companies, pdf_paths, item_code=item_code,
        )
        return _get_company_results_from_pipeline(
            companies, pipeline_result, item_code,
        )
    else:
        raise ValueError(f"未知的 itemCode: {item_code}")


def _build_error_response(request: AnalyzeRequest, error_msg: str) -> AnalyzeResponse:
    """构建错误响应（全公司 ERROR，不阻断后续检查项）"""
    error_results = [
        CompanyResult(
            companyRecordId=c.companyRecordId,
            registrationCompanyId=c.registrationCompanyId,
            sectionId=c.sectionId,
            status="ERROR",
            summary=f"检查异常: {error_msg}",
            evidence={"error": error_msg},
        ) for c in request.companies
    ]
    return AnalyzeResponse(
        batchId=request.batchId,
        itemCode=request.itemCode,
        itemName=ITEM_CODE_NAMES.get(request.itemCode, request.itemCode),
        results=error_results,
    )


def _run_analysis_sync(request: AnalyzeRequest) -> AnalyzeResponse:
    """同步执行单项检查（在 ThreadPoolExecutor 中运行）

    1. 下载 PDF（按 batchId 缓存到本地）
    2. 路由到对应 handler（轻量秒级 / 重量走完整管道）
    3. 返回 AnalyzeResponse
    """
    batch_id = request.batchId
    item_code = request.itemCode
    companies = request.companies

    t0 = time.time()
    logger.info(f"[batch={batch_id}] 开始检查 {item_code} ({len(companies)} 家公司)")

    # 1. 下载 PDF
    pdf_paths = download_batch_pdfs(batch_id, companies)

    # 2. 路由到对应 handler
    results = _dispatch_handler(item_code, companies, pdf_paths, batch_id)

    # 3. 构建响应
    response = AnalyzeResponse(
        batchId=batch_id,
        itemCode=item_code,
        itemName=ITEM_CODE_NAMES.get(item_code, item_code),
        results=results,
    )

    elapsed = time.time() - t0
    failed_count = sum(1 for r in results if r.status == "FAILED")
    logger.info(f"[batch={batch_id}] 完成 — {item_code}, "
                f"{failed_count}/{len(results)} FAILED, 耗时 {elapsed:.1f}s")

    return response


# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(
    title="围标串标检查 AI 服务",
    description="为 Java 后端提供同步围标串标单项检查接口",
    version="1.0.0",
)

# 同步执行器 + 超时控制
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

SYNC_TIMEOUT = 1800  # 30 分钟，覆盖 OCR+SBERT 重型场景
ANALYZE_EXECUTOR_WORKERS = max(
    1, int(os.environ.get("ANALYZE_EXECUTOR_WORKERS", "4"))
)
_executor = ThreadPoolExecutor(
    max_workers=ANALYZE_EXECUTOR_WORKERS,
    thread_name_prefix="analyze",
)


@app.on_event("startup")
async def startup_event():
    """启动时确保目录存在 + 预加载模型"""
    os.makedirs(DOWNLOAD_BASE, exist_ok=True)
    os.makedirs(WORK_DIR, exist_ok=True)
    logger.info("围标串标 API 服务启动")
    logger.info(f"  PDF 下载目录: {DOWNLOAD_BASE}")
    logger.info(f"  管线工作目录: {WORK_DIR}")
    logger.info(f"  API 分析线程: {ANALYZE_EXECUTOR_WORKERS}")

    # 预加载模型（避免首次请求卡住）
    print("=" * 60)
    print("正在预加载模型（首次启动需要 20-60 秒）...")
    print("=" * 60)
    logger.info("开始预加载模型...")

    # 1. SBERT 语义模型
    print("\n[1/2] 加载 SBERT 语义模型...")
    logger.info("[1/2] 加载 SBERT 语义模型（优先本地缓存）...")
    try:
        from sentence_transformers import SentenceTransformer
        global _global_sbert_model
        sbert_device = os.environ.get("SBERT_DEVICE", "cpu").lower()
        if sbert_device == "auto":
            try:
                import torch
                sbert_device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                sbert_device = "cpu"
        _global_sbert_model = SentenceTransformer(
            'paraphrase-multilingual-MiniLM-L12-v2',
            device=sbert_device, cache_folder='./models',
            trust_remote_code=True, local_files_only=True,
        )
        print(f"  SBERT 加载完成 ({sbert_device})")
        logger.info(f"  SBERT 离线加载完成 ({sbert_device})")
    except Exception as e:
        print(f"  SBERT 本地加载失败: {e}")
        print("  SBERT 将在首次请求时按需加载")
        logger.warning(f"  SBERT 本地加载失败: {e}")

    # 2. OCR 引擎
    print("\n[2/2] 加载 OCR 引擎...")
    logger.info("[2/2] 加载 OCR 引擎...")
    try:
        from config import DetectionConfig
        from image_analysis.image_ocr import ImageOCREngine
        _cfg = DetectionConfig()
        print(f"  正在初始化 {_cfg.OCR_ENGINE}...")
        engine = ImageOCREngine(
            use_gpu=_cfg.USE_GPU,
            engine=_cfg.OCR_ENGINE,
            offline=_cfg.OCR_OFFLINE_MODE,
        )
        ok = engine.is_available
        print(f"  OCR ({_cfg.OCR_ENGINE}) {'加载完成' if ok else '不可用'}")
        logger.info(f"  OCR ({_cfg.OCR_ENGINE}) {'加载完成' if ok else '不可用'}")
    except Exception as e:
        print(f"  OCR 加载异常: {e}")
        logger.warning(f"  OCR 加载异常: {e}")

    print("\n" + "=" * 60)
    print("模型预加载完成，服务就绪")
    print("=" * 60 + "\n")
    logger.info("模型预加载完成，服务就绪")


@app.post("/api/v1/collusive-check/items/analyze", status_code=200)
async def analyze_item(request: AnalyzeRequest) -> AnalyzeResponse:
    """同步单项检查

    Java 后端按 itemCode 顺序循环调用，每次只传一个检查项。
    同步返回该 itemCode 下每家公司的检查结果。

    轻量项（file_id/author/editor/contact/bidderName）秒级返回；
    重量项（tech/commercial）走完整管道，最长 30 分钟。
    单项失败或超时返回全公司 ERROR，不阻断后续检查项。
    """
    item_code = request.itemCode
    if item_code not in ITEM_CODE_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"未知 itemCode: {item_code}，支持: {list(ITEM_CODE_NAMES.keys())}"
        )

    if not request.companies:
        raise HTTPException(status_code=400, detail="companies 不能为空")
    if len(request.companies) < 2 and item_code in HEAVY_ITEMS:
        raise HTTPException(status_code=400, detail="此检查项至少需要 2 家公司")

    future = _executor.submit(_run_analysis_sync, request)
    try:
        return await asyncio.wait_for(asyncio.wrap_future(future), timeout=SYNC_TIMEOUT)
    except asyncio.TimeoutError:
        logger.error(f"[batch={request.batchId}] {item_code} 检测超时（{SYNC_TIMEOUT}s）")
        return _build_error_response(request, f"检测超时（{SYNC_TIMEOUT}秒）")
    except Exception as e:
        logger.exception(f"[batch={request.batchId}] {item_code} 分析失败: {e}")
        return _build_error_response(request, str(e))


@app.get("/api/v1/collusive-check/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "supported_items": len(ITEM_CODE_NAMES),
    }


@app.get("/api/v1/collusive-check/item-codes")
async def list_item_codes():
    """列出所有支持的检查项"""
    return {
        "items": [
            {"code": k, "name": v} for k, v in ITEM_CODE_NAMES.items()
        ]
    }


# ============================================================
# 启动
# ============================================================

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    host = os.environ.get("COLLUSIVE_HOST", "0.0.0.0")
    port = int(os.environ.get("COLLUSIVE_PORT", "8001"))

    print(f"[API] 围标串标检查服务启动: http://{host}:{port}")
    print(f"[API] PDF 下载目录: {DOWNLOAD_BASE}")
    print(f"[API] API 文档: http://{host}:{port}/docs")
    print(f"[API] 支持 {len(ITEM_CODE_NAMES)} 个检查项")

    uvicorn.run(app, host=host, port=port, log_level="info")
