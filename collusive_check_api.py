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

# ============================================================
# itemCode → 检查项名称映射
# ============================================================

ITEM_CODE_NAMES: Dict[str, str] = {
    "FILE_CODE_SIMILAR": "文件码雷同",
    "EDITOR_SIGNER_SIMILAR": "编辑经办人雷同",
    "DOC_AUTHOR_SIMILAR": "文档作者雷同",
    "BID_COMPANY_NAME_ABNORMAL": "投标文件公司名称异常",
    "SAME_BID_CONTACT_SIMILAR": "同标段单位联系人雷同",
    "TECH_BID_SIMILAR": "技术标雷同",
    "COM_BID_SIMILAR": "商务标雷同",
}

# 轻量检查（无需全量管线）
LIGHTWEIGHT_ITEMS = {
    "FILE_CODE_SIMILAR", "EDITOR_SIGNER_SIMILAR", "DOC_AUTHOR_SIMILAR",
    "BID_COMPANY_NAME_ABNORMAL", "SAME_BID_CONTACT_SIMILAR",
}

# 重量检查（技术标 + 商务标，各包含文本+图片）
TECH_BID_ITEMS = {"TECH_BID_SIMILAR"}
COM_BID_ITEMS = {"COM_BID_SIMILAR"}
HEAVY_ITEMS = TECH_BID_ITEMS | COM_BID_ITEMS

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

class TaskStatus(BaseModel):
    taskId: str
    batchId: int
    itemCode: str
    status: str          # pending / processing / completed / failed
    result: Optional[AnalyzeResponse] = None
    error: Optional[str] = None
    created_at: str = ""
    completed_at: str = ""

# ============================================================
# 任务管理器
# ============================================================

class TaskRecord:
    def __init__(self, task_id: str, batch_id: int, item_code: str, companies: List[CompanyInfo]):
        self.task_id = task_id
        self.batch_id = batch_id
        self.item_code = item_code
        self.companies = companies
        self.status = "pending"
        self.result: Optional[AnalyzeResponse] = None
        self.error: Optional[str] = None
        self.created_at = datetime.now().isoformat()
        self.completed_at = ""
        self._lock = threading.Lock()

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "taskId": self.task_id,
                "batchId": self.batch_id,
                "itemCode": self.item_code,
                "status": self.status,
                "result": self.result.model_dump() if self.result else None,
                "error": self.error,
                "created_at": self.created_at,
                "completed_at": self.completed_at,
            }


class TaskManager:
    """异步任务管理器（内存存储，进程内有效）"""

    def __init__(self):
        self._tasks: Dict[str, TaskRecord] = {}
        self._lock = threading.Lock()
        # 批次级检测结果缓存（heavy items 复用）
        self._batch_cache: Dict[str, dict] = {}
        self._batch_cache_time: Dict[str, float] = {}

    def create_task(self, batch_id: int, item_code: str, companies: List[CompanyInfo]) -> str:
        task_id = str(uuid.uuid4())
        record = TaskRecord(task_id, batch_id, item_code, companies)
        with self._lock:
            self._tasks[task_id] = record
        return task_id

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        with self._lock:
            return self._tasks.get(task_id)

    def update_task(self, task_id: str, **kwargs):
        with self._lock:
            record = self._tasks.get(task_id)
            if record:
                for k, v in kwargs.items():
                    setattr(record, k, v)

    def get_batch_cache(self, key) -> Optional[dict]:
        with self._lock:
            cached = self._batch_cache.get(key)
            ts = self._batch_cache_time.get(key, 0)
            if cached and (time.time() - ts) < CACHE_TTL:
                return cached
        return None

    def set_batch_cache(self, key, data: dict):
        with self._lock:
            self._batch_cache[key] = data
            self._batch_cache_time[key] = time.time()

    def invalidate_batch_cache(self, key):
        with self._lock:
            self._batch_cache.pop(key, None)
            self._batch_cache_time.pop(key, None)


task_manager = TaskManager()

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

    result = {}
    errors = []

    for company in companies:
        record_id = company.companyRecordId
        filename = f"{record_id}_{_sanitize_filename(company.bidderName)}.pdf"
        local_path = os.path.join(batch_dir, filename)

        # 如果已存在且文件有效，跳过下载
        if os.path.exists(local_path) and os.path.getsize(local_path) > 100:
            result[record_id] = local_path
            continue

        # 下载（带重试）
        downloaded = False
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
                    result[record_id] = local_path
                    logger.info(f"  ✅ 下载完成: {local_path} ({len(resp.content) / 1024:.0f}KB)")
                    downloaded = True
                    break
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
                        result[record_id] = local_path
                        logger.info(f"  ✅ 下载完成（跳过 SSL）")
                        downloaded = True
                        break
                except Exception as e2:
                    logger.warning(f"  SSL 跳过后仍失败: {e2}")
            except requests.exceptions.RequestException as e:
                logger.warning(f"  请求失败 (attempt {attempt}): {e}")

            if attempt < DOWNLOAD_RETRIES:
                time.sleep(2 * attempt)

        if not downloaded:
            err_msg = f"公司 {company.bidderName}({record_id}) PDF 下载失败"
            errors.append(err_msg)
            logger.error(err_msg)

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


def handle_company_name_abnormal(
    companies: List[CompanyInfo],
    pdf_paths: Dict[int, str],
) -> List[CompanyResult]:
    """投标文件公司名称异常 — 检测本公司文件中出现其他投标公司名称"""
    from extraction.contact_extractor import extract_contacts_from_text

    company_name_in_docs: Dict[int, List[str]] = {}
    for c in companies:
        path = pdf_paths.get(c.companyRecordId)
        if path and os.path.exists(path):
            text = _extract_text_preview(path)
            ci = extract_contacts_from_text(text)
            company_name_in_docs[c.companyRecordId] = ci.company_names

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
                summary="未发现公司名称异常", evidence={},
            ))
    return results


def handle_same_bid_contact_similar(
    companies: List[CompanyInfo],
    pdf_paths: Dict[int, str],
) -> List[CompanyResult]:
    """同标段单位联系人雷同 — 检查人名、手机号、邮箱跨公司雷同"""
    from extraction.contact_extractor import extract_contacts_from_text

    contacts = {}
    for c in companies:
        path = pdf_paths.get(c.companyRecordId)
        if path and os.path.exists(path):
            text = _extract_text_preview(path)
            contacts[c.companyRecordId] = extract_contacts_from_text(text)

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
    cache_key = f"{batch_id}_{dimension}"

    cached = task_manager.get_batch_cache(cache_key)
    if cached:
        logger.info(f"复用 {dimension} 管线结果 (batch {batch_id})")
        return cached

    from config import DetectionConfig
    from pipeline.orchestrator import BidDetectionOrchestrator

    work_dir = os.path.join(WORK_DIR, f"batch_{batch_id}_{uuid.uuid4().hex[:8]}")
    input_dir = os.path.join(work_dir, "input")
    output_dir = os.path.join(work_dir, "output")
    cache_dir = os.path.join(work_dir, "cache")
    ckpt_dir = os.path.join(work_dir, "checkpoints")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    doc_id_map = {}
    for c in companies:
        src = pdf_paths.get(c.companyRecordId)
        if src and os.path.exists(src):
            safe_name = f"{c.companyRecordId}_{_sanitize_filename(c.bidderName)}.pdf"
            dst = os.path.join(input_dir, safe_name)
            shutil.copy2(src, dst)
            doc_id_map[c.companyRecordId] = safe_name

    config = DetectionConfig()
    config.CACHE_DIR = cache_dir
    config.CHECKPOINT_DIR = ckpt_dir
    config.ENABLE_OCR = True
    config.OCR_ENGINE = "paddleocr"
    config.USE_GPU = False
    config.SBERT_DEVICE = "cpu"
    config.TOC_FILTER_ENABLED = True
    config.ANALYSIS_DIMENSION = dimension

    logger.info(f"运行全量管线: {len(pdf_paths)} 个文件")
    t0 = time.time()

    orchestrator = BidDetectionOrchestrator(config)
    report = orchestrator.detect(input_dir, output_dir)

    elapsed = time.time() - t0
    logger.info(f"管线完成: {elapsed:.1f}s, {report.total_pairs} 对, "
                f"{report.suspicious_pairs} 对可疑")

    doc_to_record: Dict[str, int] = {}
    for record_id, doc_name in doc_id_map.items():
        doc_to_record[doc_name] = record_id

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
        for pm in te.paragraph_matches:
            dim = pm.get("_dimension_tag", pm.get("dimension", "unknown"))
            if dim in ("tech_text", "tech_image"):
                tech_matches.append(pm)
            elif dim in ("com_text", "com_image"):
                com_matches.append(pm)
            else:
                tech_matches.append(pm)

        text_results[pair_key] = {
            "similarity": te.local_similarity,
            "tech_similarity": len(tech_matches),
            "com_similarity": len(com_matches),
            "has_tech_match": len(tech_matches) > 0,
            "has_com_match": len(com_matches) > 0,
            "pair_ids": {"a": a_rec, "b": b_rec},
        }

        tech_img_count = 0
        com_img_count = 0
        for img_pair in ie.matched_image_pairs:
            dim = img_pair.get("_dimension_tag", "unknown")
            if dim in ("tech_image", "tech_image+com_image"):
                tech_img_count += 1
            if dim in ("com_image", "tech_image+com_image"):
                com_img_count += 1

        image_results[pair_key] = {
            "total_image_matches": ie.common_image_count,
            "tech_image_matches": tech_img_count,
            "com_image_matches": com_img_count,
            "has_tech_image": tech_img_count > 0,
            "has_com_image": com_img_count > 0,
            "pair_ids": {"a": a_rec, "b": b_rec},
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

    task_manager.set_batch_cache(cache_key, result)

    def _cleanup():
        time.sleep(300)
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

    TECH_BID_SIMILAR: 检查 tech 维度的文本 + 图片匹配
    COM_BID_SIMILAR:  检查 com 维度的文本 + 图片匹配
    """
    text_results = pipeline_result.get("text_results", {})
    image_results = pipeline_result.get("image_results", {})
    dimension = pipeline_result.get("dimension", "technical")

    is_tech = dimension == "technical"
    is_com = dimension == "commercial"
    dim_label = "技术标" if is_tech else "商务标"

    company_evidence: Dict[int, dict] = {}
    for c in companies:
        cid = c.companyRecordId
        company_evidence[cid] = {"similar_ids": set(), "details": []}

    for pair_key, tr in text_results.items():
        a, b = tr["pair_ids"]["a"], tr["pair_ids"]["b"]
        if (is_tech and tr.get("has_tech_match")) or (is_com and tr.get("has_com_match")):
            company_evidence[a]["similar_ids"].add(b)
            company_evidence[b]["similar_ids"].add(a)
            company_evidence[a]["details"].append({
                "type": "text", "similarity": tr["similarity"],
                "companyRecordId": b,
            })
            company_evidence[b]["details"].append({
                "type": "text", "similarity": tr["similarity"],
                "companyRecordId": a,
            })

    for pair_key, ir in image_results.items():
        a, b = ir["pair_ids"]["a"], ir["pair_ids"]["b"]
        if (is_tech and ir.get("has_tech_image")) or (is_com and ir.get("has_com_image")):
            company_evidence[a]["similar_ids"].add(b)
            company_evidence[b]["similar_ids"].add(a)
            company_evidence[a]["details"].append({
                "type": "image", "imageMatches": ir.get("total_image_matches", 0),
                "companyRecordId": b,
            })
            company_evidence[b]["details"].append({
                "type": "image", "imageMatches": ir.get("total_image_matches", 0),
                "companyRecordId": a,
            })

    results = []
    for c in companies:
        cid = c.companyRecordId
        ev = company_evidence.get(cid, {"similar_ids": set(), "details": []})
        similar_ids = list(ev["similar_ids"])

        if similar_ids:
            similar_count = len(similar_ids)
            results.append(CompanyResult(
                companyRecordId=cid, registrationCompanyId=c.registrationCompanyId,
                sectionId=c.sectionId, status="FAILED",
                summary=f"{dim_label}与 {similar_count} 家公司雷同",
                evidence={"detail": ev["details"], "similarCompanyRecordIds": similar_ids},
            ))
        else:
            results.append(CompanyResult(
                companyRecordId=cid, registrationCompanyId=c.registrationCompanyId,
                sectionId=c.sectionId, status="SUCCESS",
                summary=f"未发现{dim_label}雷同", evidence={},
            ))
    return results


# ============================================================
# 主调度器
# ============================================================

def _run_analysis(task_id: str):
    """后台运行分析"""
    record = task_manager.get_task(task_id)
    if not record:
        return

    try:
        task_manager.update_task(task_id, status="processing")
        batch_id = record.batch_id
        item_code = record.item_code
        companies = record.companies

        # 1. 下载 PDF
        logger.info(f"[{task_id[:8]}] 下载 PDF (batch={batch_id}, {len(companies)} 个文件)")
        pdf_paths = download_batch_pdfs(batch_id, companies)

        # 2. 路由到对应 handler
        if item_code == "FILE_CODE_SIMILAR":
            results = handle_file_code_similar(companies, pdf_paths)
        elif item_code == "EDITOR_SIGNER_SIMILAR":
            results = handle_editor_signer_similar(companies, pdf_paths)
        elif item_code == "DOC_AUTHOR_SIMILAR":
            results = handle_doc_author_similar(companies, pdf_paths)
        elif item_code == "BID_COMPANY_NAME_ABNORMAL":
            results = handle_company_name_abnormal(companies, pdf_paths)
        elif item_code == "SAME_BID_CONTACT_SIMILAR":
            results = handle_same_bid_contact_similar(companies, pdf_paths)
        elif item_code in HEAVY_ITEMS:
            pipeline_result = _run_full_pipeline(
                batch_id, companies, pdf_paths, item_code=item_code,
            )
            results = _get_company_results_from_pipeline(
                companies, pipeline_result, item_code,
            )
        else:
            raise ValueError(f"未知的 itemCode: {item_code}")

        # 3. 构建响应
        response = AnalyzeResponse(
            batchId=batch_id,
            itemCode=item_code,
            itemName=ITEM_CODE_NAMES.get(item_code, item_code),
            results=results,
        )

        task_manager.update_task(
            task_id, status="completed", result=response,
            completed_at=datetime.now().isoformat(),
        )
        logger.info(f"[{task_id[:8]}] 完成 — {item_code}, "
                    f"{sum(1 for r in results if r.status == 'FAILED')}/{len(results)} FAILED")

    except Exception as e:
        logger.exception(f"[{task_id[:8]}] 分析失败: {e}")
        record = task_manager.get_task(task_id)
        if record:
            error_results = []
            for c in record.companies:
                error_results.append(CompanyResult(
                    companyRecordId=c.companyRecordId,
                    registrationCompanyId=c.registrationCompanyId,
                    sectionId=c.sectionId, status="ERROR",
                    summary=f"检查异常: {str(e)}",
                    evidence={"error": str(e)},
                ))
            error_response = AnalyzeResponse(
                batchId=record.batch_id,
                itemCode=record.item_code,
                itemName=ITEM_CODE_NAMES.get(record.item_code, record.item_code),
                results=error_results,
            )
            task_manager.update_task(
                task_id, status="failed", result=error_response,
                error=str(e), completed_at=datetime.now().isoformat(),
            )


# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(
    title="围标串标检查 AI 服务",
    description="为 Java 后端提供异步围标串标单项检查接口",
    version="1.0.0",
)


@app.on_event("startup")
async def startup_event():
    """启动时确保目录存在"""
    os.makedirs(DOWNLOAD_BASE, exist_ok=True)
    os.makedirs(WORK_DIR, exist_ok=True)
    logger.info("围标串标 API 服务启动")
    logger.info(f"  PDF 下载目录: {DOWNLOAD_BASE}")
    logger.info(f"  管线工作目录: {WORK_DIR}")


@app.post("/api/v1/collusive-check/items/analyze", status_code=202)
async def analyze_item(request: AnalyzeRequest):
    """提交单项检查任务（异步）

    Java 后端按 itemCode 顺序循环调用，每次只传一个检查项。
    重量检查（文本/图片）同 batch 复用全量管线结果。
    """
    item_code = request.itemCode
    if item_code not in ITEM_CODE_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"未知 itemCode: {item_code}，支持: {list(ITEM_CODE_NAMES.keys())}"
        )

    if not request.companies or len(request.companies) < 1:
        raise HTTPException(status_code=400, detail="companies 不能为空")
    if len(request.companies) < 2 and item_code in HEAVY_ITEMS:
        raise HTTPException(status_code=400, detail="此检查项至少需要 2 家公司")

    task_id = task_manager.create_task(
        request.batchId, item_code, request.companies,
    )

    thread = threading.Thread(target=_run_analysis, args=(task_id,), daemon=True)
    thread.start()

    return JSONResponse(
        status_code=202,
        content={
            "taskId": task_id,
            "batchId": request.batchId,
            "itemCode": item_code,
            "status": "pending",
            "message": f"检查任务已提交 ({ITEM_CODE_NAMES[item_code]})",
        },
    )


@app.get("/api/v1/collusive-check/items/{task_id}")
async def get_analyze_result(task_id: str):
    """轮询检查结果"""
    record = task_manager.get_task(task_id)
    if not record:
        raise HTTPException(status_code=404, detail="任务不存在")
    return record.to_dict()


@app.get("/api/v1/collusive-check/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "active_tasks": sum(
            1 for t in task_manager._tasks.values()
            if t.status in ("pending", "processing")
        ),
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
