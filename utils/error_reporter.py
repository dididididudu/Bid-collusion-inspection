"""Structured error reporting for API and pipeline failures."""

import json
import os
import traceback
from datetime import datetime
from typing import Any, Dict, Optional


def _safe_json_value(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except Exception:
        return str(value)


def record_error(
    *,
    work_dir: str,
    batch_id: int,
    stage: str,
    error: BaseException | str,
    item_code: str = "",
    company_record_id: Optional[int] = None,
    registration_company_id: Optional[int] = None,
    section_id: Optional[int] = None,
    doc_id: str = "",
    filename: str = "",
    pair_id: str = "",
    trace_id: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Append one structured error record to batch error_report.jsonl.

    Returns the record so callers can also include selected fields in API
    evidence.
    """
    if isinstance(error, BaseException):
        error_type = type(error).__name__
        message = str(error)
        tb = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    else:
        error_type = "Error"
        message = str(error)
        tb = ""

    record = {
        "time": datetime.now().isoformat(),
        "batchId": batch_id,
        "itemCode": item_code,
        "stage": stage,
        "errorType": error_type,
        "message": message,
        "traceId": trace_id,
    }
    if company_record_id is not None:
        record["companyRecordId"] = company_record_id
    if registration_company_id is not None:
        record["registrationCompanyId"] = registration_company_id
    if section_id is not None:
        record["sectionId"] = section_id
    if doc_id:
        record["docId"] = doc_id
    if filename:
        record["filename"] = filename
    if pair_id:
        record["pairId"] = pair_id
    if tb:
        record["traceback"] = tb
    if extra:
        record["extra"] = {k: _safe_json_value(v) for k, v in extra.items()}

    output_dir = os.path.join(work_dir, f"batch_{batch_id}", "output")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "error_report.jsonl")
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # Never let error reporting become the primary failure.
        pass
    return record
