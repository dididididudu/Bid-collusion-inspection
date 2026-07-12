# 围标串标检查 API 接口合规改造方案

## Context（背景）

接口文档 `bid-evaluation-collusive-check-ai-curl.md` 规定 Java 后端以**同步逐项调用**方式调用 py/AI：每次只传一个 `itemCode` + 一批公司（含 `bidFileUrl`），py/AI 同步返回该单项下每家公司的检测结果。

当前代码已存在 `collusive_check_api.py` 实现了大部分功能（URL 下载、轻量/重量分发、pairwise→per-company 转换），但存在三个核心偏差：

1. **异步而非同步**：端点返回 `202 + taskId`，需客户端轮询 `GET /items/{taskId}`
2. **itemCode 映射错误**：
   - `Business_BID_SIMILAR` 被当作"公司名称异常"，但文档中它表示"商务标雷同"（重量维度）
   - 代码有 `COM_BID_SIMILAR`（商务标雷同），文档中无此项
   - 文档有 `SAME_bidderName_SIMILAR`（公司名雷同），代码缺失
3. **缺少模型预加载**：首次请求时加载 SBERT/OCR 会阻塞数分钟

用户决策：
- 同步接口 + 固定超时 **1800 秒（30 分钟）**（因 OCR+SBERT 耗时长）
- **删除** `/api/detect` 端点及 `api_server.py` 全部相关代码
- Java 后端顺序调用各维度，每个维度返回后再调下一个

## 改造范围

### 文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `collusive_check_api.py` | 修改 | 核心改造：同步化 + itemCode 映射修正 + 模型预加载 |
| `api_server.py` | 删除 | 移除 /api/detect 及其全部支撑代码 |
| `test_api.py` | 重写 | 适配新的同步接口契约 |

---

## 详细改造步骤

### 步骤 1：修正 itemCode 映射表（collusive_check_api.py L70-89）

**目标**：与文档第 5.2 节的 7 个 itemCode 完全对齐。

**修改 `ITEM_CODE_NAMES`**：
```python
ITEM_CODE_NAMES: Dict[str, str] = {
    "FILE_CODE_SIMILAR": "文件码雷同",
    "EDITOR_SIGNER_SIMILAR": "编辑经办人雷同",
    "DOC_AUTHOR_SIMILAR": "文档作者雷同",
    "SAME_BID_CONTACT_SIMILAR": "人名雷同",
    "SAME_bidderName_SIMILAR": "公司名雷同",      # 新增
    "TECH_BID_SIMILAR": "技术标雷同",
    "Business_BID_SIMILAR": "商务标雷同",     # 改名（原"公司名称异常"）
}
```

**修改分类集合**：
```python
LIGHTWEIGHT_ITEMS = {
    "FILE_CODE_SIMILAR", "EDITOR_SIGNER_SIMILAR", "DOC_AUTHOR_SIMILAR",
    "SAME_BID_CONTACT_SIMILAR", "SAME_bidderName_SIMILAR",
}
TECH_BID_ITEMS = {"TECH_BID_SIMILAR"}
COMMERCIAL_BID_ITEMS = {"Business_BID_SIMILAR"}   # 改名（原 COM_BID_ITEMS）
HEAVY_ITEMS = TECH_BID_ITEMS | COMMERCIAL_BID_ITEMS
```

### 步骤 2：重命名处理器 + 新增公司名雷同处理器（L545-595）

将 `handle_company_name_abnormal` 重命名为 `handle_bidder_name_similar`，逻辑保持不变（检测本公司文件中出现其他投标公司名称），对应 `SAME_bidderName_SIMILAR`：

```python
def handle_bidder_name_similar(
    companies: List[CompanyInfo],
    pdf_paths: Dict[int, str],
) -> List[CompanyResult]:
    """公司名雷同检测 — 本公司文件中出现其他投标公司名称"""
    from extraction.contact_extractor import extract_contacts_from_text
    # ... 复用原 handle_company_name_abnormal 的逻辑 ...
```

### 步骤 3：更新 `_run_full_pipeline` 维度映射（L727）

```python
dimension = "technical" if item_code in TECH_BID_ITEMS else "commercial"
```

### 步骤 4：更新 `_run_analysis` 路由表（L954-972）

```python
if item_code == "FILE_CODE_SIMILAR":
    results = handle_file_code_similar(companies, pdf_paths)
elif item_code == "EDITOR_SIGNER_SIMILAR":
    results = handle_editor_signer_similar(companies, pdf_paths)
elif item_code == "DOC_AUTHOR_SIMILAR":
    results = handle_doc_author_similar(companies, pdf_paths)
elif item_code == "SAME_BID_CONTACT_SIMILAR":
    results = handle_same_bid_contact_similar(companies, pdf_paths)
elif item_code == "SAME_bidderName_SIMILAR":           # 新增
    results = handle_bidder_name_similar(companies, pdf_paths)
elif item_code in HEAVY_ITEMS:
    pipeline_result = _run_full_pipeline(batch_id, companies, pdf_paths, item_code=item_code)
    results = _get_company_results_from_pipeline(companies, pipeline_result, item_code)
```

### 步骤 5：简化 `_get_company_results_from_pipeline`（L859-930）

由于管道已通过 `ANALYSIS_DIMENSION` 单维度过滤，不再需要 tech/com 子计数拆分。简化为直接用 `text_evidence.local_similarity` 和 `match_count` 判定：

```python
def _get_company_results_from_pipeline(...):
    # 直接遍历 pairwise_results，不再拆 tech/com 子计数
    for pair in report.pairwise_results:
        te = pair.evidence.text_evidence
        ie = pair.evidence.image_evidence
        has_match = (te.local_similarity >= 0.3
                     or te.paragraph_matches
                     or ie.common_image_count > 0)
        if has_match:
            # 标记双向 FAILED
```

### 步骤 6：端点同步化（L1035-1070）—— 核心改造

**新增常量与执行器**：
```python
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

SYNC_TIMEOUT = 1800  # 30 分钟，覆盖 OCR+SBERT 重型场景
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="analyze")
```

**改造 `analyze_item` 端点**：
```python
@app.post("/api/v1/collusive-check/items/analyze", status_code=200)
async def analyze_item(request: AnalyzeRequest) -> AnalyzeResponse:
    # 参数校验（同原逻辑）
    item_code = request.itemCode
    if item_code not in ITEM_CODE_NAMES:
        raise HTTPException(400, f"未知 itemCode: {item_code}")
    if not request.companies:
        raise HTTPException(400, "companies 不能为空")
    if len(request.companies) < 2 and item_code in HEAVY_ITEMS:
        raise HTTPException(400, "此检查项至少需要 2 家公司")

    future = _executor.submit(_run_analysis_sync, request)
    try:
        return future.result(timeout=SYNC_TIMEOUT)
    except FutureTimeout:
        # 超时：返回全公司 ERROR，不阻断后续检查项
        return _build_error_response(request, "检测超时（30分钟）")
    except Exception as e:
        logger.exception(f"分析失败: {e}")
        return _build_error_response(request, str(e))
```

**新增 `_run_analysis_sync`**（原 `_run_analysis` 的同步版，去掉 task_manager 状态更新，直接返回 `AnalyzeResponse`）：
```python
def _run_analysis_sync(request: AnalyzeRequest) -> AnalyzeResponse:
    batch_id = request.batchId
    item_code = request.itemCode
    companies = request.companies

    pdf_paths = download_batch_pdfs(batch_id, companies)

    # 路由到对应 handler（同步骤 4 路由表）
    results = _dispatch_handler(item_code, companies, pdf_paths, batch_id)

    return AnalyzeResponse(
        batchId=batch_id,
        itemCode=item_code,
        itemName=ITEM_CODE_NAMES.get(item_code, item_code),
        results=results,
    )
```

**新增 `_build_error_response`**：
```python
def _build_error_response(request: AnalyzeRequest, error_msg: str) -> AnalyzeResponse:
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
```

**移除异步任务管理**：删除 `TaskRecord`、`TaskManager`、`task_manager`、`_run_analysis`（旧异步版）、`GET /items/{task_id}` 端点。但保留 `task_manager._batch_cache` 的批次缓存功能（heavy items 复用），将其提取为独立 `BatchCache` 类。

### 步骤 7：新增模型预加载（startup 事件）

从 `api_server.py` 的 `preload_models` 迁移逻辑：
```python
@app.on_event("startup")
async def preload_models():
    print("=" * 60)
    print("正在预加载模型（首次启动需要 20-60 秒）...")
    # 1. SBERT 模型
    # 2. OCR 引擎
    print("模型预加载完成，服务就绪")
```

### 步骤 8：删除 api_server.py

完全删除该文件。其有用的元素（模型预加载、日志配置）已迁移到 `collusive_check_api.py`。

### 步骤 9：重写 test_api.py

适配新的同步接口：
- 测试 7 个 itemCode 的正常调用
- 测试参数校验（未知 itemCode、空 companies、heavy item 不足 2 家）
- 测试异常处理（无效 URL、超时模拟）
- 测试结果结构（status/summary/evidence 字段完整性）

---

## 关键设计说明

### 维度隔离已正确实现

经核实，`collusive_check_api.py` 的维度隔离逻辑正确：
- **轻量维度**（file_id/author/editor/contact/bidderName）：直接用 `fitz` 提取元数据或文本预览，完全跳过 orchestrator/OCR/SBERT
- **重量维度**（tech/commercial）：跑完整管道，通过 `config.ANALYSIS_DIMENSION` 过滤段落和图片
- **批次缓存**：同 batch + 同维度复用管线结果（`cache_key = f"{batch_id}_{dimension}"`）

无需修改 `pipeline/orchestrator.py`。

### 同步执行的并发安全

- `ThreadPoolExecutor(max_workers=2)` 控制并发，与原 `MAX_CONCURRENT_TASKS=2` 一致
- SBERT 全局模型只读，线程安全
- `_run_full_pipeline` 每次创建独立 work_dir 和 config，任务间隔离
- `BatchCache` 用 `threading.Lock` 保护

### Java 后端顺序调用的兼容性

文档要求"每个维度返回后再调下一个"。同步接口天然满足此要求：
- Java 调 itemCode A → 阻塞等待 → 收到响应 → 调 itemCode B
- 重量维度间通过 `BatchCache` 复用已下载的 PDF（`batch_downloads/{batchId}/`）
- 但 tech 和 commercial 是不同 `ANALYSIS_DIMENSION`，管线结果不共享缓存（正确行为，因段落过滤不同）

---

## 测试计划

### 测试 1：服务启动验证
```bash
python collusive_check_api.py
# 验证：模型预加载日志、端口 8001 监听、/docs 可访问
```

### 测试 2：健康检查 + itemCode 列表
```bash
curl http://127.0.0.1:8001/api/v1/collusive-check/health
curl http://127.0.0.1:8001/api/v1/collusive-check/item-codes
# 验证：返回 7 个 itemCode，无 COM_BID_SIMILAR，有 SAME_bidderName_SIMILAR
```

### 测试 3：轻量维度同步返回
```bash
curl -X POST http://127.0.0.1:8001/api/v1/collusive-check/items/analyze \
  -H "Content-Type: application/json" \
  -d '{"batchId":1,"projectId":1,"checkMode":"SAME_SECTION",
       "itemCode":"DOC_AUTHOR_SIMILAR",
       "companies":[...]}'
# 验证：HTTP 200，直接返回 AnalyzeResponse（非 202+taskId）
# 验证：results 数组长度=公司数，每项有 status/summary/evidence
```

### 测试 4：重量维度同步返回
```bash
# itemCode=TECH_BID_SIMILAR，上传 2 个测试 PDF
# 验证：HTTP 200，30 分钟内返回，results 含技术标雷同判定
```

### 测试 5：异常处理
- 未知 itemCode → HTTP 400
- 空 companies → HTTP 400
- 无效 bidFileUrl → 该公司 ERROR，其余正常
- 全部 URL 无效 → 全公司 ERROR

### 测试 6：test_api.py 自动化
```bash
python test_api.py --module all
# 验证：所有测试用例通过
```

---

## 验收标准

1. **接口契约**：`POST /api/v1/collusive-check/items/analyze` 同步返回 `AnalyzeResponse`（HTTP 200），不再返回 `202 + taskId`
2. **itemCode 覆盖**：支持文档定义的全部 7 个 itemCode，映射关系正确
3. **维度隔离**：轻量维度不触发 OCR/SBERT/orchestrator，秒级返回；重量维度走完整管道
4. **结果结构**：每家公司返回 `companyRecordId/registrationCompanyId/sectionId/status/summary/evidence`，status ∈ {SUCCESS, FAILED, ERROR}
5. **异常隔离**：单项失败标 ERROR 不阻断；超时 30 分钟返回全公司 ERROR
6. **旧代码清理**：`api_server.py` 已删除，无遗留引用
7. **测试通过**：`test_api.py` 全部用例通过

## 关键文件路径

- `c:\dongyuhang\project\Bid collusion inspection\collusive_check_api.py`（主改造文件）
- `c:\dongyuhang\project\Bid collusion inspection\api_server.py`（待删除）
- `c:\dongyuhang\project\Bid collusion inspection\test_api.py`（待重写）
- `c:\dongyuhang\project\Bid collusion inspection\extraction\contact_extractor.py`（复用，不改）
- `c:\dongyuhang\project\Bid collusion inspection\pipeline\orchestrator.py`（复用，不改）
- `c:\dongyuhang\project\Bid collusion inspection\bid-evaluation-collusive-check-ai-curl.md`（接口规范来源）
