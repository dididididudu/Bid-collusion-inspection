# 围标串标检测 API 测试指南

## 基础信息

| 项目 | 值 |
|------|-----|
| 服务地址 | `http://118.178.240.30:8001` |
| API 文档 | `http://118.178.240.30:8001/docs` (Swagger UI) |
| 支持检查项 | 7 个（5 轻量 + 2 重量） |

## 1. 服务状态检查

### 1.1 健康检查

```bash
curl -s http://118.178.240.30:8001/api/v1/collusive-check/health | python3 -m json.tool
```

**预期返回：**
```json
{
    "status": "ok",
    "timestamp": "2026-07-10T15:00:00.000000",
    "active_tasks": 0
}
```

### 1.2 检查项列表

```bash
curl -s http://118.178.240.30:8001/api/v1/collusive-check/item-codes | python3 -m json.tool
```

**预期返回（7 项）：**
```json
{
    "items": [
        {"code": "FILE_CODE_SIMILAR", "name": "文件码雷同"},
        {"code": "EDITOR_SIGNER_SIMILAR", "name": "编辑经办人雷同"},
        {"code": "DOC_AUTHOR_SIMILAR", "name": "文档作者雷同"},
        {"code": "BID_COMPANY_NAME_ABNORMAL", "name": "投标文件公司名称异常"},
        {"code": "SAME_BID_CONTACT_SIMILAR", "name": "同标段单位联系人雷同"},
        {"code": "TECH_BID_SIMILAR", "name": "技术标雷同"},
        {"code": "COM_BID_SIMILAR", "name": "商务标雷同"}
    ]
}
```

---

## 2. 调用流程

### 请求 → 异步任务 → 轮询结果

```
POST /api/v1/collusive-check/items/analyze
  │
  ├─ 返回 202 Accepted
  │   {taskId: "xxx", status: "pending"}
  │
  └─ 轮询 GET /api/v1/collusive-check/items/{taskId}
      │
      ├─ status: "processing" → 继续轮询
      ├─ status: "completed"  → 拿到结果
      └─ status: "failed"     → 查看错误
```

---

## 3. 轻量检查测试（5 项）

### 3.1 准备测试 PDF

先将 PDF 文件预制到下载目录，使 API 跳过下载步骤：

```bash
# 建立 batch 目录
mkdir -p batch_downloads/100

# 复制 PDF（文件名格式：{companyRecordId}_{公司名}.pdf）
cp input/投标文件-雀翼0828.pdf batch_downloads/100/501_公司A.pdf
cp input/投标文件.pdf       batch_downloads/100/502_公司B.pdf
cp input/测试6.pdf          batch_downloads/100/503_公司C.pdf
```

> **注意：** 文件名中的公司名不能包含 `\ / : * ? " < > |` 等特殊字符。

### 3.2 文件码雷同检测

```bash
curl -s -X POST http://118.178.240.30:8001/api/v1/collusive-check/items/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "batchId": 100,
    "projectId": 10001,
    "itemCode": "FILE_CODE_SIMILAR",
    "companies": [
      {"companyRecordId": 501, "registrationCompanyId": 1, "sectionId": 1,
       "bidderName": "公司A", "bidFileUrl": "http://placeholder/a.pdf"},
      {"companyRecordId": 502, "registrationCompanyId": 2, "sectionId": 1,
       "bidderName": "公司B", "bidFileUrl": "http://placeholder/b.pdf"},
      {"companyRecordId": 503, "registrationCompanyId": 3, "sectionId": 1,
       "bidderName": "公司C", "bidFileUrl": "http://placeholder/c.pdf"}
    ]
  }' | python3 -m json.tool
```

**轮询结果（替换 `<taskId>` 为上一步返回的值）：**

```bash
curl -s http://118.178.240.30:8001/api/v1/collusive-check/items/<taskId> | python3 -m json.tool
```

**结果说明：**
- `status: "SUCCESS"` → 未发现雷同
- `status: "FAILED"` → 发现雷同，`evidence.similarCompanyRecordIds` 列出雷同的公司

### 3.3 编辑经办人雷同

```bash
curl -s -X POST http://118.178.240.30:8001/api/v1/collusive-check/items/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "batchId": 100,
    "projectId": 10001,
    "itemCode": "EDITOR_SIGNER_SIMILAR",
    "companies": [
      {"companyRecordId": 501, "registrationCompanyId": 1, "sectionId": 1,
       "bidderName": "公司A", "bidFileUrl": "http://placeholder/a.pdf"},
      {"companyRecordId": 502, "registrationCompanyId": 2, "sectionId": 1,
       "bidderName": "公司B", "bidFileUrl": "http://placeholder/b.pdf"}
    ]
  }' | python3 -m json.tool
```

### 3.4 文档作者雷同

```bash
curl -s -X POST http://118.178.240.30:8001/api/v1/collusive-check/items/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "batchId": 100,
    "projectId": 10001,
    "itemCode": "DOC_AUTHOR_SIMILAR",
    "companies": [
      {"companyRecordId": 501, "registrationCompanyId": 1, "sectionId": 1,
       "bidderName": "公司A", "bidFileUrl": "http://placeholder/a.pdf"},
      {"companyRecordId": 502, "registrationCompanyId": 2, "sectionId": 1,
       "bidderName": "公司B", "bidFileUrl": "http://placeholder/b.pdf"}
    ]
  }' | python3 -m json.tool
```

### 3.5 公司名称异常

```bash
curl -s -X POST http://118.178.240.30:8001/api/v1/collusive-check/items/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "batchId": 100,
    "projectId": 10001,
    "itemCode": "BID_COMPANY_NAME_ABNORMAL",
    "companies": [
      {"companyRecordId": 501, "registrationCompanyId": 1, "sectionId": 1,
       "bidderName": "公司A", "bidFileUrl": "http://placeholder/a.pdf"},
      {"companyRecordId": 502, "registrationCompanyId": 2, "sectionId": 1,
       "bidderName": "公司B", "bidFileUrl": "http://placeholder/b.pdf"}
    ]
  }' | python3 -m json.tool
```

### 3.6 联系人雷同

```bash
curl -s -X POST http://118.178.240.30:8001/api/v1/collusive-check/items/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "batchId": 100,
    "projectId": 10001,
    "itemCode": "SAME_BID_CONTACT_SIMILAR",
    "companies": [
      {"companyRecordId": 501, "registrationCompanyId": 1, "sectionId": 1,
       "bidderName": "公司A", "bidFileUrl": "http://placeholder/a.pdf"},
      {"companyRecordId": 502, "registrationCompanyId": 2, "sectionId": 1,
       "bidderName": "公司B", "bidFileUrl": "http://placeholder/b.pdf"}
    ]
  }' | python3 -m json.tool
```

---

## 4. 重量检查测试（2 项）

### 4.1 技术标雷同

首次运行会自动触发全量管线（Phase 0-5），耗时较长（几分钟到几十分钟）。

```bash
curl -s -X POST http://118.178.240.30:8001/api/v1/collusive-check/items/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "batchId": 200,
    "projectId": 10001,
    "itemCode": "TECH_BID_SIMILAR",
    "companies": [
      {"companyRecordId": 601, "registrationCompanyId": 1, "sectionId": 1,
       "bidderName": "公司A", "bidFileUrl": "http://placeholder/a.pdf"},
      {"companyRecordId": 602, "registrationCompanyId": 2, "sectionId": 1,
       "bidderName": "公司B", "bidFileUrl": "http://placeholder/b.pdf"}
    ]
  }' | python3 -m json.tool
```

拿到 `taskId` 后轮询（可多等一会儿再查）：

```bash
# 每 10 秒轮询一次
curl -s http://118.178.240.30:8001/api/v1/collusive-check/items/<taskId> | python3 -m json.tool
```

### 4.2 商务标雷同

**与上一步使用同一个 batchId，管线结果自动复用缓存，秒级返回：**

```bash
curl -s -X POST http://118.178.240.30:8001/api/v1/collusive-check/items/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "batchId": 200,
    "projectId": 10001,
    "itemCode": "COM_BID_SIMILAR",
    "companies": [
      {"companyRecordId": 601, "registrationCompanyId": 1, "sectionId": 1,
       "bidderName": "公司A", "bidFileUrl": "http://placeholder/a.pdf"},
      {"companyRecordId": 602, "registrationCompanyId": 2, "sectionId": 1,
       "bidderName": "公司B", "bidFileUrl": "http://placeholder/b.pdf"}
    ]
  }' | python3 -m json.tool
```

### 4.3 重量检查返回示例

```json
{
    "batchId": 200,
    "itemCode": "TECH_BID_SIMILAR",
    "itemName": "技术标雷同",
    "results": [
        {
            "companyRecordId": 601,
            "registrationCompanyId": 1,
            "sectionId": 1,
            "status": "FAILED",
            "summary": "技术标与 1 家公司雷同",
            "evidence": {
                "detail": [
                    {"type": "text", "similarity": 0.85, "companyRecordId": 602},
                    {"type": "image", "imageMatches": 3, "companyRecordId": 602}
                ],
                "similarCompanyRecordIds": [602]
            }
        },
        {
            "companyRecordId": 602,
            "registrationCompanyId": 2,
            "sectionId": 1,
            "status": "SUCCESS",
            "summary": "未发现技术标雷同",
            "evidence": {}
        }
    ]
}
```

---

## 5. 缓存机制验证

同一 batch 的检查结果会自动缓存（TTL=3600s），验证方式：

```bash
# 1. 先提交一个 batch 的重量检查（首次，慢）
# 2. 再次提交同一 batch 的相同 itemCode（秒级返回）
# 3. 提交同一 batch 的另一个重量检查 itemCode（也是秒级，独立缓存）
# 4. 换一个 batchId 提交（重新跑管线）
```

缓存 key 格式：
- `TECH_BID_SIMILAR` → `{batchId}_technical`
- `COM_BID_SIMILAR` → `{batchId}_commercial`

---

## 6. 完整测试脚本

项目自带的健康检查脚本可一键运行全部测试：

```bash
# 仅轻量检查（需要 input/ 目录有 PDF 文件）
python scripts/server_health_check.py --pdf-dir ./input --lightweight-only

# 完整检查（含管线）
python scripts/server_health_check.py --pdf-dir ./input

# 测试远程服务器
python scripts/server_health_check.py --api-url http://118.178.240.30:8001 --pdf-dir ./input
```

---

## 7. 常见问题

### 7.1 端口被占用

```bash
# 查找占用进程
netstat -tlnp | grep 8001

# 杀掉
pkill -f "collusive_check_api.py"

# 重新启动
python collusive_check_api.py
```

### 7.2 管线运行到一半卡住

查看日志输出，常见原因：
- **PaddleOCR 未装** → `pip install paddleocr==2.10.0 paddlepaddle`
- **SBERT 模型首次下载慢** → 等待或离线部署
- **内存不足** → 减少 `PHASE1_WORKERS` 和 `PHASE3_WORKERS`

### 7.3 PDF 下载失败

将 PDF 手动预制到 `batch_downloads/{batchId}/` 目录即可跳过下载。
