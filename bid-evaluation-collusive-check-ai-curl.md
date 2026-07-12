# 围标串标检查后端调用 AI 说明

## 1. 调用时机

围标串标检查由 Java 后端在线程池中异步调用 py/AI。

后端创建检查批次和批次公司快照后，会按用户选择的检查项顺序逐项调用 py/AI。每次调用只处理一个检查项，但会带上本批次选中的所有投标公司。

围标串标检查的核心调用方式是：

```text
一个检查项 + 多家公司投标文件 -> py/AI 分析 -> 返回每家公司在该检查项下的结果
```

不是按公司逐个调用。

## 2. AI 调用配置

Java 后端通过远程调用模块发起 py/AI 请求。

说明：

- `service-code` 用于远程调用模块定位 py 服务。
- `analyze-item-path` 是 py/AI 的围标串标单项分析接口路径。
- `scene` 固定为 `bid_evaluation_collusive_check`，用于远程调用链路区分业务场景。
- `bizId` 使用检查批次 ID，便于追踪同一次检查任务。

## 3. 后端请求 AI 的 curl 示例

以下 curl 表示 Java 后端实际发给 py/AI 的请求结构。

```bash
curl -X POST "http://python-service/api/v1/collusive-check/items/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "batchId": 1900000000000000001,
    "projectId": 10001,
    "checkMode": "SAME_SECTION",
    "itemCode": "DOC_AUTHOR_SIMILAR",
    "companies": [
      {
        "companyRecordId": 501,
        "registrationCompanyId": 101,
        "sectionId": 11,
        "bidderName": "A公司",
        "bidFileUrl": "https://example.com/files/a-bid.pdf"
      },
      {
        "companyRecordId": 502,
        "registrationCompanyId": 102,
        "sectionId": 11,
        "bidderName": "B公司",
        "bidFileUrl": "https://example.com/files/b-bid.pdf"
      },
      {
        "companyRecordId": 503,
        "registrationCompanyId": 103,
        "sectionId": 11,
        "bidderName": "C公司",
        "bidFileUrl": "https://example.com/files/c-bid.pdf"
      }
    ]
  }'
```

说明：

- `http://python-service` 是示例服务地址。
- 实际服务地址由远程调用模块根据 `serviceCode=python` 解析。
- Java 后端不会一次性把所有检查项传给 py/AI。
- Java 后端会按检查项顺序循环调用 py/AI，每次只传一个 `itemCode`。

## 4. 请求参数说明

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `batchId` | Long | 是 | 检查批次 ID，对应 `bid_evaluation_collusive_check_batch.id`。 |
| `projectId` | Long | 是 | 项目 ID，对应当前围标串标检查所属项目。 |
| `checkMode` | String | 是 | 检查模式：`SAME_SECTION` 同标段，`CROSS_SECTION` 异标段。 |
| `itemCode` | String | 是 | 当前正在分析的检查项编码。每次 AI 调用只处理一个检查项。 |
| `companies` | Array | 是 | 当前批次参与该检查项分析的全部投标公司。 |

## 5. 枚举说明

### 5.1 checkMode 枚举

| 枚举值 | 名称 | 说明 |
| --- | --- | --- |
| `SAME_SECTION` | 同标段检查 | 检测同一个标段内多家投标公司的投标文件。 |
| `CROSS_SECTION` | 异标段检查 | 检测同一个项目下跨多个标段的投标公司的投标文件。 |

### 5.2 itemCode 枚举

实际可用检查项以 `bid_evaluation_collusive_check_item` 字典表为准。当前围标串标检查项设计如下：

| 检查大类 | 小类 | itemCode | 检查项名称 |
| --- | --- | --- | --- |
| 物理检测 | - | `FILE_CODE_SIMILAR` | 文件码雷同 |
| 物理检测 | - | `EDITOR_SIGNER_SIMILAR` | 编辑经办人雷同 |
| 物理检测 | - | `DOC_AUTHOR_SIMILAR` | 文档作者雷同 |
| 内容检测 | 同标段雷同性 | `SAME_BID_CONTACT_SIMILAR` | 人名雷同 |
| 内容检测 | 同标段雷同性 | `SAME_bidderName_SIMILAR` | 公司名雷同 |
| 内容检测 | 技术标 | `TECH_BID_SIMILAR` | 技术标雷同 |
| 内容检测 | 商务标 | `Business_BID_SIMILAR` | 商务标雷同 |

## 6. companies 参数说明

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `companyRecordId` | Long | 是 | 本次检查批次公司记录 ID，对应 `bid_evaluation_collusive_check_company.id`。 |
| `registrationCompanyId` | Long | 是 | 投标报名公司记录 ID，对应 `integration_project_registration_company.id`。 |
| `sectionId` | Long | 是 | 公司所属标段 ID，对应 `integration_project_section.id`。 |
| `bidderName` | String | 是 | 投标公司名称。 |
| `bidFileUrl` | String | 是 | 投标文件可访问地址，py/AI 根据该地址读取投标文件。 |

## 7. 检查项调用顺序

假设本次检查选择了三个检查项：

```json
[
  "FILE_CODE_SIMILAR",
  "DOC_AUTHOR_SIMILAR",
  "TECH_BID_SIMILAR"
]
```

Java 后端会按顺序调用 py/AI 三次：

| 调用顺序 | itemCode | 说明 |
| --- | --- | --- |
| 第 1 次 | `FILE_CODE_SIMILAR` | 文件码雷同检测 |
| 第 2 次 | `DOC_AUTHOR_SIMILAR` | 文档作者雷同检测 |
| 第 3 次 | `TECH_BID_SIMILAR` | 技术标雷同检测 |

每次调用都会带上同一批次的全部公司。

## 8. AI 返回结构

py/AI 返回当前检查项下每家公司的检查结论。

备注：

- 该接口返回的数据不仅用于当前检查结果落库和前端结果展示，也需要为后续“最终报告文档接口”提供数据支撑。
- 当前返回结构中的 `summary` 和 `evidence` 是报告生成的主要依据。
- 如果后续报告文档需要展示更细的证据，例如页码、段落原文、雷同片段、相似度、关联对象、规则命中编号等，而当前返回字段无法支撑，则可以继续调整 py/AI 返回结构和 Java 接收结构。
- 因此该接口的返回数据结构不是绝对固定的，应以最终报告内容需求为准进行扩展。

```json
{
  "batchId": 1900000000000000001,
  "itemCode": "DOC_AUTHOR_SIMILAR",
  "itemName": "文档作者雷同",
  "results": [
    {
      "companyRecordId": 501,
      "registrationCompanyId": 101,
      "sectionId": 11,
      "status": "FAILED",
      "summary": "文档作者与B公司一致",
      "evidence": {
        "author": "张三",
        "similarCompanyRecordIds": [502]
      }
    },
    {
      "companyRecordId": 502,
      "registrationCompanyId": 102,
      "sectionId": 11,
      "status": "FAILED",
      "summary": "文档作者与A公司一致",
      "evidence": {
        "author": "张三",
        "similarCompanyRecordIds": [501]
      }
    },
    {
      "companyRecordId": 503,
      "registrationCompanyId": 103,
      "sectionId": 11,
      "status": "SUCCESS",
      "summary": "未发现文档作者雷同",
      "evidence": {}
    }
  ]
}
```

## 9. 返回参数说明

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `batchId` | Long | 检查批次 ID。 |
| `itemCode` | String | 当前检查项编码。 |
| `itemName` | String | 当前检查项名称。 |
| `results` | Array | 公司级检查结果列表。 |
| `results[].companyRecordId` | Long | 批次公司记录 ID。 |
| `results[].registrationCompanyId` | Long | 投标报名公司记录 ID。 |
| `results[].sectionId` | Long | 公司所属标段 ID。 |
| `results[].status` | String | 检查结果：`SUCCESS` 通过，`FAILED` 存在异常，`ERROR` 检查异常。 |
| `results[].summary` | String | 检查结论摘要或异常原因。 |
| `results[].evidence` | Object | 证据信息，结构随 itemCode 不同而变化，详见下方各维度说明。 |

### 9.1 各检测维度 evidence 结构

#### 文件码雷同（FILE_CODE_SIMILAR）

```json
{
  "fileId": "a1b2c3d4e5f6...",
  "similarCompanyRecordIds": [502]
}
```

| 字段 | 说明 |
| --- | --- |
| `fileId` | PDF 文件唯一标识（MD5）。 |
| `similarCompanyRecordIds` | 文件码雷同的公司记录 ID 列表。 |

#### 文档作者雷同（DOC_AUTHOR_SIMILAR）

```json
{
  "author": "张三",
  "similarCompanyRecordIds": [502]
}
```

| 字段 | 说明 |
| --- | --- |
| `author` | PDF 文档作者名称。 |
| `similarCompanyRecordIds` | 作者雷同的公司记录 ID 列表。 |

#### 编辑经办人雷同（EDITOR_SIGNER_SIMILAR）

```json
{
  "editor": "PDF Creator 1.7",
  "similarCompanyRecordIds": [502]
}
```

#### 人名雷同（SAME_BID_CONTACT_SIMILAR）

```json
{
  "commonMobiles": ["13800138000"],
  "commonPhones": ["010-12345678"],
  "commonEmails": ["test@example.com"],
  "commonPersons": ["张三"],
  "similarCompanyRecordIds": [502]
}
```

#### 公司名雷同（SAME_bidderName_SIMILAR）

```json
{
  "foundCompanyNames": ["B公司"],
  "similarCompanyRecordIds": [502]
}
```

#### 技术标/商务标雷同（TECH_BID_SIMILAR / Business_BID_SIMILAR）

```json
{
  "similarCompanyRecordIds": [502],
  "similarParagraphs": [
    {
      "companyRecordId": 502,
      "similarity": 0.85,
      "paragraphMatches": [
        {
          "paragraph_a": "本项目施工方案采用...",
          "paragraph_b": "本项目施工方案采用...",
          "similarity": 0.95,
          "paragraph_a_index": 12,
          "paragraph_b_index": 8,
          "detection_method": "SBERT"
        }
      ]
    }
  ],
  "similarImages": [
    {
      "companyRecordId": 502,
      "imageMatchCount": 3,
      "similarImages": [
        {
          "source_a": "page5_img0.png",
          "source_b": "page3_img0.png",
          "confidence": 0.92,
          "reasons": ["phash_match", "orb_match"],
          "ocr_text_a": "施工平面图",
          "ocr_text_b": "施工平面图"
        }
      ]
    }
  ]
}
```

| 字段 | 说明 |
| --- | --- |
| `similarCompanyRecordIds` | 雷同公司记录 ID 列表。 |
| `similarParagraphs` | 文本相似段落详情列表。 |
| `similarParagraphs[].companyRecordId` | 对方公司记录 ID。 |
| `similarParagraphs[].similarity` | 文本整体相似度。 |
| `similarParagraphs[].paragraphMatches` | 相似段落对列表（最多 20 对，每段截断 300 字符）。 |
| `similarParagraphs[].paragraphMatches[].paragraph_a` | 本公司段落文本。 |
| `similarParagraphs[].paragraphMatches[].paragraph_b` | 对方公司段落文本。 |
| `similarParagraphs[].paragraphMatches[].similarity` | 段落对相似度。 |
| `similarParagraphs[].paragraphMatches[].detection_method` | 检测方法（SBERT/Jaccard/Exact）。 |
| `similarImages` | 图片相似详情列表。 |
| `similarImages[].companyRecordId` | 对方公司记录 ID。 |
| `similarImages[].imageMatchCount` | 匹配图片总数。 |
| `similarImages[].similarImages` | 相似图片对列表（最多 20 对）。 |
| `similarImages[].similarImages[].source_a` | 本公司图片引用。 |
| `similarImages[].similarImages[].source_b` | 对方公司图片引用。 |
| `similarImages[].similarImages[].confidence` | 匹配置信度。 |
| `similarImages[].similarImages[].reasons` | 匹配原因列表。 |
| `similarImages[].similarImages[].ocr_text_a` | 本公司图片 OCR 文本（截断 200 字符）。 |
| `similarImages[].similarImages[].ocr_text_b` | 对方公司图片 OCR 文本（截断 200 字符）。 |

## 10. 异常处理规则

- 某个检查项调用 py/AI 失败时，Java 后端会构造该检查项的错误结果。
- 该检查项下所有公司结果写为 `ERROR`。
- 当前检查项失败不会阻断后续检查项。
- 用户选择的所有检查项都会继续执行。
- 所有检查项执行完成后，Java 后端汇总公司最终状态和批次最终状态。
