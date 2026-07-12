# BatchBidCollusionDetector

投标文件串标围标自动检测系统。基于多维度证据链（文本相似度、元数据比对、联系人/公司信息提取、图片比对），识别投标文件中的异常关联。

## 系统架构

系统采用六阶段流式管道架构，支持大规模文档的并行处理与断点续传。

```
输入PDF目录
    │
    ▼
Phase 0: 扫描与元数据提取  ── 提取文件码、作者、创建时间等元数据
    │
    ▼
Phase 1: 特征提取          ── 文本提取、段落分词、MinHash 签名、图片哈希
    │                         （支持多进程并行）
    ▼
Phase 1.5: SBERT 嵌入编码  ── 全局段落嵌入编码，持久化至 SQLite
    │                         （Phase 3 仅查表，不重复调用模型）
    ▼
Phase 2: 候选对选择        ── LSH + 元数据指纹 + 文档向量 + 图片哈希四级筛选
    │
    ▼
Phase 3: 逐对分析          ── 三阶段段落匹配 + 四层图片比对 + 联系人提取
    │
    ▼
Phase 4: 报告生成          ── JSON 结构化数据
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动 API 服务

```bash
python collusive_check_api.py
```

服务默认监听 `http://0.0.0.0:8001`，API 文档自动生成在 `http://localhost:8001/docs`。

### 3. 提交检测任务

```bash
curl -X POST http://localhost:8000/api/detect \
  -F "files=@公司A_标书.pdf" \
  -F "files=@公司B_标书.pdf" \
  -F "content_similarity=true"
```

### 4. 轮询结果

```bash
curl http://localhost:8000/api/detect/{task_id}
```

### 5. 下载报告

```bash
curl http://localhost:8000/api/detect/{task_id}/report -o report.pdf
```

## API 接口

| 端点                          | 方法   | 说明                     |
|-------------------------------|--------|--------------------------|
| `/api/detect`                 | POST   | 提交检测任务，返回 task_id |
| `/api/detect/{task_id}`       | GET    | 查询任务状态与结果        |
| `/api/detect/{task_id}/report`| GET    | 下载 PDF 报告            |
| `/api/dimensions`             | GET    | 获取可用检测维度          |
| `/api/health`                 | GET    | 健康检查                 |

### 检测维度

| 维度                 | 说明                           | 默认启用 |
|----------------------|--------------------------------|----------|
| `content_similarity` | 文本段落匹配 + 图片哈希比对     | 是       |
| `file_id`            | PDF 文件码（/ID[0]）雷同        | 是       |
| `author`             | 文档作者雷同                   | 是       |
| `editor`             | 编辑经办人（creator/producer）雷同 | 是    |
| `contact`            | 单位联系人（手机/邮箱/姓名）雷同 | 是       |
| `company_name`       | 公司名称异常                   | 是       |
| `credit_code`        | 统一社会信用代码雷同           | 是       |
| `member_id`          | 会员号雷同                     | 否       |

## 核心检测能力

### 三阶段段落匹配

| 阶段    | 方法                       | 说明                             |
|---------|----------------------------|----------------------------------|
| Stage 1 | MinHash Jaccard（向量化）   | numpy 广播批量计算，100x 加速    |
| Stage 2a| 精确词级 Jaccard           | 基于 jieba 分词的词集比对         |
| Stage 2b| SBERT 语义验证             | 识别改写、同义词替换等隐蔽相似    |

### 四层图片比对

| 层级 | 检测内容             | 说明                       |
|------|----------------------|----------------------------|
| L1   | 多哈希共识匹配       | pHash + dHash + 长宽比联合 |
| L2   | 图片文字语义比对     | SBERT / Jaccard 回退       |
| L3   | 相同错别字 / 稀有词  | 共享 OCR 错误与特有词汇    |
| L4   | PS 嫌疑检测          | 文字不同但背景相同         |

### 联系人/公司信息提取

通过正则表达式从文档全文中提取手机号、邮箱、公司名称、统一社会信用代码，跨文档交叉比对。

## 输出报告

系统生成 JSON 格式报告：

| 格式                     | 说明                               |
|--------------------------|------------------------------------|
| `detection_report.json`  | 完整结构化数据                     |

## 本地测试

```bash
# 全功能自动化测试（覆盖所有维度，自动清理临时文件）
python test_suite.py
```

## 命令行工具

```bash
# 批量检测
python main.py --input ./bids/ --output ./report/

# 启用 GPU 加速
python main.py --input ./bids/ --output ./report/ --gpu

# 限制检测维度
python main.py --input ./bids/ --output ./report/ --dimensions text,contact

# 仅运行环境诊断
python main.py --diagnose
```

## 配置

系统通过 `config.py` 中的 `DetectionConfig` 类管理配置，支持 JSON 文件加载与环境变量覆盖。

| 参数                      | 默认值 | 说明                      |
|---------------------------|--------|---------------------------|
| `PARAGRAPH_MIN_JACCARD`   | 0.10   | 段落匹配最低 MinHash 相似度 |
| `SBERT_BASE_THRESHOLD`    | 0.60   | SBERT 语义匹配阈值         |
| `PHASE1_WORKERS`          | 4      | Phase 1 并行进程数         |
| `CHUNK_PAGE_SIZE`         | 50     | 每个 Chunk 的页数          |
| `ENABLE_OCR`              | True   | 是否启用图片 OCR           |

## 离线部署

系统默认使用 `HF_ENDPOINT=https://hf-mirror.com` 镜像下载模型。SBERT 模型缓存于 `./models/` 目录。模型加载策略为**优先加载本地缓存**，仅本地不存在时尝试在线下载。

```bash
set TRANSFORMERS_OFFLINE=1
python collusive_check_api.py
```

## 系统要求

- Python 3.8+
- 操作系统：Windows / Linux / macOS
- 可选：NVIDIA GPU（CUDA 加速）
- 磁盘空间：约 4GB（含模型缓存）
