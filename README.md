# 围标串标智能检测系统 (Bid Collusion Inspection)

> 检测投标文件中多家公司之间的异常关联，自动识别围标串标风险。系统接收 PDF 投标文件，通过 **7 个检测维度** 进行多维度交叉比对，输出每家公司每个维度的检查结论（通过/异常/错误）及详细证据链。设计目标是为 Java 后端提供异步调用的 AI 检测服务。

---

## 功能特性

### 7 大检测维度

| 维度 | 检测内容 | 类型 | 耗时 |
| :--- | :--- | :--- | :--- |
| **文件码雷同** | PDF 文件唯一标识码（`/ID[0]`）比对 | PDF 指纹 | 秒级 |
| **文档作者雷同** | PDF 元数据 Author 字段比对 | 元数据 | 秒级 |
| **编辑经办人雷同** | PDF Creator/Producer 字段比对 | 元数据 | 秒级 |
| **人名雷同** | 从正文中提取手机号/邮箱/联系人名，交叉比对 | 正文提取 | 秒级 |
| **公司名雷同** | 从正文中提取公司名称，交叉比对 | 正文提取 | 秒级 |
| **技术标雷同** | 文本段落匹配 + 图片比对 + OCR 文字比对（限技术标页） | 全文分析 | 数十分钟 |
| **商务标雷同** | 文本段落匹配 + 图片比对 + OCR 文字比对（限商务标页） | 全文分析 | 数十分钟 |

### 技术亮点

- **三阶段段落匹配**: MinHash Jaccard（批量向量化）→ 精确词级 Jaccard → SBERT 语义验证
- **四层图片比对**: 多哈希共识 → 图片文字语义 → 相同错别字/稀有词 → PS 嫌疑检测
- **六阶段流式管线**: 支持断点续传、多进程并行、SQLite 缓存复用
- **技术标/商务标分离**: 页级分类，严格模式下互不污染
- **跨批次缓存**: SBERT embedding 按 text_hash 缓存，同文本不重复编码

---

## 环境要求

| 项目 | 要求 |
| :--- | :--- |
| **Python** | **3.10.11**（3.10.x，3.12 部分依赖不兼容） |
| **操作系统** | CentOS 7+ / Ubuntu 20.04+ / Windows 10+ |
| **CPU** | 最低 4 核，推荐 8 核以上 |
| **内存** | 最低 8GB，推荐 16GB+（大 PDF 峰值可达 12GB） |
| **GPU** | NVIDIA + CUDA 11.8+（可选，仅用于 SBERT 加速） |
| **磁盘** | 约 4GB（含模型缓存）+ 投标文件存储 |
| **系统库** | `libgl1` `libglib2.0-0` `gcc`（Linux OCR 必需） |

---

## 安装步骤

### 1. 克隆并创建虚拟环境

```bash
git clone <仓库地址>
cd Bid-collusion-inspection

# CentOS / Ubuntu
python3.10 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 2. 安装依赖

```bash
# CPU 服务器（推荐）
pip install -r requirements-cpu.txt
pip install -r deploy/requirements.api.txt

# GPU 服务器（额外安装 CUDA 依赖）
pip install -r deploy/requirements.gpu.txt
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 按需编辑 .env 文件
```

关键环境变量：

| 变量 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `OCR_ENGINE` | `rapidocr` | OCR 引擎：`rapidocr` / `paddleocr` / `easyocr` |
| `USE_GPU` | `false` | 是否启用 GPU（仅 SBERT） |
| `SBERT_DEVICE` | `cpu` | `cpu` / `cuda` / `mps` |
| `PHASE1_WORKERS` | `2` | Phase 1 并行进程数 |
| `PHASE3_WORKERS` | `2` | Phase 3 并行进程数 |
| `COLLUSIVE_ENABLE_OCR` | `1` | 启用 OCR 识别 |
| `COLLUSIVE_HOST` | `0.0.0.0` | API 绑定地址 |
| `COLLUSIVE_PORT` | `8001` | API 端口 |

完整变量表见 [docs/交接文档.md](docs/交接文档.md#环境变量速查表)。

### 4. 启动服务

```bash
# 开发模式
python collusive_check_api.py

# 生产模式（systemd 托管，CentOS/Ubuntu）
sudo bash deploy/deploy_cpu.sh --port 8001
```

服务默认监听 `http://0.0.0.0:8001`，Swagger 文档在 `http://localhost:8001/docs`。

### 5. 命令行批量检测（离线模式）

```bash
# 批量检测 PDF 目录
python main.py --input ./input/ --output ./output/

# 仅运行环境诊断（不执行检测）
python main.py --diagnose
```

---

## 配置说明

### 配置加载优先级（从低到高）

1. `config.py` 默认值 → 2. `config.json` → 3. 环境变量 → 4. `.env` 文件

**不要在代码或 JSON 配置中写入真实密钥。**

### .env.example

```bash
OCR_ENGINE=rapidocr
USE_GPU=false
SBERT_DEVICE=cpu
PHASE1_WORKERS=2
PDF_CHUNK_WORKERS=2
OCR_COLLECT_WORKERS=2
OCR_WORKERS=1
PHASE3_WORKERS=2
OCR_BATCH_SIZE=4
COLLUSIVE_ENABLE_OCR=1
COLLUSIVE_ENABLE_IMAGE_ANALYSIS=1
```

---

## 如何验证跑通了

### 方式 1：健康检查

```bash
curl http://localhost:8001/api/health
```

预期返回：`{"status": "healthy", "ocr_engine": "rapidocr", "message": "围标串标AI服务运行正常"}`

### 方式 2：轻量检测测试

```bash
python scripts/perf_test_collusive_api.py \
  --pdf-dir batch_downloads/75689 \
  --api http://127.0.0.1:8001 \
  --items FILE_CODE_SIMILAR DOC_AUTHOR_SIMILAR EDITOR_SIGNER_SIMILAR
```

预期 `status=200`，秒级完成。

### 方式 3：全量测试（含重型维度）

```bash
python scripts/perf_test_collusive_api.py \
  --pdf-dir batch_downloads/75689 \
  --api http://127.0.0.1:8001 \
  --heavy
```

重型检测耗时数秒到数十分钟不等。

### 方式 4：内部单元测试（不需要 API）

```bash
python test_api.py --module internal
```

---

## 项目结构

```
Bid-collusion-inspection/
├── collusive_check_api.py   # API 服务入口（FastAPI）
├── main.py                  # 命令行批量检测入口
│
├── extraction/              # PDF 提取层（文本/元数据/图片/联系人）
├── pipeline/                # 核心管线编排（6 阶段）
├── matching/                # 文本匹配引擎（MinHash/SBERT）
├── embedding/               # SBERT 向量编码与缓存
├── image_analysis/          # 图片哈希、比对、OCR
├── config.py                # 配置参数系统
├── data_structures.py       # 数据结构定义
├── scoring.py               # 风险评分引擎
├── report.py                # 报告生成器
│
├── deploy/                  # 部署脚本（systemd/GPU/模型下载）
├── scripts/                 # 性能测试/健康检查脚本
├── client/                  # Python 客户端调用示例
├── docs/                    # 详细文档
│   ├── 运行流程与测试流程.md
│   ├── 交接文档.md
│   └── API测试指南.md
│
├── models/                  # 模型缓存（SBERT/RapidOCR）
├── batch_downloads/         # PDF 下载缓存
├── collusive_workdir/       # 管线工作目录
├── input/ / output/         # 测试输入输出
│
├── test_api.py              # 综合测试
├── test_evidence_unit.py    # 证据单元测试
└── requirements*.txt        # 依赖清单
```

---

## API 接口

| 端点 | 方法 | 说明 |
| :--- | :--- | :--- |
| `/api/v1/collusive-check/items/analyze` | POST | 提交单项检测任务 |
| `/api/v1/collusive-check/items/{taskId}` | GET | 轮询检测结果 |
| `/api/health` | GET | 健康检查 |

详细接口规范（请求体结构、响应格式、evidence 字段说明）见：

- [`bid-evaluation-collusive-check-ai-curl.md`](bid-evaluation-collusive-check-ai-curl.md)
- [`docs/运行流程与测试流程.md`](docs/运行流程与测试流程.md)

---

## 相关文档

| 文档 | 说明 |
| :--- | :--- |
| [交接文档](docs/交接文档.md) | 模块详细说明、代码文件职责、坑点汇总、已知问题 |
| [运行流程与测试流程](docs/运行流程与测试流程.md) | 完整运行链路、部署注意事项、测试方法、常见问题排查 |
| [API 接口规范](bid-evaluation-collusive-check-ai-curl.md) | Java 后端调用 curl 示例、请求/响应结构、枚举说明 |
