# BatchBidCollusionDetector

投标文件串标围标检测系统

## 系统概述

自动检测投标文件中的串标、围标行为。支持两种运行模式：

- **传统模式**：适合少量小文件（10-30个，<200页），全量加载到内存
- **流式模式**：适合大量大文件（100+个，1000+页），低内存占用，支持断点续传

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 传统模式（适合小规模）
python main.py --input ./bids/ --output ./report/

# 流式模式（适合大规模，推荐）
python main.py --input ./bids/ --output ./report/ --streaming

# 启用 GPU 加速
python main.py --input ./bids/ --output ./report/ --streaming --gpu

# 使用自定义配置
python main.py --input ./bids/ --output ./report/ --streaming --config config.json
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input` | 输入 PDF 目录（必需） | — |
| `--output` | 输出报告目录（必需） | — |
| `--config` | JSON 配置文件路径 | — |
| `--log-level` | 日志级别: DEBUG/INFO/WARNING/ERROR | INFO |
| `--streaming` | 启用流式管道（适合大量大文件） | False |
| `--gpu` | 启用 GPU 加速（CUDA/MPS） | False |
| `--no-checkpoint` | 禁用断点续传 | False |

## 系统架构

```
输入PDF → Phase0扫描 → Phase1提取 → Phase2筛选 → Phase3分析 → Phase4评分 → Phase5报告
```

### 模块结构

```
├── main.py                     # 入口，支持传统/流式双模式
├── config.py                   # 配置参数（20+可调参数）
├── data_structures.py          # 数据结构定义
├── analyzer.py                 # 传统模式分析器（保留向后兼容）
├── selector.py                 # 传统模式选择器
├── extractor.py                # 传统模式提取器（pdfplumber）
├── scoring.py                  # 风险评分引擎
├── report.py                   # 报告生成（HTML/JSON/CSV/TXT）
│
├── extraction/                 # 流式 PDF 处理模块
│   ├── base.py                 #   提取器抽象基类
│   ├── pdf_extractor.py        #   PyMuPDF 高速解析（10x加速）
│   ├── text_processor.py       #   分块分词 + 聚合哈希
│   └── feature_cache.py        #   SQLite 持久化存储
│
├── matching/                   # 流式相似度比对模块
│   ├── selector.py             #   datasketch MinHashLSH 候选筛选
│   ├── lsh_index.py            #   文档/段落级 LSH 索引
│   ├── paragraph_matcher.py    #   三阶段匹配引擎
│   └── semantic_matcher.py     #   GPU/ONNX SBERT 推理
│
└── pipeline/                   # 流式编排模块
    ├── orchestrator.py         #   5 阶段管道管理
    ├── checkpoint.py           #   断点续传
    └── streaming_context.py    #   LRU 内存管理
```

### 三阶段匹配引擎

```
阶段1: MinHash Jaccard 向量化筛选
  100K 句子对 → ~3000 候选 (numpy 广播，100x 加速)

阶段2a: 精确单词 Jaccard (jieba 分词)
  ≥0.75 → 确认为匹配（完全相同/高度相同）
  0.15-0.75 → 进入 SBERT 验证

阶段2b: SBERT 语义验证
  识别改写/同义词替换等隐蔽相似
```

## 流式模式 vs 传统模式

| 特性 | 传统模式 | 流式模式 |
|------|----------|----------|
| PDF 解析引擎 | pdfplumber | PyMuPDF (fitz) |
| 解析速度 | ~0.5-2s/页 | ~0.006s/页 (100-300x) |
| 文本截断 | 10万字符 | 无截断 |
| 内存占用 | ~500MB (30文档) | ~80MB |
| 候选筛选 | SimHash O(n²) | datasketch LSH O(n) |
| 图片提取 | 嵌入图片 | 嵌入图片 + 页面渲染 |
| 文本切分粒度 | 段落级 (30-2000字) | 句子级 (15-500字) |
| 断点续传 | ❌ | ✅ |
| GPU 加速 | ❌ | ✅ CUDA/MPS/ONNX |
| 扫描版支持 | 跳过 | 页级图片哈希比对 |

## 输出报告

系统在输出目录生成 4 种格式：

| 文件 | 说明 |
|------|------|
| `detection_report.html` | HTML 可视化报告，高亮相似文本 |
| `detection_report.json` | 完整 JSON 数据 |
| `summary.txt` | 文本摘要（含单文档风险评估） |
| `suspicious_pairs.csv` | 可疑对 CSV 列表 |

## 配置参数

完整配置见 `config.example.json`。关键参数：

```json
{
  "STREAMING_MODE": false,
  "CHUNK_PAGE_SIZE": 50,
  "ENABLE_CHECKPOINT": true,
  "USE_GPU": false,
  "SBERT_DEVICE": "cpu",
  "MINHASH_LSH_THRESHOLD": 0.3,
  "PARAGRAPH_MIN_JACCARD": 0.05,
  "SBERT_BASE_THRESHOLD": 0.60,
  "PDF_EXTRACTOR_BACKEND": "pymupdf",
  "MAX_WORKERS": 8
}
```

### 阈值调优建议

| 场景 | 建议 |
|------|------|
| 提高召回率 | 降低 `SBERT_BASE_THRESHOLD` (0.50-0.55)，降低 `PARAGRAPH_MIN_JACCARD` (0.03) |
| 降低误报率 | 提高 `SBERT_BASE_THRESHOLD` (0.70+)，提高 `CLONE_BLOCK_MIN_LENGTH` (5) |
| GPU 加速 | `--gpu` 或设置 `SBERT_DEVICE: "cuda"` |
| 超大文件 | 增大 `CHUNK_PAGE_SIZE` (100)，减少 `MAX_CHUNKS_IN_MEMORY` (3) |

## 系统要求

- Python 3.7+
- 文本型 PDF（可直接提取字符）优先
- 扫描版 PDF 使用图片比对
- 首次运行自动下载 SBERT 模型（~400MB），存放于 `models/` 目录

