# BatchBidCollusionDetector

投标文件串标围标检测系统

## 系统概述

BatchBidCollusionDetector 是一个用于检测投标文件中串标、围标行为的自动化分析系统。系统通过对多个PDF投标文件进行批量特征提取、相似度计算、风险评级和聚类分析，自动识别可疑的串标围标行为。

## 核心功能

- **文档解析与特征提取**: 并行提取PDF文件的文本、元数据、报价、图片指纹等特征
- **智能初筛**: 使用SimHash/MinHash/LSH技术快速筛选可疑文档对，降低计算复杂度
- **两层相似度检测**: SequenceMatcher预过滤 + SBERT语义验证，精准识别改写、同义词替换等隐蔽抄袭
- **连续克隆块检测**: 识别超过3个连续段落雷同的克隆块，标记为高风险
- **风险评级**: 基于混合评分模型的风险评分算法，自动判定风险等级
- **聚类分析**: 发现围标团伙，生成风险聚类
- **单文档风险评估**: 对每个文档进行独立风险评级，存在高相似文档即判定为高风险
- **报告生成**: 输出JSON、文本摘要和CSV格式的检测报告，清晰展示相似内容

## 系统架构

系统分为5个核心模块：

### 模块A - 文档解析与特征提取引擎 (`extractor.py`)
- 并行读取PDF文件
- 提取文本、元数据、报价、图片哈希
- 生成SimHash、MinHash等特征向量
- 文本噪声清洗（移除目录虚线、页码、纯符号行）

### 模块B - 快速初筛引擎 (`selector.py`)
- SimHash汉明距离初筛
- LSH桶分桶
- 元数据和图片哈希倒排索引

### 模块C - 精细相似度计算引擎 (`analyzer.py`)
- **两层检测流程**: SequenceMatcher预过滤 + SBERT语义验证
- **混合评分策略**: 质量分数 × 覆盖率衰减因子
- **连续克隆块检测**: 识别超过3个连续相似段落
- 元数据关联分析
- 报价规律检测
- 图片重复检测

### 模块D - 风险评级与聚类引擎 (`scoring.py`)
- 多维度风险评分
- 覆盖率门限检查（避免少量相似段落误判高风险）
- 风险等级判定
- 图聚类分析（连通分量）

### 模块E - 报告生成引擎 (`report.py`)
- JSON完整报告
- 文本摘要报告（含单文档风险评估）
- CSV可疑对列表

## 核心算法与计算方式

### 1. 两层相似度检测流程

```
段落比对流程：
┌─────────────────────────────────────────────────────────────┐
│  段落A + 段落B                                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  SequenceMatcher预过滤（快速文本相似度计算）                   │
│  ├── similarity > 0.85  → 直接判定高相似                    │
│  ├── similarity < 0.4   → 直接判定低相似                    │
│  └── 0.4 ≤ similarity < 0.85 → 进入SBERT验证               │
└─────────────────────────────────────────────────────────────┘
                            │ (需验证)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  SBERT语义验证（深度学习语义匹配）                            │
│  ├── 动态阈值调整：                                          │
│  │   - 短段落（<100字符）：阈值 0.85                        │
│  │   - 正常段落：阈值 0.75                                  │
│  │   - 候选数量>100：阈值上调0.02                           │
│  │   - 候选数量<10：阈值下调0.02                            │
│  └── similarity > 阈值 → 判定高相似                         │
└─────────────────────────────────────────────────────────────┘
```

### 2. 混合评分策略（乘法模型）

**质量分数计算：**
```
quality_score = (SCORE_WEIGHT_MAX × max_sim) + 
                (SCORE_WEIGHT_TOP_K × top_k_sim) + 
                (SCORE_WEIGHT_MEAN × mean_sim)

其中：
- max_sim: 所有匹配段落的最大相似度
- top_k_sim: 前K个最高相似度的平均值（K=5）
- mean_sim: 所有匹配段落的平均相似度
- 默认权重: max(40%) + top_k(30%) + mean(20%) = 90%
```

**覆盖率衰减因子：**
```
coverage_factor = 1 - exp(-5 × coverage)

其中：
coverage = (covered_paras_a + covered_paras_b) / total_paras
- covered_paras_a: 文档A中被覆盖的段落数
- covered_paras_b: 文档B中被覆盖的段落数
- total_paras: 两文档总段落数
```

**综合相似度：**
```
mixed_score = quality_score × coverage_factor
```

### 3. 风险等级判定规则

```
判定流程：
┌─────────────────────────────────────────────────────────────┐
│  text_local ≥ 0.75                                          │
│  ├── coverage ≥ 0.1 且 match_count ≥ 5 → HIGH（高风险）      │
│  └── coverage < 0.1 或 match_count < 5 → MEDIUM（中等风险） │
├─────────────────────────────────────────────────────────────┤
│  text_local < 0.75 → LOW（低风险）                           │
└─────────────────────────────────────────────────────────────┘
```

### 4. 连续克隆块检测

```
连续克隆块定义：
- 连续3个及以上段落相似度 > 0.85
- 允许最大间隔1个段落
- 每个克隆块标记组ID和长度
```

### 5. 单文档风险评估

```
单文档风险评级：
- 该文档存在至少一个高相似文档 → 高风险 🔴
- 该文档存在中等相似文档但无高相似 → 低风险 🟡
- 该文档无任何相似文档 → 无风险 🟢
```

### 6. 文本噪声清洗规则

```
清洗规则（预处理阶段）：
1. 移除目录虚线行（包含5个以上连续的点/省略号）
2. 移除纯页码行（仅数字）
3. 移除纯符号行（不含中文字符且长度<20）
```

## 安装依赖

```bash
pip install -r requirements.txt
```

首次运行时会自动下载SBERT模型（约400MB），也可手动预下载：
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
```

## 使用方法

### 基本用法

```bash
python main.py --input ./bids/ --output ./report/
```

### 使用自定义配置

```bash
python main.py --input ./bids/ --output ./report/ --config config.json
```

### 调整日志级别

```bash
python main.py --input ./bids/ --output ./report/ --log-level DEBUG
```

## 命令行参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `--input` | str | 是 | 输入PDF文件目录路径 |
| `--output` | str | 是 | 输出报告目录路径 |
| `--config` | str | 否 | 配置文件路径（JSON格式） |
| `--log-level` | str | 否 | 日志级别（默认INFO，可选DEBUG/INFO/WARNING/ERROR） |

## 配置参数

系统支持通过JSON配置文件自定义检测参数。配置文件示例：

```json
{
  "TEXT_GLOBAL_THRESHOLD": 0.85,
  "TEXT_LOCAL_THRESHOLD": 0.92,
  "TYPO_MIN_LENGTH": 4,
  
  "SCORE_WEIGHT_MAX": 0.4,
  "SCORE_WEIGHT_TOP_K": 0.3,
  "SCORE_WEIGHT_MEAN": 0.2,
  "SCORE_TOP_K": 5,
  
  "SBERT_BASE_THRESHOLD": 0.75,
  "SBERT_SHORT_PARAGRAPH_THRESHOLD": 0.85,
  "SBERT_SHORT_PARAGRAPH_LEN": 100,
  
  "CLONE_BLOCK_MIN_LENGTH": 3,
  "CLONE_BLOCK_MAX_GAP": 1,
  
  "METADATA_MATCH_THRESHOLD": 3,
  "QUOTE_COMMON_THRESHOLD": 2,
  "IMAGE_COMMON_THRESHOLD": 1,
  
  "RISK_HIGH_THRESHOLD": 70,
  "RISK_MEDIUM_THRESHOLD": 40,
  "RISK_LOW_THRESHOLD": 15,
  
  "MAX_WORKERS": 8,
  "MAX_CANDIDATE_PAIRS": 5000,
  "MAX_TEXT_LENGTH": 100000
}
```

### 配置参数详细说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TEXT_GLOBAL_THRESHOLD` | 0.85 | SBERT全局语义相似度阈值 |
| `TEXT_LOCAL_THRESHOLD` | 0.92 | 段落级相似度阈值 |
| `SCORE_WEIGHT_MAX` | 0.4 | 最大相似度权重 |
| `SCORE_WEIGHT_TOP_K` | 0.3 | Top-K相似度权重 |
| `SCORE_WEIGHT_MEAN` | 0.2 | 平均相似度权重 |
| `SCORE_TOP_K` | 5 | Top-K取前K个 |
| `SBERT_BASE_THRESHOLD` | 0.75 | SBERT基础阈值 |
| `SBERT_SHORT_PARAGRAPH_THRESHOLD` | 0.85 | 短段落SBERT阈值 |
| `SBERT_SHORT_PARAGRAPH_LEN` | 100 | 短段落长度阈值 |
| `CLONE_BLOCK_MIN_LENGTH` | 3 | 最小连续克隆块长度 |
| `CLONE_BLOCK_MAX_GAP` | 1 | 允许的最大间隔 |
| `MAX_WORKERS` | 8 | 并行进程数 |
| `MAX_TEXT_LENGTH` | 100000 | 文本长度限制（防止内存溢出） |

## 输出报告

系统会在输出目录生成以下文件：

### 1. detection_report.json
完整的检测报告（JSON格式），包含所有检测数据和证据链

### 2. summary.txt
文本摘要报告，包含：
- 总体统计信息
- 风险聚类结果
- 高风险对详情（含相似段落标记）
- 单文档风险评估

### 3. suspicious_pairs.csv
可疑文档对列表（CSV格式），可用Excel打开

## 风险等级说明

| 等级 | 条件 | 说明 |
|------|------|------|
| **HIGH** | text_local ≥ 0.75 且 coverage ≥ 0.1 且 match_count ≥ 5 | 存在明显的串标围标特征 |
| **MEDIUM** | text_local ≥ 0.75 但 coverage < 0.1 或 match_count < 5 | 存在可疑特征但覆盖率不足 |
| **LOW** | text_local < 0.75 | 存在少量可疑特征 |

## 检测维度

### 1. 文本相似度
- **SequenceMatcher预过滤**: 快速文本相似度计算，过滤70%低相似对
- **SBERT语义验证**: 深度学习模型理解文本含义，识别改写、同义词替换
- **连续克隆块**: 检测超过3个连续相似段落
- **相似段落标记**: 使用高亮标记重复内容

### 2. 元数据关联
- 作者、创建软件、生成时间等元数据的匹配度
- 时间桶分析（按小时分组）

### 3. 报价规律
- 共同金额、尾数分布、固定差额/比例规律

### 4. 图片重复
- 嵌入图片的感知哈希（pHash）匹配

## 技术特点

- **高性能**: SequenceMatcher预过滤减少70%的SBERT调用
- **并行化**: 段落比对使用ThreadPoolExecutor并行处理
- **高精度**: 乘法评分模型综合考虑质量和覆盖率
- **鲁棒性**: 噪声清洗（目录虚线、页码等）避免误判
- **可解释**: 生成详细的证据链，每个风险判定都有明确依据
- **可配置**: 所有阈值参数可通过配置文件调整
- **日志轮转**: 自动日志轮转（10MB/文件，保留3个备份）

## 系统要求

- Python 3.7+
- 处理对象：普通文本型PDF（可直接提取字符）
- 扫描版PDF仅参与元数据和图片比对

## 注意事项

1. 扫描版PDF（无法提取文本）仅参与元数据和图片比对
2. 单文件大小建议不超过100MB
3. 文档数量N>500时，建议调整MAX_CANDIDATE_PAIRS参数
4. 首次运行会下载jieba分词词典和SBERT模型（约400MB）
5. 模型文件存放在`models/`目录，已排除在版本控制之外

## 许可证

本项目仅供学习研究使用。

## 作者

BatchBidCollusionDetector Development Team