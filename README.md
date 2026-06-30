# BatchBidCollusionDetector

投标文件串标围标检测系统

## 系统概述

BatchBidCollusionDetector 是一个用于检测投标文件中串标、围标行为的自动化分析系统。系统通过对多个PDF投标文件进行批量特征提取、相似度计算、风险评级和聚类分析，自动识别可疑的串标围标行为。

## 核心功能

- **文档解析与特征提取**: 并行提取PDF文件的文本、元数据、报价、图片指纹等特征
- **智能初筛**: 使用SimHash/MinHash/LSH技术快速筛选可疑文档对，降低计算复杂度
- **纯SBERT语义相似度检测**: 全面使用SBERT深度学习模型进行文本向量化和语义匹配
- **风险评级**: 基于规则的风险评分算法，自动判定风险等级
- **聚类分析**: 发现围标团伙，生成风险聚类
- **报告生成**: 输出JSON、文本摘要和CSV格式的检测报告，清晰展示相似内容

## 系统架构

系统分为5个核心模块：

1. **模块A - 文档解析与特征提取引擎** (`extractor.py`)
   - 并行读取PDF文件
   - 提取文本、元数据、报价、图片指纹
   - 生成SimHash、MinHash等特征向量

2. **模块B - 快速初筛引擎** (`selector.py`)
   - SimHash汉明距离初筛
   - LSH桶分桶
   - 元数据和图片哈希倒排索引

3. **模块C - 精细相似度计算引擎** (`analyzer.py`)
   - 纯SBERT方案：全局和局部相似度均使用SBERT语义向量
   - 文本全局相似度：SBERT编码整个文档（前3000字符）
   - 文本局部相似度：SBERT批量编码段落，计算相似度矩阵
   - 元数据关联分析
   - 报价规律检测
   - 图片重复检测
   - 自动降级：SBERT不可用时使用TF-IDF

4. **模块D - 风险评级与聚类引擎** (`scoring.py`)
   - 多维度风险评分
   - 风险等级判定
   - 图聚类分析（连通分量）

5. **模块E - 报告生成引擎** (`report.py`)
   - JSON完整报告
   - 文本摘要报告
   - CSV可疑对列表

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

- `--input`: 输入PDF文件目录路径（必需）
- `--output`: 输出报告目录路径（必需）
- `--config`: 配置文件路径（可选，JSON格式）
- `--log-level`: 日志级别（可选，默认INFO，可选DEBUG/INFO/WARNING/ERROR）

## 配置参数

系统支持通过JSON配置文件自定义检测参数。配置文件示例：

```json
{
  "TEXT_GLOBAL_THRESHOLD": 0.70,
  "TEXT_LOCAL_THRESHOLD": 0.85,
  "METADATA_MATCH_THRESHOLD": 3,
  "QUOTE_COMMON_THRESHOLD": 2,
  "RISK_HIGH_THRESHOLD": 70,
  "RISK_MEDIUM_THRESHOLD": 40,
  "RISK_LOW_THRESHOLD": 15,
  "MAX_WORKERS": 8
}
```

详细配置参数说明请参考 `config.py`。

## 输出报告

系统会在输出目录生成以下文件：

1. **detection_report.json**: 完整的检测报告（JSON格式），包含所有检测数据和证据链
2. **summary.txt**: 文本摘要报告，包含总体统计、风险聚类、高风险对详情
3. **suspicious_pairs.csv**: 可疑文档对列表（CSV格式），可用Excel打开

## 风险等级说明

- **HIGH（高风险）**: 评分≥70，存在明显的串标围标特征
- **MEDIUM（中风险）**: 评分≥40，存在较多可疑特征
- **LOW（低风险）**: 评分≥15，存在少量可疑特征
- **NONE（无风险）**: 评分<15，未发现明显可疑特征

## 检测维度

系统从以下维度检测串标围标行为：

1. **文本相似度**: 
   - **全局相似度**：SBERT编码整个文档的语义向量，计算余弦相似度
   - **局部相似度**：SBERT批量编码段落（最多30×30），计算相似度矩阵
   - **语义理解**：能识别改写、同义词替换、句式变化等隐蔽抄袭
   - **高性能**：批量编码+矩阵运算，避免逐对计算
   - **自动降级**：SBERT不可用时自动切换到TF-IDF
   - **共同异常表述**：检测频繁出现的相同词组

2. **元数据关联**: 作者、创建软件、生成时间等元数据的匹配度

3. **报价规律**: 共同金额、尾数分布、固定差额/比例规律

4. **图片重复**: 嵌入图片的感知哈希（pHash）匹配

## 技术特点

- **高精度**: 全面使用SBERT语义向量，能精准识别改写、同义词替换等隐蔽抄袭
- **语义理解**: 深度学习模型理解文本含义，不依赖表面词汇匹配
- **批量优化**: 段落批量编码+矩阵运算，大幅提升计算效率
- **鲁棒性**: 自动降级机制，SBERT不可用时自动使用TF-IDF
- **多维度**: 文本、元数据、报价、图片多维度综合分析
- **可解释**: 生成详细的证据链，每个风险判定都有明确依据，清晰展示相似段落和内容
- **可扩展**: 模块化设计，易于扩展新的检测维度
- **可配置**: 所有阈值参数可通过配置文件调整

## 系统要求

- Python 3.7+
- 处理对象：普通文本型PDF（可直接提取字符）
- 不支持加密PDF和纯扫描版PDF的文本比对

## 注意事项

1. 扫描版PDF（无法提取文本）仅参与元数据和图片比对
2. 单文件大小建议不超过100MB
3. 文档数量N>500时，建议调整MAX_CANDIDATE_PAIRS参数
4. 首次运行会下载jieba分词词典

## 许可证

本项目仅供学习研究使用。

## 作者

BatchBidCollusionDetector Development Team
