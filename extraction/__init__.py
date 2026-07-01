"""
extraction - 增量 PDF 处理模块

子模块:
- base: 提取器抽象基类
- pdf_extractor: PyMuPDF (fitz) 高速 PDF 解析
- text_processor: 分块分词 + 聚合哈希
- feature_cache: SQLite 持久化特征存储
"""
