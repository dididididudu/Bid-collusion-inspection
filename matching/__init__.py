"""
matching - 文档相似度比对模块

子模块:
- lsh_index: datasketch MinHash LSH 封装（文档级 + 段落级）
- selector: 候选文档对选择器
- paragraph_matcher: 两阶段段落匹配引擎
- semantic_matcher: GPU/ONNX SBERT 推理引擎
"""
