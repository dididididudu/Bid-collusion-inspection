"""
模块 C：精细相似度计算引擎
"""
import logging
import re
import math
from typing import Dict, List, Tuple
from difflib import SequenceMatcher
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

from data_structures import (
    BidFeature, PairwiseResult, EvidenceChain,
    TextEvidence, MetadataEvidence, ImageEvidence
)
from config import DetectionConfig

logger = logging.getLogger(__name__)


class PairwiseAnalyzer:
    """文档对精细分析器"""

    def __init__(self, config: DetectionConfig):
        self.config = config
        self._sbert_model = None
        
        self.paragraph_embedding_cache = {}
        self.paragraph_cache = {}
    
    @property
    def sbert_model(self):
        if self._sbert_model is None and SBERT_AVAILABLE:
            logger.info("正在加载SBERT模型...")
            try:
                import os
                os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
                
                self._sbert_model = SentenceTransformer(
                    'paraphrase-multilingual-MiniLM-L12-v2',
                    device='cpu',
                    cache_folder='./models',
                    trust_remote_code=True,
                    local_files_only=True
                )
                logger.info("SBERT模型加载完成")
            except Exception as e:
                logger.warning(f"SBERT模型加载失败: {e}，将使用基础方法")
                self._sbert_model = None
        return self._sbert_model

    def analyze(self, doc_a: BidFeature, doc_b: BidFeature) -> PairwiseResult:
        pair_id = "::".join(sorted([doc_a.doc_id, doc_b.doc_id]))

        evidence = EvidenceChain()
        similarity_scores = {}

        if not doc_a.is_scanned and not doc_b.is_scanned:
            text_evidence = self._analyze_text_similarity(doc_a, doc_b)
            evidence.text_evidence = text_evidence
            similarity_scores['text_local'] = text_evidence.local_similarity
        else:
            similarity_scores['text_local'] = 0.0

        metadata_evidence = self._analyze_metadata_similarity(doc_a, doc_b)
        evidence.metadata_evidence = metadata_evidence
        similarity_scores['metadata_match'] = len(metadata_evidence.matched_fields)

        image_evidence = self._analyze_image_similarity(doc_a, doc_b)
        evidence.image_evidence = image_evidence
        similarity_scores['image_common'] = image_evidence.common_image_count

        return PairwiseResult(
            pair_id=pair_id,
            doc_a_id=doc_a.doc_id,
            doc_b_id=doc_b.doc_id,
            similarity_scores=similarity_scores,
            evidence=evidence
        )

    def _analyze_text_similarity(self, doc_a: BidFeature, doc_b: BidFeature) -> TextEvidence:
        evidence = TextEvidence()

        content_a = self._extract_substantive_content(doc_a.text_content)
        content_b = self._extract_substantive_content(doc_b.text_content)

        if doc_a.doc_id not in self.paragraph_cache:
            self.paragraph_cache[doc_a.doc_id] = self._split_paragraphs(content_a)
        if doc_b.doc_id not in self.paragraph_cache:
            self.paragraph_cache[doc_b.doc_id] = self._split_paragraphs(content_b)
        
        paras_a = self.paragraph_cache[doc_a.doc_id]
        paras_b = self.paragraph_cache[doc_b.doc_id]
        
        if len(paras_a) > 0 and len(paras_b) > 0:
            sbert_candidates = []
            seq_matcher_matches = []
            
            # === 并行化段落比对 ===
            seq_matcher_matches, sbert_candidates = self._parallel_sequence_matcher(paras_a, paras_b)
            
            # === SBERT验证（动态阈值）===
            sbert_matches = []
            if self.sbert_model is not None and sbert_candidates:
                try:
                    sbert_matches = self._sbert_verify_with_dynamic_threshold(sbert_candidates)
                except Exception as e:
                    logger.error(f"SBERT验证失败: {e}", exc_info=True)
            
            # === 合并所有匹配结果 ===
            paragraph_matches = seq_matcher_matches + sbert_matches
            
            # === 检测连续克隆块 ===
            continuous_clone_blocks = self._detect_continuous_clone_blocks(paragraph_matches)
            
            # === 使用字典索引更新连续克隆标记（替代三重循环）===
            self._update_clone_marks_with_dict(paragraph_matches, continuous_clone_blocks)
            
            evidence.continuous_clone_blocks = continuous_clone_blocks
            
            detection_summary = {
                'sequence_matcher_count': len(seq_matcher_matches),
                'sbert_count': len(sbert_matches),
                'continuous_clone_block_count': len(continuous_clone_blocks)
            }
            evidence.detection_summary = detection_summary
            
            # === 混合评分策略 ===
            evidence.local_similarity = self._calculate_mixed_score(paragraph_matches, paras_a, paras_b)
        else:
            evidence.local_similarity = 0.0

        paragraph_matches.sort(key=lambda x: x['similarity'], reverse=True)
        evidence.paragraph_matches = paragraph_matches
        
        if paragraph_matches:
            evidence.common_paragraphs = [m['paragraph_a'] for m in paragraph_matches[:3]]

        return evidence

    def _build_match_info(self, similarity: float, para_a: str, para_b: str, 
                          idx_a: int, idx_b: int, method: str) -> dict:
        """构建匹配信息（消除重复代码）"""
        highlighted_a, highlighted_b = self._highlight_repeated_parts(para_a, para_b)
        return {
            'similarity': similarity,
            'paragraph_a': para_a[:500],
            'paragraph_b': para_b[:500],
            'paragraph_a_index': idx_a,
            'paragraph_b_index': idx_b,
            'detection_method': method,
            'is_continuous_clone': False,
            'continuous_clone_group_id': '',
            'highlighted_text_a': highlighted_a[:500] if highlighted_a else '',
            'highlighted_text_b': highlighted_b[:500] if highlighted_b else ''
        }

    def _parallel_sequence_matcher(self, paras_a: List[str], paras_b: List[str]) -> tuple:
        """并行化SequenceMatcher预过滤"""
        seq_matcher_matches = []
        sbert_candidates = []
        
        def process_pair(args):
            i, j, para_a, para_b = args
            seq_sim = self._sequence_matcher_similarity(para_a, para_b)
            
            if seq_sim >= 0.85:
                return ('match', i, j, seq_sim, para_a, para_b)
            elif 0.4 <= seq_sim < 0.85:
                return ('candidate', i, j, para_a, para_b)
            return None
        
        tasks = []
        for i, para_a in enumerate(paras_a):
            for j, para_b in enumerate(paras_b):
                if len(para_a) > 50 and len(para_b) > 50:
                    tasks.append((i, j, para_a, para_b))
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_task = {executor.submit(process_pair, task): task for task in tasks}
            for future in as_completed(future_to_task):
                result = future.result()
                if result:
                    if result[0] == 'match':
                        _, i, j, seq_sim, para_a, para_b = result
                        seq_matcher_matches.append(
                            self._build_match_info(seq_sim, para_a, para_b, i, j, 'SequenceMatcher')
                        )
                    elif result[0] == 'candidate':
                        _, i, j, para_a, para_b = result
                        sbert_candidates.append((i, j, para_a, para_b))
        
        return seq_matcher_matches, sbert_candidates

    def _sbert_verify_with_dynamic_threshold(self, sbert_candidates: list) -> list:
        """SBERT验证（动态阈值策略）"""
        sbert_matches = []
        
        candidate_paras_a = [c[2] for c in sbert_candidates]
        candidate_paras_b = [c[3] for c in sbert_candidates]
        
        embeddings_a = self.sbert_model.encode(candidate_paras_a, batch_size=32, show_progress_bar=False)
        embeddings_b = self.sbert_model.encode(candidate_paras_b, batch_size=32, show_progress_bar=False)
        
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        similarity_matrix = cos_sim(embeddings_a, embeddings_b)
        
        for idx, (i, j, para_a, para_b) in enumerate(sbert_candidates):
            sbert_sim = float(similarity_matrix[idx][idx])
            
            # 动态阈值：短段落需要更高阈值
            avg_length = (len(para_a) + len(para_b)) / 2
            if avg_length < self.config.SBERT_SHORT_PARAGRAPH_LEN:
                threshold = self.config.SBERT_SHORT_PARAGRAPH_THRESHOLD
            else:
                threshold = self.config.SBERT_BASE_THRESHOLD
            
            # 根据候选数量微调阈值
            if len(sbert_candidates) > 100:
                threshold = min(0.80, threshold + 0.02)
            elif len(sbert_candidates) < 10:
                threshold = max(0.70, threshold - 0.02)
            
            if sbert_sim >= threshold:
                sbert_matches.append(
                    self._build_match_info(sbert_sim, para_a, para_b, i, j, 'SBERT')
                )
        
        return sbert_matches

    def _calculate_mixed_score(self, paragraph_matches: list, paras_a: list, paras_b: list) -> float:
        """混合评分策略（乘法模型）：quality_score × coverage_factor"""
        if not paragraph_matches:
            return 0.0
        
        similarities = [m['similarity'] for m in paragraph_matches]
        
        max_sim = max(similarities)
        
        top_k = min(self.config.SCORE_TOP_K, len(similarities))
        top_k_similarities = sorted(similarities, reverse=True)[:top_k]
        top_k_sim = sum(top_k_similarities) / top_k
        
        mean_sim = sum(similarities) / len(similarities)
        
        total_paras = len(paras_a) + len(paras_b)
        covered_paras_a = len(set(m['paragraph_a_index'] for m in paragraph_matches))
        covered_paras_b = len(set(m['paragraph_b_index'] for m in paragraph_matches))
        coverage = (covered_paras_a + covered_paras_b) / total_paras if total_paras > 0 else 0
        
        quality_score = (
            self.config.SCORE_WEIGHT_MAX * max_sim +
            self.config.SCORE_WEIGHT_TOP_K * top_k_sim +
            self.config.SCORE_WEIGHT_MEAN * mean_sim
        )
        
        coverage_factor = 1 - math.exp(-5 * coverage) if coverage > 0 else 0
        
        mixed_score = quality_score * coverage_factor
        
        return mixed_score

    def _update_clone_marks_with_dict(self, paragraph_matches: list, continuous_clone_blocks: list):
        """使用字典索引更新连续克隆标记"""
        clone_index = {}
        for block in continuous_clone_blocks:
            for pair in block['pairs']:
                key = (pair['a_index'], pair['b_index'])
                clone_index[key] = {
                    'is_clone': True,
                    'group_id': block['group_id']
                }
        
        for match in paragraph_matches:
            key = (match['paragraph_a_index'], match['paragraph_b_index'])
            if key in clone_index:
                match['is_continuous_clone'] = True
                match['continuous_clone_group_id'] = clone_index[key]['group_id']

    def _sequence_matcher_similarity(self, text_a: str, text_b: str) -> float:
        matcher = SequenceMatcher(None, text_a, text_b)
        return matcher.ratio()

    def _highlight_repeated_parts(self, text_a: str, text_b: str) -> tuple:
        """使用字符串切片替代逐字符操作，提升效率"""
        matcher = SequenceMatcher(None, text_a, text_b)
        
        parts_a = []
        parts_b = []
        
        last_end_a = 0
        last_end_b = 0
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal' and i2 - i1 >= 10:
                if i1 > last_end_a:
                    parts_a.append(text_a[last_end_a:i1])
                parts_a.append('【重复】' + text_a[i1:i2] + '【/重复】')
                last_end_a = i2
                
                if j1 > last_end_b:
                    parts_b.append(text_b[last_end_b:j1])
                parts_b.append('【重复】' + text_b[j1:j2] + '【/重复】')
                last_end_b = j2
            else:
                if i1 > last_end_a:
                    parts_a.append(text_a[last_end_a:i1])
                last_end_a = i2
                
                if j1 > last_end_b:
                    parts_b.append(text_b[last_end_b:j1])
                last_end_b = j2
        
        if last_end_a < len(text_a):
            parts_a.append(text_a[last_end_a:])
        if last_end_b < len(text_b):
            parts_b.append(text_b[last_end_b:])
        
        return ''.join(parts_a), ''.join(parts_b)

    def _detect_continuous_clone_blocks(self, paragraph_matches: list) -> list:
        """检测连续克隆块（支持最大间隔）"""
        min_length = self.config.CLONE_BLOCK_MIN_LENGTH
        max_gap = self.config.CLONE_BLOCK_MAX_GAP
        
        if len(paragraph_matches) < min_length:
            return []
        
        matches_sorted = sorted(paragraph_matches, key=lambda x: (x['paragraph_a_index'], x['paragraph_b_index']))
        
        blocks = []
        current_block = []
        group_id_counter = 0
        
        for match in matches_sorted:
            if not current_block:
                current_block.append(match)
            else:
                last_match = current_block[-1]
                a_gap = match['paragraph_a_index'] - last_match['paragraph_a_index'] - 1
                b_gap = match['paragraph_b_index'] - last_match['paragraph_b_index'] - 1
                
                if a_gap <= max_gap and b_gap <= max_gap:
                    current_block.append(match)
                else:
                    if len(current_block) >= min_length:
                        blocks.append(self._build_clone_block(current_block, group_id_counter))
                        group_id_counter += 1
                    current_block = [match]
        
        if len(current_block) >= min_length:
            blocks.append(self._build_clone_block(current_block, group_id_counter))
        
        return blocks
    
    def _build_clone_block(self, matches: list, group_id: int) -> dict:
        """构建克隆块信息"""
        return {
            'group_id': f'clone_block_{group_id}',
            'pairs': [{'a_index': m['paragraph_a_index'], 'b_index': m['paragraph_b_index']} 
                      for m in matches],
            'similarity': sum(m['similarity'] for m in matches) / len(matches),
            'length': len(matches)
        }

    def _extract_substantive_content(self, text: str) -> str:
        """提取实质性内容（优化：移除噪声行，如目录虚线、页码等）"""
        if len(text) > 50000:
            text = text[:50000]
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if re.search(r'[\.。…·\-_]{5,}', line):
                continue
            
            if re.match(r'^\d{1,4}$', line):
                continue
            
            if len(line) < 20 and not re.search(r'[\u4e00-\u9fff]', line):
                if re.match(r'^[\s\d\W]+$', line):
                    continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def _split_paragraphs(self, text: str) -> List[str]:
        """分割段落 - 增强版：添加标题识别、列表识别等规则"""
        # 预处理：移除多余空白
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 方法1：尝试用双换行符分割
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 50]
        
        # 如果分割出的段落太少，尝试其他方法
        if len(paragraphs) <= 3:
            # 方法2：智能分割（识别标题和列表）
            paragraphs = self._smart_paragraph_split(text)
        
        # 方法3：如果仍然段落太少，按固定长度分割
        if len(paragraphs) <= 3 and len(text) > 500:
            paragraphs = []
            for i in range(0, len(text), 500):
                segment = text[i:i+500].strip()
                if len(segment) > 100:
                    paragraphs.append(segment)
        
        # 方法4：按标点符号分割
        if len(paragraphs) <= 3 and len(text) > 1000:
            sentences = re.split(r'[。？！；;]', text)
            current_chunk = ""
            paragraphs = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 10:
                    continue
                    
                if len(current_chunk) + len(sentence) < 300:
                    current_chunk += sentence + "。"
                else:
                    if len(current_chunk) > 50:
                        paragraphs.append(current_chunk.strip())
                    current_chunk = sentence + "。"
            
            if len(current_chunk.strip()) > 50:
                paragraphs.append(current_chunk.strip())
        
        # 最终过滤
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 50]
        
        if not paragraphs and len(text.strip()) > 50:
            paragraphs = [text.strip()]
        
        return paragraphs
    
    def _smart_paragraph_split(self, text: str) -> List[str]:
        """智能段落分割：识别标题、列表、表格等结构"""
        lines = text.split('\n')
        paragraphs = []
        current_paragraph = ""
        
        for line in lines:
            line = line.strip()
            
            if not line:
                if current_paragraph:
                    paragraphs.append(current_paragraph.strip())
                    current_paragraph = ""
                continue
            
            # 识别标题（通常较短，可能以数字/序号开头）
            if self._is_title(line):
                if current_paragraph:
                    paragraphs.append(current_paragraph.strip())
                # 标题作为段落的开头
                current_paragraph = line
                continue
            
            # 识别列表项（以数字、字母、符号开头）
            if self._is_list_item(line):
                if current_paragraph and len(current_paragraph) > 200:
                    paragraphs.append(current_paragraph.strip())
                    current_paragraph = line
                elif current_paragraph:
                    current_paragraph += " " + line
                else:
                    current_paragraph = line
                continue
            
            # 普通文本行
            if current_paragraph:
                if len(current_paragraph) > 300:
                    paragraphs.append(current_paragraph.strip())
                    current_paragraph = line
                else:
                    current_paragraph += " " + line
            else:
                current_paragraph = line
        
        if current_paragraph and len(current_paragraph.strip()) > 50:
            paragraphs.append(current_paragraph.strip())
        
        return paragraphs
    
    def _is_title(self, line: str) -> bool:
        """判断是否为标题"""
        # 标题通常较短
        if len(line) > 100:
            return False
        
        # 以章节号开头（如"1."、"1.1"、"一、"、"第一章"等）
        title_patterns = [
            r'^[一二三四五六七八九十]+[、．.．]',
            r'^第[一二三四五六七八九十]+[章节部分]',
            r'^\d+[\.．]\s*',
            r'^\d+[\.．]\d+[\.．]?\s*',
            r'^[（(]\d+[)）]\s*',
            r'^[ABCDEFGHIJKLMNOPQRSTUVWXYZ][、．.]\s*',
        ]
        
        for pattern in title_patterns:
            if re.match(pattern, line):
                return True
        
        return False
    
    def _is_list_item(self, line: str) -> bool:
        """判断是否为列表项"""
        list_patterns = [
            r'^[\-—–*•●○□△]+\s*',
            r'^\d+[\.．、)]+\s*',
            r'^[（(]\d+[)）]\s*',
            r'^[①②③④⑤⑥⑦⑧⑨⑩]+',
            r'^[a-z][、．)]+\s*',
            r'^[A-Z][、．)]+\s*',
        ]
        
        for pattern in list_patterns:
            if re.match(pattern, line):
                return True
        
        return False

    def _segment_sampling(self, text: str, max_segments: int = 5, segment_length: int = 1000) -> List[str]:
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 50]
        
        if not paragraphs:
            return [text[:segment_length]]
        
        num_paragraphs = len(paragraphs)
        if num_paragraphs <= max_segments:
            return paragraphs
        
        interval = num_paragraphs // max_segments
        segments = []
        
        for i in range(max_segments):
            idx = i * interval
            if idx < num_paragraphs:
                segment = paragraphs[idx]
                while len(segment) < segment_length and idx + 1 < num_paragraphs:
                    idx += 1
                    segment += '\n' + paragraphs[idx]
                segments.append(segment[:segment_length])
        
        return segments if segments else [text[:segment_length]]

    def _fallback_tfidf_similarity(self, text_a: str, text_b: str) -> float:
        try:
            vectorizer = TfidfVectorizer(
                tokenizer=jieba.lcut,
                max_features=5000,
                ngram_range=(1, 2),
                use_idf=True
            )
            tfidf_matrix = vectorizer.fit_transform([text_a, text_b])
            tfidf_sim = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
            return tfidf_sim
        except Exception as e:
            logger.error(f"TF-IDF降级方案失败: {e}")
            return 0.0
    
    def _minhash_similarity(self, hashes_a: List[str], hashes_b: List[str]) -> float:
        if not hashes_a or not hashes_b:
            return 0.0
        
        set_a = set(hashes_a)
        set_b = set(hashes_b)
        
        if not set_a or not set_b:
            return 0.0
        
        intersection = set_a & set_b
        union = set_a | set_b
        
        return len(intersection) / len(union) if union else 0.0
    
    def _fallback_jaccard_paragraphs(self, paras_a: List[str], paras_b: List[str]) -> List[Tuple[float, str]]:
        similar_pairs = []
        
        sample_a = paras_a[:20] if len(paras_a) > 20 else paras_a
        sample_b = paras_b[:20] if len(paras_b) > 20 else paras_b
        
        for para_a in sample_a:
            for para_b in sample_b:
                if len(para_a) > 50 and len(para_b) > 50:
                    jaccard_sim = self._jaccard_similarity(para_a, para_b)
                    if jaccard_sim > 0.85:
                        similar_pairs.append((jaccard_sim, para_a))
        
        return similar_pairs

    def _jaccard_similarity(self, text_a: str, text_b: str) -> float:
        words_a = set(jieba.cut(text_a))
        words_b = set(jieba.cut(text_b))

        if not words_a or not words_b:
            return 0.0

        intersection = words_a & words_b
        union = words_a | words_b

        return len(intersection) / len(union) if union else 0.0

    def _analyze_metadata_similarity(self, doc_a: BidFeature, doc_b: BidFeature) -> MetadataEvidence:
        evidence = MetadataEvidence()

        meta_a = doc_a.metadata
        meta_b = doc_b.metadata

        fields_to_check = ['author', 'creator', 'producer', 'software_fingerprint']
        matched_fields = []
        matched_values = {}

        for field in fields_to_check:
            val_a = getattr(meta_a, field, '').lower().strip()
            val_b = getattr(meta_b, field, '').lower().strip()

            if val_a and val_b and val_a == val_b:
                matched_fields.append(field)
                matched_values[field] = val_a

        evidence.matched_fields = matched_fields
        evidence.matched_values = matched_values

        if meta_a.time_bucket and meta_b.time_bucket:
            evidence.same_time_bucket = (meta_a.time_bucket == meta_b.time_bucket)

        return evidence

    def _analyze_image_similarity(self, doc_a: BidFeature, doc_b: BidFeature) -> ImageEvidence:
        evidence = ImageEvidence()

        hashes_a = set(doc_a.image_hashes)
        hashes_b = set(doc_b.image_hashes)

        common_hashes = list(hashes_a & hashes_b)
        evidence.common_image_count = len(common_hashes)
        evidence.common_image_hashes = common_hashes

        return evidence