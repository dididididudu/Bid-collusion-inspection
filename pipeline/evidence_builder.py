"""
共享证据构建模块 — 供 orchestrator 和 parallel_workers 共用

避免 ~200 行重复代码，确保文本/图片/元数据证据的一致性。
"""

import math
import logging
from typing import List, Dict

from config import DetectionConfig
from data_structures import (
    BidFeature, TextEvidence, MetadataEvidence, ImageEvidence,
)
from image_analysis.image_ocr import OCRResult
from image_analysis.image_matcher import ImageMatcher

logger = logging.getLogger(__name__)


# ================================================================
# 元数据证据
# ================================================================

def build_metadata_evidence(doc_a: BidFeature, doc_b: BidFeature) -> MetadataEvidence:
    """构建元数据证据"""
    evidence = MetadataEvidence()
    fields_to_check = ['author', 'creator', 'producer', 'software_fingerprint']
    for field in fields_to_check:
        val_a = getattr(doc_a.metadata, field, '').lower().strip()
        val_b = getattr(doc_b.metadata, field, '').lower().strip()
        if val_a and val_b and val_a == val_b:
            evidence.matched_fields.append(field)
            evidence.matched_values[field] = val_a
    if doc_a.metadata.time_bucket and doc_b.metadata.time_bucket:
        evidence.same_time_bucket = (
            doc_a.metadata.time_bucket == doc_b.metadata.time_bucket
        )
    # 文件码对比：同一源文件 → 极强串标证据
    if doc_a.metadata.file_id and doc_b.metadata.file_id:
        evidence.same_file_id = (doc_a.metadata.file_id == doc_b.metadata.file_id)
    return evidence


# ================================================================
# 联系人/公司雷同证据
# ================================================================

def build_contact_evidence(
    doc_a_id: str, doc_b_id: str, cache
) -> 'ContactEvidence':
    """对比两个文档的联系人/公司信息"""
    from data_structures import ContactEvidence
    evidence = ContactEvidence()

    fp_a = cache.load_contact_fingerprint(doc_a_id)
    fp_b = cache.load_contact_fingerprint(doc_b_id)
    if not fp_a or not fp_b:
        return evidence

    for key, target in [('company_names', 'common_companies'),
                         ('contact_names', 'common_contacts'),
                         ('mobile_phones', 'common_mobiles'),
                         ('emails', 'common_emails'),
                         ('credit_codes', 'common_credit_codes')]:
        set_a = set(fp_a.get(key, []))
        set_b = set(fp_b.get(key, []))
        common = list(set_a & set_b)
        if common:
            setattr(evidence, target, common)

    return evidence


# ================================================================
# 图片证据（四层检测）
# ================================================================

def build_image_evidence(
    doc_a: BidFeature,
    doc_b: BidFeature,
    cache,
    image_matcher: ImageMatcher = None,
    file_path_a: str = None,
    file_path_b: str = None,
    output_dir: str = None,
) -> ImageEvidence:
    """构建增强图片证据 — 四层检测（哈希 + OCR + 错字 + 文字相同）

    如果提供 file_path 和 output_dir，会将匹配的图片保存到磁盘，
    供 HTML 报告嵌入展示。图片数据不写入 JSON 报告。
    """
    evidence = ImageEvidence()

    if image_matcher is None:
        image_matcher = ImageMatcher()

    # 精确哈希匹配
    hashes_a = doc_a.image_hashes
    hashes_b = doc_b.image_hashes
    common_exact = list(set(hashes_a) & set(hashes_b))
    evidence.common_image_count = len(common_exact)
    evidence.common_image_hashes = common_exact

    # 如果提供了输出目录，保存匹配的图片
    matched_images = {}
    if output_dir and common_exact and (file_path_a or file_path_b):
        from image_analysis.image_exporter import find_matching_images_from_pdf
        if file_path_a and os.path.exists(file_path_a):
            imgs_a = find_matching_images_from_pdf(
                file_path_a, doc_a.doc_id, common_exact, output_dir,
            )
            matched_images['doc_a'] = imgs_a
        if file_path_b and os.path.exists(file_path_b):
            imgs_b = find_matching_images_from_pdf(
                file_path_b, doc_b.doc_id, common_exact, output_dir,
            )
            matched_images['doc_b'] = imgs_b
    evidence.matched_image_paths = matched_images

    # 加载 OCR 结果
    ocr_a = cache.load_image_ocr_results(doc_a.doc_id)
    ocr_b = cache.load_image_ocr_results(doc_b.doc_id)

    if ocr_a:
        evidence.ocr_results_a = ocr_a
    if ocr_b:
        evidence.ocr_results_b = ocr_b

    # 转换为 OCRResult 对象
    ocr_objects_a = [
        OCRResult(
            text=r['ocr_text'], words=r['ocr_words'],
            bboxes=r['bboxes'], confidence=r['confidence'],
            image_hash=r.get('image_hash', ''),
            non_text_hash=r.get('non_text_hash', ''),
            image_width=r.get('image_width', 0),
            image_height=r.get('image_height', 0),
        ) for r in ocr_a
    ]
    ocr_objects_b = [
        OCRResult(
            text=r['ocr_text'], words=r['ocr_words'],
            bboxes=r['bboxes'], confidence=r['confidence'],
            image_hash=r.get('image_hash', ''),
            non_text_hash=r.get('non_text_hash', ''),
            image_width=r.get('image_width', 0),
            image_height=r.get('image_height', 0),
        ) for r in ocr_b
    ]

    match_result = image_matcher.analyze(
        hashes_a=hashes_a,
        hashes_b=hashes_b,
        ocr_results_a=ocr_objects_a if ocr_objects_a else None,
        ocr_results_b=ocr_objects_b if ocr_objects_b else None,
    )

    # 填充增强字段
    evidence.exact_image_count = match_result.exact_image_count
    evidence.near_identical_count = match_result.near_identical_count
    evidence.similar_image_count = match_result.similar_image_count
    evidence.ps_suspicious = match_result.ps_suspicious
    evidence.ps_suspicious_count = match_result.ps_suspicious_count
    evidence.shared_typos = match_result.shared_typos
    evidence.shared_typo_count = match_result.shared_typo_count
    evidence.text_identical_count = match_result.text_identical_count
    evidence.text_similar_count = match_result.text_similar_count
    evidence.image_risk_score = match_result.image_risk_score
    evidence.image_risk_factors = match_result.image_risk_factors

    # === 填充逐对详情 ===
    # L1: 图片匹配对
    for v in match_result.image_verdicts:
        thumb_a_b64 = _thumbnail_to_base64(v.sig_a.thumbnail)
        thumb_b_b64 = _thumbnail_to_base64(v.sig_b.thumbnail)
        # 从 source_id 获取 OCR 文本
        ocr_text_a = _find_ocr_text_by_hash(ocr_objects_a, v.sig_a.phash or v.sig_a.dhash)
        ocr_text_b = _find_ocr_text_by_hash(ocr_objects_b, v.sig_b.phash or v.sig_b.dhash)
        evidence.matched_image_pairs.append({
            'source_a': v.sig_a.source_id,
            'source_b': v.sig_b.source_id,
            'phash_dist': v.phash_dist,
            'dhash_dist': v.dhash_dist,
            'orb_match_ratio': round(v.orb_match_ratio, 3),
            'histogram_correlation': round(v.histogram_correlation, 3),
            'confidence': round(v.confidence, 3),
            'reasons': v.reasons,
            'thumbnail_base64_a': thumb_a_b64,
            'thumbnail_base64_b': thumb_b_b64,
            'ocr_text_a': ocr_text_a,
            'ocr_text_b': ocr_text_b,
            'l1_pass': v.l1_pass,
            'l2_pass': v.l2_pass,
            'l3_pass': v.l3_pass,
        })

    # L2: 文字匹配对
    for t in match_result.text_matches:
        evidence.matched_text_pairs.append(t)

    # L4: PS 嫌疑详情
    for p in match_result.ps_details:
        evidence.ps_detail_list.append(p)

    return evidence



def _thumbnail_to_base64(thumb_bytes: bytes) -> str:
    """将缩略图字节转为 base64 data URI"""
    if not thumb_bytes:
        return ''
    import base64
    encoded = base64.b64encode(thumb_bytes).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded}"


def _find_ocr_text_by_hash(ocr_objects: list, hash_val: str) -> str:
    """根据哈希值在 OCR 结果列表中找对应文本

    兼容 OCRResult 对象和字典两种格式。
    """
    if not hash_val:
        return ''
    for obj in ocr_objects:
        if hasattr(obj, 'image_hash'):
            img_hash = obj.image_hash
            text = obj.text
        else:
            img_hash = obj.get('image_hash', '')
            text = obj.get('ocr_text', '')
        if img_hash and (hash_val in img_hash or img_hash in hash_val):
            return text[:200]
    return ''


# ================================================================
# 文本证据
# ================================================================

def build_text_evidence(
    doc_a: BidFeature,
    doc_b: BidFeature,
    paragraph_matches: List[Dict],
    config: DetectionConfig,
    compute_highlight: bool = False,
) -> TextEvidence:
    """构建文本证据

    Args:
        doc_a, doc_b: 文档特征
        paragraph_matches: 匹配的段落列表
        config: 检测配置
        compute_highlight: 是否计算差异高亮（计算量大，worker 中跳过）
    """
    evidence = TextEvidence()
    evidence.paragraph_matches = paragraph_matches

    if not paragraph_matches:
        return evidence

    similarities = [m['similarity'] for m in paragraph_matches]
    max_sim = max(similarities) if similarities else 0.0

    top_k = min(config.SCORE_TOP_K, len(similarities))
    top_k_similarities = sorted(similarities, reverse=True)[:top_k]
    top_k_sim = sum(top_k_similarities) / top_k if top_k_similarities else 0.0

    weighted_sum = sum(s * s for s in similarities)
    weighted_mean = weighted_sum / sum(similarities) if sum(similarities) > 0 else 0.0

    quality_score = 0.50 * max_sim + 0.35 * top_k_sim + 0.15 * weighted_mean

    # 覆盖率分数
    covered_a = len(set(m['paragraph_a_index'] for m in paragraph_matches))
    covered_b = len(set(m['paragraph_b_index'] for m in paragraph_matches))
    estimated_total = max(1, covered_a + covered_b)
    coverage_ratio = (covered_a + covered_b) / (estimated_total * 2) if estimated_total > 0 else 0
    coverage_score = 1.0 - math.exp(-4 * coverage_ratio) if coverage_ratio > 0 else 0.0

    # 一致性分数
    sorted_by_a = sorted(paragraph_matches, key=lambda x: x['paragraph_a_index'])
    consecutive = sum(
        1 for k in range(1, len(sorted_by_a))
        if (sorted_by_a[k]['paragraph_a_index'] - sorted_by_a[k - 1]['paragraph_a_index'] == 1 and
            sorted_by_a[k]['paragraph_b_index'] - sorted_by_a[k - 1]['paragraph_b_index'] == 1)
    )
    consistency_score = min(1.0, 0.5 + consecutive * 0.01) if consecutive >= 3 else 0.5

    evidence.local_similarity = min(
        1.0,
        0.60 * quality_score + 0.25 * coverage_score + 0.15 * consistency_score
    )

    # 检测连续克隆块
    clone_blocks = _detect_clone_blocks(paragraph_matches, config)
    evidence.continuous_clone_blocks = clone_blocks

    # 更新克隆标记
    clone_index = {}
    for block in clone_blocks:
        for pair in block['pairs']:
            clone_index[(pair['a_index'], pair['b_index'])] = {
                'is_clone': True, 'group_id': block['group_id']
            }
    for match in paragraph_matches:
        key = (match['paragraph_a_index'], match['paragraph_b_index'])
        if key in clone_index:
            match['is_continuous_clone'] = True
            match['continuous_clone_group_id'] = clone_index[key]['group_id']

    evidence.detection_summary = {
        'sbert_match_count': len(paragraph_matches),
        'continuous_clone_block_count': len(clone_blocks),
    }

    evidence.common_paragraphs = [
        m.get('paragraph_a', '')[:200] for m in paragraph_matches[:5]
    ]

    # 差异高亮（仅在需要时计算）
    if compute_highlight:
        from utils.text_diff import compute_text_diff
        for match in paragraph_matches:
            text_a = match.get('paragraph_a', '')
            text_b = match.get('paragraph_b', '')
            if text_a and text_b:
                highlighted_a, highlighted_b, common_parts = compute_text_diff(text_a, text_b)
                match['highlighted_text_a'] = highlighted_a
                match['highlighted_text_b'] = highlighted_b
                match['common_parts'] = common_parts

    return evidence


# ================================================================
# 连续克隆块检测
# ================================================================

def _detect_clone_blocks(
    paragraph_matches: List[Dict],
    config: DetectionConfig,
) -> List[Dict]:
    """检测连续克隆块"""
    min_len = config.CLONE_BLOCK_MIN_LENGTH
    max_gap = config.CLONE_BLOCK_MAX_GAP

    if len(paragraph_matches) < min_len:
        return []

    matches_sorted = sorted(
        paragraph_matches,
        key=lambda x: (x['paragraph_a_index'], x['paragraph_b_index'])
    )

    blocks = []
    current_block = []
    group_id = 0

    for match in matches_sorted:
        if not current_block:
            current_block.append(match)
        else:
            last = current_block[-1]
            a_gap = match['paragraph_a_index'] - last['paragraph_a_index'] - 1
            b_gap = match['paragraph_b_index'] - last['paragraph_b_index'] - 1

            if a_gap <= max_gap and b_gap <= max_gap:
                current_block.append(match)
            else:
                if len(current_block) >= min_len:
                    blocks.append({
                        'group_id': f'clone_block_{group_id}',
                        'pairs': [
                            {'a_index': m['paragraph_a_index'],
                             'b_index': m['paragraph_b_index']}
                            for m in current_block
                        ],
                        'similarity': sum(m['similarity'] for m in current_block) / len(current_block),
                        'length': len(current_block),
                    })
                    group_id += 1
                current_block = [match]

    if len(current_block) >= min_len:
        blocks.append({
            'group_id': f'clone_block_{group_id}',
            'pairs': [
                {'a_index': m['paragraph_a_index'],
                 'b_index': m['paragraph_b_index']}
                for m in current_block
            ],
            'similarity': sum(m['similarity'] for m in current_block) / len(current_block),
            'length': len(current_block),
        })

    return blocks
