"""
模块 D：风险评级与聚类引擎
"""
import logging
from typing import List, Dict, Set
from collections import defaultdict, deque
import uuid

# 风险等级排序常量，避免每次调用时重建
_RISK_ORDER = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}

from data_structures import (
    BidFeature, PairwiseResult, GlobalReport,
    Cluster, FileProfile
)
from config import DetectionConfig

logger = logging.getLogger(__name__)


class RiskScoringEngine:
    """风险评分引擎"""

    def __init__(self, config: DetectionConfig):
        self.config = config

    def generate_report(
        self,
        pairwise_results: List[PairwiseResult],
        features: List[BidFeature]
    ) -> GlobalReport:
        """生成全局检测报告"""

        # 1. 对每个文档对进行风险评分
        scored_results = []
        for result in pairwise_results:
            scored_result = self._score_pair(result)
            scored_results.append(scored_result)

        # 2. 统计信息
        suspicious_pairs = [r for r in scored_results if r.risk_level != "NONE"]
        high_risk_pairs = [r for r in scored_results if r.risk_level == "HIGH"]

        # 3. 风险聚类
        risk_clusters = self._cluster_risks(scored_results, features)

        # 4. 生成单文档画像
        file_profiles = self._generate_file_profiles(scored_results, features)

        # 5. 生成报告
        from datetime import datetime
        report = GlobalReport(
            report_id=str(uuid.uuid4()),
            generated_at=datetime.now().isoformat(),
            total_files=len(features),
            total_pairs=len(features) * (len(features) - 1) // 2,
            candidate_pairs=len(pairwise_results),
            suspicious_pairs=len(suspicious_pairs),
            high_risk_pairs=len(high_risk_pairs),
            risk_clusters=risk_clusters,
            pairwise_results=scored_results,
            file_profiles=file_profiles
        )

        logger.info(f"报告生成完成: {len(suspicious_pairs)} 对可疑, {len(high_risk_pairs)} 对高风险")
        return report

    def _score_pair(self, result: PairwiseResult) -> PairwiseResult:
        """对单个文档对进行风险评分（优化版：增加覆盖率门限检查）"""
        scores = result.similarity_scores
        evidence = result.evidence

        score = 0
        risk_factors = []

        # 文件码相同 → 极强串标证据（同一源文件直接派生）
        if evidence.metadata_evidence.same_file_id:
            score += 40
            risk_factors.append("文件码相同: 两份PDF从同一源文件生成 (PDF /ID[0] 匹配)")

        text_local = scores.get('text_local', 0)
        paragraph_matches = evidence.text_evidence.paragraph_matches
        continuous_clone_blocks = evidence.text_evidence.continuous_clone_blocks
        detection_summary = evidence.text_evidence.detection_summary

        match_count = len(paragraph_matches)
        coverage = 0.0
        if match_count > 0:
            covered_a = len(set(m['paragraph_a_index'] for m in paragraph_matches))
            covered_b = len(set(m['paragraph_b_index'] for m in paragraph_matches))
            # 改进覆盖率计算：使用实际可能的总段落数
            total_covered = covered_a + covered_b
            coverage = min(1.0, total_covered / 50.0)  # 归一化到0-1

        # 计算风险分数（文本 70 + 图片 30 = 0-100）
        max_para_sim = max((m['similarity'] for m in paragraph_matches), default=0)
        text_score = int(text_local * 60 + max_para_sim * 30 + coverage * 10)
        text_score = min(70, max(0, text_score))

        # 图片维度评分（来自四层检测结果）
        image_evidence = evidence.image_evidence
        image_score = min(30, image_evidence.image_risk_score)

        score = min(100, text_score + image_score)

        # 图片风险因素
        for img_factor in image_evidence.image_risk_factors:
            risk_factors.append(f"📷 {img_factor}")

        # 改进的风险评级逻辑
        clone_count = len(continuous_clone_blocks)

        if text_local >= 0.70 or max_para_sim >= 0.85:
            # 高风险条件
            if match_count >= 5 or clone_count >= 2 or max_para_sim >= 0.90:
                risk_level = "HIGH"

                if detection_summary.get('sbert_match_count', 0) > 0:
                    risk_factors.append(f"✓ SBERT验证通过 ({detection_summary['sbert_match_count']}对匹配)")

                if clone_count > 0:
                    for block in continuous_clone_blocks:
                        risk_factors.append(f"⚠️ 发现连续克隆块(长度{block['length']}, 相似度{block['similarity']:.4f})")

                risk_factors.append(f"文本相似度: {text_local:.4f}")
                risk_factors.append(f"匹配段落数: {match_count}, 最高单段相似度: {max_para_sim:.4f}")
            else:
                risk_level = "MEDIUM"
                risk_factors.append(f"⚠️ 匹配段落数({match_count})较少，降级为中等风险")
                risk_factors.append(f"文本相似度: {text_local:.4f}")
        elif text_local >= 0.40 or match_count >= 3:
            risk_level = "MEDIUM"
            if paragraph_matches:
                risk_factors.append(f"存在相似段落 ({match_count}对, 最高相似度{max_para_sim:.4f})")
            risk_factors.append(f"文本相似度: {text_local:.4f}")
        elif text_local >= 0.20 or match_count >= 1:
            risk_level = "LOW"
            if paragraph_matches:
                risk_factors.append(f"存在低相似段落 ({match_count}对)")
            risk_factors.append(f"文本相似度: {text_local:.4f}")
        else:
            risk_level = "LOW"
            risk_factors.append("未发现显著相似段落")

        # === 图片风险上调：图片证据独立影响风险等级 ===
        if image_score >= 20:
            # 强图片证据 → 上调一级
            if risk_level == "NONE":
                risk_level = "LOW"
                risk_factors.insert(0,
                    f"图片证据较强（风险分{image_score}/30），生成低风险标记"
                )
            elif risk_level == "LOW":
                risk_level = "MEDIUM"
                risk_factors.insert(0,
                    f"图片证据叠加（风险分{image_score}/30），风险等级上调至MEDIUM"
                )
            elif risk_level == "MEDIUM":
                risk_level = "HIGH"
                risk_factors.insert(0,
                    f"图片证据强烈（风险分{image_score}/30），风险等级上调至HIGH"
                )
        elif image_score >= 10:
            # 中等图片证据 → NONE 升级为 LOW
            if risk_level == "NONE":
                risk_level = "LOW"
                risk_factors.insert(0,
                    f"存在图片雷同线索（风险分{image_score}/30）"
                )

        result.risk_score = score
        result.risk_level = risk_level
        result.risk_factors = risk_factors

        return result

    def _cluster_risks(
        self,
        pairwise_results: List[PairwiseResult],
        features: List[BidFeature]
    ) -> List[Cluster]:
        """风险聚类 - 发现围标团伙"""
        clusters = []

        # 构建图结构：节点为文档，边为风险等级 >= LOW 的文档对
        graph = defaultdict(set)

        # 同时构建 pair_id → result 字典，避免后续 O(n) 查找
        pair_lookup = {r.pair_id: r for r in pairwise_results}

        for result in pairwise_results:
            if result.risk_level in ["LOW", "MEDIUM", "HIGH"]:
                graph[result.doc_a_id].add(result.doc_b_id)
                graph[result.doc_b_id].add(result.doc_a_id)

        # 使用BFS/DFS查找连通分量
        visited = set()
        cluster_id = 0

        for doc_id in graph:
            if doc_id not in visited:
                # BFS查找连通分量
                component = self._bfs_component(doc_id, graph, visited)

                if len(component) >= 3:
                    # 判断聚类类型
                    cluster_type = self._determine_cluster_type(component, pair_lookup)

                    cluster = Cluster(
                        cluster_id=f"cluster_{cluster_id}",
                        doc_ids=list(component),
                        cluster_type=cluster_type,
                        confidence=0.8
                    )
                    clusters.append(cluster)
                    cluster_id += 1
                elif len(component) == 2:
                    # 检查是否为文本克隆
                    doc_ids = list(component)
                    pair_id = "::".join(sorted([doc_ids[0], doc_ids[1]]))
                    pair_result = pair_lookup.get(pair_id)
                    if pair_result and pair_result.evidence.text_evidence.local_similarity > 0.9:
                        cluster = Cluster(
                            cluster_id=f"cluster_{cluster_id}",
                            doc_ids=doc_ids,
                            cluster_type="TEXT_CLONE",
                            confidence=0.95
                        )
                        clusters.append(cluster)
                        cluster_id += 1

        logger.info(f"发现 {len(clusters)} 个风险聚类")
        return clusters

    def _bfs_component(self, start: str, graph: Dict[str, Set[str]], visited: Set[str]) -> Set[str]:
        """BFS查找连通分量（使用 deque 实现 O(1) 出队）"""
        component = set()
        queue = deque([start])
        visited.add(start)
        component.add(start)

        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)

        return component

    def _determine_cluster_type(
        self,
        component: Set[str],
        pair_lookup: Dict[str, PairwiseResult]
    ) -> str:
        """判断聚类类型（使用 pair_lookup 字典实现 O(1) 查找）"""
        text_high_count = 0
        metadata_high_count = 0

        doc_ids = list(component)
        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                pair_id = "::".join(sorted([doc_ids[i], doc_ids[j]]))
                pair_result = pair_lookup.get(pair_id)
                if pair_result:
                    if pair_result.evidence.text_evidence.local_similarity > 0.7:
                        text_high_count += 1
                    if len(pair_result.evidence.metadata_evidence.matched_fields) >= 3:
                        metadata_high_count += 1

        # 根据主要特征判断类型
        if text_high_count >= len(doc_ids):
            return "TEXT_CLONE"
        elif metadata_high_count >= len(doc_ids):
            return "META_GROUP"
        else:
            return "TEXT_CLONE"  # 默认文本克隆

    def _generate_file_profiles(
        self,
        pairwise_results: List[PairwiseResult],
        features: List[BidFeature]
    ) -> Dict[str, FileProfile]:
        """生成单文档风险画像"""
        profiles = {}

        # 初始化每个文档的画像
        for feature in features:
            profiles[feature.doc_id] = FileProfile(
                doc_id=feature.doc_id,
                filename=feature.filename
            )

        # 统计每个文档的关联信息
        for result in pairwise_results:
            if result.risk_level != "NONE":
                # 更新doc_a
                profile_a = profiles[result.doc_a_id]
                profile_a.related_suspicious_count += 1
                if self._is_higher_risk(result.risk_level, profile_a.max_risk_level):
                    profile_a.max_risk_level = result.risk_level

                # 更新doc_b
                profile_b = profiles[result.doc_b_id]
                profile_b.related_suspicious_count += 1
                if self._is_higher_risk(result.risk_level, profile_b.max_risk_level):
                    profile_b.max_risk_level = result.risk_level

        return profiles

    def _is_higher_risk(self, level1: str, level2: str) -> bool:
        """比较风险等级（使用模块级常量，避免每次调用时重建字典）"""
        return _RISK_ORDER.get(level1, 0) > _RISK_ORDER.get(level2, 0)
