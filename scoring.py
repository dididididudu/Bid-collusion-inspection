"""
模块 D：风险评级与聚类引擎（简化版 — 仅报告编排，不评分）
"""
import logging
from typing import List, Dict, Set
from collections import defaultdict, deque
import uuid

from data_structures import (
    BidFeature, PairwiseResult, GlobalReport,
    Cluster, FileProfile
)
from config import DetectionConfig

logger = logging.getLogger(__name__)


class RiskScoringEngine:
    """风险评分引擎（简化版：仅负责报告编排和聚类，不计算风险评分/等级）"""

    def __init__(self, config: DetectionConfig):
        self.config = config

    def generate_report(
        self,
        pairwise_results: List[PairwiseResult],
        features: List[BidFeature]
    ) -> GlobalReport:
        """生成全局检测报告（不评分，直接编排）"""

        # 1. 计数有证据的文档对
        suspicious_pairs = [r for r in pairwise_results if r.has_evidence()]

        # 2. 风险聚类
        risk_clusters = self._cluster_risks(pairwise_results, features)

        # 3. 生成单文档画像
        file_profiles = self._generate_file_profiles(pairwise_results, features)

        # 4. 生成报告
        from datetime import datetime
        report = GlobalReport(
            report_id=str(uuid.uuid4()),
            generated_at=datetime.now().isoformat(),
            total_files=len(features),
            total_pairs=len(features) * (len(features) - 1) // 2,
            candidate_pairs=len(pairwise_results),
            suspicious_pairs=len(suspicious_pairs),
            high_risk_pairs=0,  # 不再区分高风险
            risk_clusters=risk_clusters,
            pairwise_results=pairwise_results,
            file_profiles=file_profiles
        )

        logger.info(
            f"报告生成完成: {len(suspicious_pairs)} 对有雷同项, "
            f"{len(risk_clusters)} 个风险聚类"
        )
        return report

    def _score_pair(self, result: PairwiseResult,
                     enabled_dims: dict = None) -> PairwiseResult:
        """简化版：不计算风险评分/等级，直接返回原结果"""
        return result

    def _cluster_risks(
        self,
        pairwise_results: List[PairwiseResult],
        features: List[BidFeature]
    ) -> List[Cluster]:
        """风险聚类 — 通过连通分量发现涉嫌围标的文档组"""
        clusters = []

        # 构建图结构：节点为文档，边为有雷同证据的文档对
        graph = defaultdict(set)

        for result in pairwise_results:
            if result.has_evidence():
                graph[result.doc_a_id].add(result.doc_b_id)
                graph[result.doc_b_id].add(result.doc_a_id)

        # 使用BFS查找连通分量
        visited = set()
        cluster_id = 0

        for doc_id in graph:
            if doc_id not in visited:
                component = self._bfs_component(doc_id, graph, visited)

                if len(component) >= 3:
                    cluster_type = self._determine_cluster_type(component, pairwise_results)
                    cluster = Cluster(
                        cluster_id=f"cluster_{cluster_id}",
                        doc_ids=list(component),
                        cluster_type=cluster_type,
                        confidence=0.8
                    )
                    clusters.append(cluster)
                    cluster_id += 1
                elif len(component) == 2:
                    doc_ids = list(component)
                    # 检查是否为文本克隆
                    for r in pairwise_results:
                        if (r.doc_a_id in doc_ids and r.doc_b_id in doc_ids) or \
                           (r.doc_b_id in doc_ids and r.doc_a_id in doc_ids):
                            if r.evidence.text_evidence.local_similarity > 0.9:
                                cluster = Cluster(
                                    cluster_id=f"cluster_{cluster_id}",
                                    doc_ids=doc_ids,
                                    cluster_type="TEXT_CLONE",
                                    confidence=0.95
                                )
                                clusters.append(cluster)
                                cluster_id += 1
                            break

        logger.info(f"发现 {len(clusters)} 个风险聚类")
        return clusters

    def _bfs_component(self, start: str, graph: Dict[str, Set[str]], visited: Set[str]) -> Set[str]:
        """BFS查找连通分量"""
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
        pairwise_results: List[PairwiseResult],
    ) -> str:
        """判断聚类类型"""
        text_high_count = 0
        metadata_high_count = 0

        # 构建 pair_id → result 字典
        pair_lookup = {r.pair_id: r for r in pairwise_results}

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

        if text_high_count >= len(doc_ids):
            return "TEXT_CLONE"
        elif metadata_high_count >= len(doc_ids):
            return "META_GROUP"
        else:
            return "TEXT_CLONE"

    def _generate_file_profiles(
        self,
        pairwise_results: List[PairwiseResult],
        features: List[BidFeature]
    ) -> Dict[str, FileProfile]:
        """生成单文档风险画像（简化：仅统计关联数量）"""
        profiles = {}

        for feature in features:
            profiles[feature.doc_id] = FileProfile(
                doc_id=feature.doc_id,
                filename=feature.filename
            )

        for result in pairwise_results:
            if result.has_evidence():
                if result.doc_a_id in profiles:
                    profiles[result.doc_a_id].related_suspicious_count += 1
                if result.doc_b_id in profiles:
                    profiles[result.doc_b_id].related_suspicious_count += 1

        return profiles
