"""
模块 D：风险评级与聚类引擎
"""
import logging
from typing import List, Dict, Set
from collections import defaultdict
import uuid

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

        text_local = scores.get('text_local', 0)
        paragraph_matches = evidence.text_evidence.paragraph_matches
        continuous_clone_blocks = evidence.text_evidence.continuous_clone_blocks
        detection_summary = evidence.text_evidence.detection_summary

        match_count = len(paragraph_matches)
        coverage = 0.0
        if match_count > 0:
            covered_a = len(set(m['paragraph_a_index'] for m in paragraph_matches))
            covered_b = len(set(m['paragraph_b_index'] for m in paragraph_matches))
            coverage = (covered_a + covered_b) / 100.0

        if text_local >= 0.75:
            if coverage < 0.1 and match_count < 5:
                risk_level = "MEDIUM"
                risk_factors.append(f"⚠️ 覆盖率不足({coverage:.2f})，匹配段落数({match_count})较少，降级为中等风险")
            else:
                risk_level = "HIGH"
                
                if detection_summary.get('sequence_matcher_count', 0) > 0:
                    risk_factors.append(f"✓ SequenceMatcher检测到高相似段落 ({detection_summary['sequence_matcher_count']}对)")
                
                if detection_summary.get('sbert_count', 0) > 0:
                    risk_factors.append(f"✓ SBERT验证通过 ({detection_summary['sbert_count']}对)")
                
                if len(continuous_clone_blocks) > 0:
                    for block in continuous_clone_blocks:
                        risk_factors.append(f"⚠️ 发现连续克隆块(长度{block['length']}, 相似度{block['similarity']:.4f})")
                
                risk_factors.append(f"文本相似度: {text_local:.4f}")
        else:
            risk_level = "LOW"
            if paragraph_matches:
                risk_factors.append(f"存在低相似段落 ({len(paragraph_matches)}对)")
            else:
                risk_factors.append("未发现显著相似段落")

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
                    cluster_type = self._determine_cluster_type(component, pairwise_results)

                    cluster = Cluster(
                        cluster_id=f"cluster_{cluster_id}",
                        doc_ids=list(component),
                        cluster_type=cluster_type,
                        confidence=0.8  # 简化版，实际可以更复杂的置信度计算
                    )
                    clusters.append(cluster)
                    cluster_id += 1
                elif len(component) == 2:
                    # 检查是否为文本克隆
                    doc_ids = list(component)
                    pair_result = self._find_pair_result(doc_ids[0], doc_ids[1], pairwise_results)
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
        """BFS查找连通分量"""
        component = set()
        queue = [start]
        visited.add(start)
        component.add(start)

        while queue:
            node = queue.pop(0)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)

        return component

    def _determine_cluster_type(
        self,
        component: Set[str],
        pairwise_results: List[PairwiseResult]
    ) -> str:
        """判断聚类类型"""
        # 统计聚类内的主要风险特征
        text_high_count = 0
        metadata_high_count = 0
        quote_high_count = 0

        doc_ids = list(component)
        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                pair_result = self._find_pair_result(doc_ids[i], doc_ids[j], pairwise_results)
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

    def _find_pair_result(
        self,
        doc_a_id: str,
        doc_b_id: str,
        pairwise_results: List[PairwiseResult]
    ) -> PairwiseResult:
        """查找特定文档对的结果"""
        pair_id = "::".join(sorted([doc_a_id, doc_b_id]))
        for result in pairwise_results:
            if result.pair_id == pair_id:
                return result
        return None

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
        """比较风险等级"""
        risk_order = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
        return risk_order.get(level1, 0) > risk_order.get(level2, 0)
