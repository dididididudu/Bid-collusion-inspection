"""
模块 D：相似内容聚类与报告编排引擎
"""
import logging
from typing import List, Dict, Set, Optional
from collections import defaultdict, deque
import uuid

from data_structures import (
    BidFeature, PairwiseResult, GlobalReport,
    Cluster, FileProfile, MetadataGroup,
)
from config import DetectionConfig

logger = logging.getLogger(__name__)


class RiskScoringEngine:
    """报告引擎"""

    def __init__(self, config: DetectionConfig):
        self.config = config

    def generate_report(
        self,
        pairwise_results: List[PairwiseResult],
        features: List[BidFeature]
    ) -> GlobalReport:
        """生成全局检测报告（相似内容聚类与编排）"""

        # 1. 计数有证据的文档对
        suspicious_pairs = [r for r in pairwise_results if r.has_evidence()]

        # 2. 风险聚类
        risk_clusters = self._cluster_risks(pairwise_results, features)

        # ★ 3. 元数据聚合组（取代冗余的 pairwise 展示）
        metadata_groups = self._build_metadata_groups(pairwise_results, features)

        # 4. 生成单文档画像
        file_profiles = self._generate_file_profiles(pairwise_results, features)

        # 5. 生成报告
        from datetime import datetime
        report = GlobalReport(
            report_id=str(uuid.uuid4()),
            generated_at=datetime.now().isoformat(),
            total_files=len(features),
            total_pairs=len(features) * (len(features) - 1) // 2,
            candidate_pairs=len(pairwise_results),
            suspicious_pairs=len(suspicious_pairs),
            high_risk_pairs=0,
            risk_clusters=risk_clusters,
            metadata_groups=metadata_groups,  # ★
            pairwise_results=pairwise_results,
            file_profiles=file_profiles
        )

        logger.info(
            f"报告生成完成: {len(suspicious_pairs)} 对有雷同项, "
            f"{len(risk_clusters)} 个风险聚类, "
            f"{len(metadata_groups)} 个元数据聚合组"
        )
        return report

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

    def _build_metadata_groups(
        self,
        pairwise_results: List[PairwiseResult],
        features: List[BidFeature],
    ) -> List[MetadataGroup]:
        """从 pairwise 结果中聚合元数据组

        元数据雷同是传递性的：A↔B 作者相同 + B↔C 作者相同 → A、B、C 作者相同。
        当前 pairwise 展示有 N×(N-1)/2 条冗余，聚合成一条组更清晰。

        聚合策略：
        - 值相同型（author、creator、contact 等）：以 field+value 为 key 收集文档
        - 连通型（file_id）：通过同 file_id 子图的连通分量聚合
        """
        from collections import defaultdict

        filename_map = {f.doc_id: f.filename for f in features}

        # 值相同型：(group_type, value) → set[doc_id]
        value_groups: Dict[tuple, Set[str]] = defaultdict(set)

        # 连通型：same_file_id 子图
        file_id_graph: Dict[str, Set[str]] = defaultdict(set)
        file_id_docs: Set[str] = set()

        for r in pairwise_results:
            ev = r.evidence

            # ── 元数据证据（author / creator / producer / software_fingerprint）──
            me = ev.metadata_evidence
            for field in me.matched_fields:
                val = me.matched_values.get(field, '').strip()
                if not val:
                    continue
                if field == 'author':
                    gt = 'author'
                elif field in ('creator', 'producer', 'software_fingerprint'):
                    gt = 'editor'
                else:
                    gt = field
                value_groups[(gt, val)].add(r.doc_a_id)
                value_groups[(gt, val)].add(r.doc_b_id)

            # ── 文件码雷同（连通型）──
            if me.same_file_id:
                file_id_graph[r.doc_a_id].add(r.doc_b_id)
                file_id_graph[r.doc_b_id].add(r.doc_a_id)
                file_id_docs.add(r.doc_a_id)
                file_id_docs.add(r.doc_b_id)

            # ── 联系人雷同（值相同型）──
            ce = ev.contact_evidence
            for v in ce.common_mobiles:
                value_groups[('contact_mobile', v)].update([r.doc_a_id, r.doc_b_id])
            for v in ce.common_emails:
                value_groups[('contact_email', v)].update([r.doc_a_id, r.doc_b_id])
            for v in ce.common_contacts:
                value_groups[('contact_name', v)].update([r.doc_a_id, r.doc_b_id])
            for v in ce.common_companies:
                value_groups[('company_name', v)].update([r.doc_a_id, r.doc_b_id])
            for v in ce.common_credit_codes:
                value_groups[('credit_code', v)].update([r.doc_a_id, r.doc_b_id])

        # ── 构建 MetadataGroup（值相同型）──
        groups: List[MetadataGroup] = []
        for (gt, val), doc_ids in value_groups.items():
            if len(doc_ids) < 2:
                continue
            dids_sorted = sorted(doc_ids)
            groups.append(MetadataGroup(
                group_type=gt,
                shared_value=val,
                doc_ids=dids_sorted,
                doc_count=len(dids_sorted),
                filenames=[filename_map.get(d, d) for d in dids_sorted],
            ))

        # ── 构建 MetadataGroup（文件码 — 连通分量）──
        if file_id_docs:
            doc_id_to_file_id = {
                f.doc_id: f.metadata.file_id
                for f in features if f.metadata.file_id
            }
            visited: Set[str] = set()
            for start in file_id_docs:
                if start in visited:
                    continue
                # BFS 找连通分量
                component: Set[str] = set()
                queue = [start]
                visited.add(start)
                while queue:
                    node = queue.pop(0)
                    component.add(node)
                    for nb in file_id_graph.get(node, []):
                        if nb not in visited:
                            visited.add(nb)
                            queue.append(nb)
                if len(component) < 2:
                    continue
                dids_sorted = sorted(component)
                # 取任一个文档的 file_id 值
                file_id_val = ''
                for d in dids_sorted:
                    if d in doc_id_to_file_id:
                        file_id_val = doc_id_to_file_id[d]
                        break
                groups.append(MetadataGroup(
                    group_type='file_id',
                    shared_value=file_id_val or '相同文件码',
                    doc_ids=dids_sorted,
                    doc_count=len(dids_sorted),
                    filenames=[filename_map.get(d, d) for d in dids_sorted],
                ))

        logger.info(f"构建了 {len(groups)} 个元数据聚合组")
        return groups

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
