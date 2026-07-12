"""
检查点系统 — 支持崩溃后从最近检查点恢复

设计:
- JSON 格式存储管道状态
- 每阶段转换时写入检查点
- Phase 3（分析阶段）每 N 对增量写入
- 加载时验证 config_hash 防止配置漂移

文件结构:
  checkpoints/
    ├── pipeline_state.json    # 管道整体状态
    └── phase3_progress.json   # Phase 3 增量进度
"""

import os
import json
import hashlib
import logging
from datetime import datetime
from typing import Set

from data_structures import CheckpointState
from config import DetectionConfig

logger = logging.getLogger(__name__)


class CheckpointManager:
    """检查点管理器"""

    def __init__(self, checkpoint_dir: str, config: DetectionConfig):
        self.checkpoint_dir = checkpoint_dir
        self.config = config
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.state_path = os.path.join(checkpoint_dir, "pipeline_state.json")
        self.progress_path = os.path.join(checkpoint_dir, "phase3_progress.json")

    def load_or_new(self) -> CheckpointState:
        """加载检查点，如果不存在或配置变更则创建新状态"""
        if not getattr(self.config, 'ENABLE_CHECKPOINT', True):
            logger.info("断点续传已禁用，创建新状态")
            return self._new_state()

        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 验证配置哈希（检测配置漂移）
                saved_hash = data.get('config_hash', '')
                current_hash = self._hash_config()
                if saved_hash and saved_hash != current_hash:
                    logger.warning(
                        "配置已变更，之前的检查点可能不兼容。"
                        "将从头开始。"
                    )
                    return self._new_state()

                # 恢复状态
                state = CheckpointState(
                    phase=data.get('phase', 0),
                    completed_pairs=data.get('completed_pairs', 0),
                    total_pairs=data.get('total_pairs', 0),
                    processed_files=set(data.get('processed_files', [])),
                    completed_pair_ids=set(data.get('completed_pair_ids', [])),
                    start_time=data.get('start_time', ''),
                    config_hash=saved_hash,
                    input_hash=data.get('input_hash', ''),
                    version=data.get('version', 1),
                )

                logger.info(
                    f"检查点已加载: phase={state.phase}, "
                    f"completed_pairs={state.completed_pairs}/{state.total_pairs}"
                )
                return state

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"检查点文件损坏: {e}，将创建新状态")
                return self._new_state()

        return self._new_state()

    def save(self, state: CheckpointState):
        """保存管道状态"""
        if not getattr(self.config, 'ENABLE_CHECKPOINT', True):
            return
        state.config_hash = self._hash_config()

        data = {
            'phase': state.phase,
            'completed_pairs': state.completed_pairs,
            'total_pairs': state.total_pairs,
            'processed_files': list(state.processed_files),
            'completed_pair_ids': list(state.completed_pair_ids),
            'start_time': state.start_time,
            'config_hash': state.config_hash,
            'input_hash': state.input_hash,
            'version': state.version,
            'updated_at': datetime.now().isoformat(),
        }

        # 原子写入：先写临时文件，再重命名
        tmp_path = self.state_path + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        os.replace(tmp_path, self.state_path)
        logger.debug(f"检查点已保存: phase={state.phase}")

    def save_phase3_progress(self, completed_pair_ids: Set[str]):
        """增量保存 Phase 3 进度（每 N 对调用一次）"""
        data = {
            'completed_pair_ids': list(completed_pair_ids),
            'count': len(completed_pair_ids),
            'updated_at': datetime.now().isoformat(),
        }

        tmp_path = self.progress_path + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        os.replace(tmp_path, self.progress_path)

    def load_phase3_progress(self) -> Set[str]:
        """加载 Phase 3 增量进度"""
        if os.path.exists(self.progress_path):
            try:
                with open(self.progress_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return set(data.get('completed_pair_ids', []))
            except (json.JSONDecodeError, KeyError):
                pass
        return set()

    def clear(self):
        """清除所有检查点"""
        for path in [self.state_path, self.progress_path]:
            if os.path.exists(path):
                os.remove(path)
        logger.info("检查点已清除")

    def _new_state(self) -> CheckpointState:
        """创建新的初始状态"""
        return CheckpointState(
            phase=0,
            completed_pairs=0,
            total_pairs=0,
            processed_files=set(),
            completed_pair_ids=set(),
            start_time=datetime.now().isoformat(),
            config_hash=self._hash_config(),
            version=2,
        )

    def _hash_config(self) -> str:
        """计算配置哈希"""
        config_dict = {
            k: v for k, v in self.config.__dict__.items()
            if not k.startswith('_')
        }
        # 仅对影响检测结果的参数进行哈希
        relevant_keys = [
            'TEXT_GLOBAL_THRESHOLD', 'TEXT_LOCAL_THRESHOLD',
            'SBERT_BASE_THRESHOLD', 'SBERT_SHORT_PARAGRAPH_THRESHOLD',
            'MINHASH_LSH_THRESHOLD', 'PARAGRAPH_LSH_THRESHOLD',
            'CLONE_BLOCK_MIN_LENGTH', 'CLONE_BLOCK_MAX_GAP',
            'PARAGRAPH_MIN_JACCARD', 'MINHASH_NUM_HASHES',
        ]
        relevant = {
            k: str(config_dict.get(k, ''))
            for k in relevant_keys
        }
        config_str = json.dumps(relevant, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]
