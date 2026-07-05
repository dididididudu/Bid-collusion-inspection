"""
SQLite 特征缓存 - 将所有大型文本数据从内存转移到持久化存储

设计目标:
- 支持 100+ 文档 × 1000+ 页 PDF 的特征存储
- 文本块使用 zlib 压缩存储（压缩比约 5:1~8:1）
- 惰性加载：按需从 SQLite 加载文本内容
- WAL 模式 + 批量提交，优化写入性能
"""

import os
import zlib
import json
import sqlite3
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from contextlib import contextmanager

from data_structures import (
    BidFeature, ChunkMetadata, ChunkResult, PairwiseResult,
    MetadataFeature, QuoteSignature
)
from config import DetectionConfig
from utils.text_diff import compute_text_diff

logger = logging.getLogger(__name__)


class DocumentCache:
    """SQLite 支持的文档特征缓存"""

    # 当前数据库模式版本
    SCHEMA_VERSION = 7  # v7: paragraphs 新增 source 列 (text/ocr)

    def __init__(self, cache_dir: str, config: Optional[DetectionConfig] = None):
        """
        Args:
            cache_dir: SQLite 数据库文件目录
            config: 检测配置（可选，用于读取参数）
        """
        self.config = config
        os.makedirs(cache_dir, exist_ok=True)
        self.db_path = os.path.join(cache_dir, "features.db")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        _busy_ms = getattr(config, 'DB_BUSY_TIMEOUT', 30000)
        self._apply_pragmas(self.conn, _busy_ms)

        self._create_schema()
        logger.info(f"SQLite 缓存已初始化: {self.db_path} (WAL 模式)")

    @staticmethod
    def _apply_pragmas(conn: sqlite3.Connection, busy_timeout: int = 30000) -> None:
        """对 SQLite 连接应用性能优化 PRAGMA 配置"""
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")  # 64MB 缓存
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute(f"PRAGMA busy_timeout={busy_timeout}")

    def create_thread_connection(self) -> sqlite3.Connection:
        """为工作线程创建独立的 SQLite 连接（线程安全）

        每个线程需要自己的连接对象。使用 WAL 模式支持并发读写。
        """
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        _busy_ms = getattr(self.config, 'DB_BUSY_TIMEOUT', 30000)
        self._apply_pragmas(conn, _busy_ms)
        return conn

    @contextmanager
    def transaction(self):
        """事务上下文管理器，自动提交或回滚"""
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # ================================================================
    # 数据库模式
    # ================================================================

    def _create_schema(self):
        """创建数据库表结构"""
        cursor = self.conn.cursor()

        # 版本管理
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)

        # 检查版本
        cursor.execute("SELECT MAX(version) FROM schema_version")
        row = cursor.fetchone()
        current_version = row[0] if row[0] is not None else 0

        if current_version < self.SCHEMA_VERSION:
            self._create_tables(cursor)
            # v2 迁移：新增嵌入和元数据指纹支持
            if current_version < 2:
                self._migrate_v2(cursor)
            # v3 迁移：新增图片 OCR 结果表
            if current_version < 3:
                self._migrate_v3(cursor)
            # v4 迁移：image_ocr_results 新增 non_text_hash 列
            if current_version < 4:
                self._migrate_v4(cursor)
            # v5 迁移：image_ocr_results 新增 image_width/image_height 列
            if current_version < 5:
                self._migrate_v5(cursor)
            # v6 迁移：image_ocr_results 新增 thumbnail 列
            if current_version < 6:
                self._migrate_v6(cursor)
            # v7 迁移：paragraphs 新增 source 列
            if current_version < 7:
                self._migrate_v7(cursor)
            cursor.execute(
                "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                (self.SCHEMA_VERSION,)
            )
            self.conn.commit()
            logger.info(f"数据库模式已创建/升级到版本 {self.SCHEMA_VERSION}")

    def _migrate_v2(self, cursor):
        """v2 迁移：嵌入列 + 新表"""
        # 段落表增加 embedding 列
        try:
            cursor.execute(
                "ALTER TABLE paragraphs ADD COLUMN embedding BLOB"
            )
        except Exception:
            pass  # 列可能已存在

        # 文档级嵌入表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_embeddings (
                doc_id TEXT PRIMARY KEY,
                embedding BLOB,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            )
        """)

        # 元数据指纹表（加速 Phase 2 元数据查询）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata_fingerprints (
                doc_id TEXT PRIMARY KEY,
                author TEXT DEFAULT '',
                creator TEXT DEFAULT '',
                producer TEXT DEFAULT '',
                software_fingerprint TEXT DEFAULT '',
                time_bucket TEXT DEFAULT '',
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_md_fp "
            "ON metadata_fingerprints(software_fingerprint, time_bucket)"
        )
        logger.info("v2 模式迁移完成：嵌入 + 文档向量 + 元数据指纹")

    def _migrate_v3(self, cursor):
        """v3 迁移：图片 OCR 结果表"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS image_ocr_results (
                doc_id TEXT NOT NULL,
                page_num INTEGER NOT NULL,
                image_hash TEXT NOT NULL,
                ocr_text TEXT DEFAULT '',
                ocr_words_json TEXT DEFAULT '[]',
                text_bboxes_json TEXT DEFAULT '[]',
                confidence REAL DEFAULT 0.0,
                PRIMARY KEY (doc_id, page_num, image_hash),
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_ocr_doc "
            "ON image_ocr_results(doc_id)"
        )
        logger.info("v3 模式迁移完成：图片 OCR 结果表")

    def _migrate_v4(self, cursor):
        """v4 迁移：image_ocr_results 新增 non_text_hash 列"""
        try:
            cursor.execute(
                "ALTER TABLE image_ocr_results "
                "ADD COLUMN non_text_hash TEXT DEFAULT ''"
            )
            logger.info("v4 模式迁移完成：新增 non_text_hash 列")
        except Exception:
            # 列可能已存在
            pass

    def _migrate_v5(self, cursor):
        """v5 迁移：image_ocr_results 新增 image_width/image_height 列"""
        for col in ['image_width', 'image_height']:
            try:
                cursor.execute(
                    f"ALTER TABLE image_ocr_results "
                    f"ADD COLUMN {col} INTEGER DEFAULT 0"
                )
                logger.info(f"v5 迁移完成：新增 {col} 列")
            except Exception:
                pass

    def _migrate_v6(self, cursor):
        """v6 迁移：image_ocr_results 新增 thumbnail 列"""
        try:
            cursor.execute(
                "ALTER TABLE image_ocr_results "
                "ADD COLUMN thumbnail BLOB DEFAULT NULL"
            )
            logger.info("v6 迁移完成：新增 thumbnail 列")
        except Exception:
            pass

    def _migrate_v7(self, cursor):
        """v7 迁移：paragraphs 新增 source 列 (text/ocr)"""
        try:
            cursor.execute(
                "ALTER TABLE paragraphs ADD COLUMN source TEXT DEFAULT 'text'"
            )
            logger.info("v7 迁移完成：paragraphs 新增 source 列")
        except Exception:
            pass

    def _create_tables(self, cursor):
        """创建所有表"""
        # 文档注册表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_size INTEGER DEFAULT 0,
                page_count INTEGER DEFAULT 0,
                total_text_length INTEGER DEFAULT 0,
                is_scanned INTEGER DEFAULT 0,
                doc_simhash TEXT DEFAULT '',
                doc_minhash TEXT DEFAULT '',
                metadata_json TEXT DEFAULT '{}',
                quote_values_json TEXT DEFAULT '[]',
                quote_tail_dist_json TEXT DEFAULT '{}',
                quote_count INTEGER DEFAULT 0,
                quote_integer_ratio REAL DEFAULT 0.0,
                quote_mean REAL DEFAULT 0.0,
                quote_std REAL DEFAULT 0.0,
                image_hashes_json TEXT DEFAULT '[]',
                image_hash_count INTEGER DEFAULT 0,
                chunk_count INTEGER DEFAULT 0,
                extracted_at TEXT DEFAULT (datetime('now')),
                processed INTEGER DEFAULT 0
            )
        """)

        # 文本块表（压缩存储）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                start_page INTEGER NOT NULL,
                end_page INTEGER NOT NULL,
                text_length INTEGER NOT NULL,
                text_blob BLOB,
                simhash TEXT DEFAULT '',
                paragraph_count INTEGER DEFAULT 0,
                paragraph_hashes_json TEXT DEFAULT '[]',
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id),
                UNIQUE(doc_id, chunk_index)
            )
        """)

        # 段落表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paragraphs (
                para_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                para_index INTEGER NOT NULL,
                text TEXT,
                minhash TEXT DEFAULT '',
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id),
                FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id),
                UNIQUE(doc_id, para_index)
            )
        """)

        # 候选文档对表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS candidate_pairs (
                pair_id TEXT PRIMARY KEY,
                doc_a_id TEXT NOT NULL,
                doc_b_id TEXT NOT NULL,
                selection_method TEXT DEFAULT '',
                doc_level_similarity REAL DEFAULT 0.0,
                processed INTEGER DEFAULT 0,
                UNIQUE(doc_a_id, doc_b_id)
            )
        """)

        # 分析结果表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pairwise_results (
                pair_id TEXT PRIMARY KEY,
                doc_a_id TEXT NOT NULL,
                doc_b_id TEXT NOT NULL,
                text_similarity REAL DEFAULT 0.0,
                risk_level TEXT DEFAULT 'NONE',
                risk_score INTEGER DEFAULT 0,
                risk_factors_json TEXT DEFAULT '[]',
                match_count INTEGER DEFAULT 0,
                clone_block_count INTEGER DEFAULT 0,
                metadata_match_count INTEGER DEFAULT 0,
                image_match_count INTEGER DEFAULT 0,
                evidence_json TEXT DEFAULT '{}',
                processed_at TEXT DEFAULT (datetime('now'))
            )
        """)

        # 段落匹配详情表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paragraph_matches (
                match_id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair_id TEXT NOT NULL,
                similarity REAL NOT NULL,
                para_a_index INTEGER NOT NULL,
                para_b_index INTEGER NOT NULL,
                detection_method TEXT DEFAULT '',
                is_continuous_clone INTEGER DEFAULT 0,
                clone_group_id TEXT DEFAULT '',
                FOREIGN KEY (pair_id) REFERENCES pairwise_results(pair_id)
            )
        """)

        # 管道状态表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_state (
                key TEXT PRIMARY KEY,
                value TEXT DEFAULT '',
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)

        # 创建索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id, chunk_index)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_paragraphs_doc ON paragraphs(doc_id, para_index)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_paragraphs_chunk ON paragraphs(chunk_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_candidate_pairs_processed ON candidate_pairs(processed)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_paragraph_matches_pair ON paragraph_matches(pair_id)")

    # ================================================================
    # 文档 CRUD
    # ================================================================

    def store_document(self, doc: BidFeature) -> None:
        """存储文档级特征"""
        metadata_json = json.dumps({
            'author': doc.metadata.author,
            'creator': doc.metadata.creator,
            'producer': doc.metadata.producer,
            'created_time': doc.metadata.created_time,
            'modified_time': doc.metadata.modified_time,
            'software_fingerprint': doc.metadata.software_fingerprint,
            'time_bucket': doc.metadata.time_bucket,
        }, ensure_ascii=False)

        with self.transaction() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO documents (
                    doc_id, filename, file_size, page_count, total_text_length,
                    is_scanned, doc_simhash, doc_minhash,
                    metadata_json,
                    quote_values_json, quote_tail_dist_json,
                    quote_count, quote_integer_ratio, quote_mean, quote_std,
                    image_hashes_json, image_hash_count,
                    chunk_count, extracted_at, processed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc.doc_id,
                doc.filename,
                doc.file_size,
                doc.page_count,
                doc.text_length,
                1 if doc.is_scanned else 0,
                doc.text_simhash,
                json.dumps(doc.doc_minhash) if doc.doc_minhash else '',
                metadata_json,
                json.dumps(doc.quotes),
                json.dumps(doc.quote_signature.tail_distribution),
                doc.quote_signature.count,
                doc.quote_signature.integer_ratio,
                doc.quote_signature.mean,
                doc.quote_signature.std,
                json.dumps(doc.image_hashes),
                len(doc.image_hashes),
                doc.chunk_count,
                doc.extracted_at,
                1  # processed = True
            ))

    def load_document(self, doc_id: str) -> Optional[BidFeature]:
        """加载单个文档特征"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_bid_feature(row)

    def load_all_documents(self) -> List[BidFeature]:
        """加载所有已处理文档的特征（轻量级，不含文本内容）"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM documents")
        docs = []
        for row in cursor.fetchall():
            docs.append(self._row_to_bid_feature(row))
        return docs

    def get_unprocessed_docs(self) -> List[str]:
        """获取尚未提取特征的文档路径列表（用于 Phase 1 恢复）"""
        # 从 pipeline_state 中读取已处理的文件列表
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT value FROM pipeline_state WHERE key = 'processed_files'"
        )
        row = cursor.fetchone()
        if row and row[0]:
            return json.loads(row[0])
        return []

    def set_processed_files(self, file_paths: List[str]) -> None:
        """记录已处理的文件列表"""
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO pipeline_state (key, value, updated_at) VALUES (?, ?, datetime('now'))",
                ('processed_files', json.dumps(file_paths))
            )

    def _row_to_bid_feature(self, row: tuple) -> BidFeature:
        """将数据库行转换为 BidFeature"""
        columns = [
            'doc_id', 'filename', 'file_size', 'page_count', 'total_text_length',
            'is_scanned', 'doc_simhash', 'doc_minhash',
            'metadata_json', 'quote_values_json', 'quote_tail_dist_json',
            'quote_count', 'quote_integer_ratio', 'quote_mean', 'quote_std',
            'image_hashes_json', 'image_hash_count', 'chunk_count', 'extracted_at', 'processed'
        ]
        data = dict(zip(columns, row))

        metadata_raw = json.loads(data['metadata_json'] or '{}')
        metadata = MetadataFeature(
            author=metadata_raw.get('author', ''),
            creator=metadata_raw.get('creator', ''),
            producer=metadata_raw.get('producer', ''),
            created_time=metadata_raw.get('created_time', ''),
            modified_time=metadata_raw.get('modified_time', ''),
            software_fingerprint=metadata_raw.get('software_fingerprint', ''),
            time_bucket=metadata_raw.get('time_bucket', ''),
        )

        quotes = json.loads(data['quote_values_json'] or '[]')

        quote_sig = QuoteSignature(
            count=data['quote_count'] or 0,
            values=quotes,
            tail_distribution=json.loads(data['quote_tail_dist_json'] or '{}'),
            integer_ratio=data['quote_integer_ratio'] or 0.0,
            mean=data['quote_mean'] or 0.0,
            std=data['quote_std'] or 0.0,
        )

        doc_minhash = None
        if data['doc_minhash']:
            try:
                doc_minhash = json.loads(data['doc_minhash'])
            except (json.JSONDecodeError, TypeError):
                doc_minhash = None

        return BidFeature(
            doc_id=data['doc_id'],
            filename=data['filename'],
            file_size=data['file_size'] or 0,
            text_content="",  # 流式模式下不加载文本
            text_length=data['total_text_length'] or 0,
            text_simhash=data['doc_simhash'] or '',
            paragraphs=[],  # 流式模式下不加载
            paragraph_hashes=[],
            metadata=metadata,
            quotes=quotes,
            quote_signature=quote_sig,
            image_hashes=json.loads(data['image_hashes_json'] or '[]'),
            extracted_at=data['extracted_at'] or '',
            is_scanned=bool(data['is_scanned']),
            page_count=data['page_count'] or 0,
            doc_minhash=doc_minhash,
            chunk_count=data['chunk_count'] or 0,
        )

    # ================================================================
    # 文本块 CRUD
    # ================================================================

    def store_chunk(self, chunk_result: ChunkResult) -> None:
        """存储文本块（文本内容 zlib 压缩）"""
        # 压缩文本内容
        text_bytes = chunk_result.text.encode('utf-8')
        compressed = zlib.compress(text_bytes, level=6)

        chunk_id = self._chunk_id(chunk_result.doc_id, chunk_result.chunk_index)
        paragraph_hashes_json = json.dumps(chunk_result.paragraph_hashes)

        with self.transaction() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO chunks (
                    chunk_id, doc_id, chunk_index, start_page, end_page,
                    text_length, text_blob, simhash, paragraph_count, paragraph_hashes_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk_id,
                chunk_result.doc_id,
                chunk_result.chunk_index,
                chunk_result.start_page,
                chunk_result.end_page,
                chunk_result.text_length if hasattr(chunk_result, 'text_length') else len(chunk_result.text),
                compressed,
                chunk_result.simhash,
                len(chunk_result.paragraphs),
                paragraph_hashes_json,
            ))

            # 批量存储段落数据（executemany 减少 Python-SQLite 往返）
            source = getattr(chunk_result, 'source', 'text')
            para_rows = [
                (
                    self._para_id(chunk_result.doc_id, para_idx),
                    chunk_result.doc_id, chunk_id,
                    para_idx, para_text, para_hash, source
                )
                for para_idx, (para_text, para_hash) in enumerate(
                    zip(chunk_result.paragraphs, chunk_result.paragraph_hashes)
                )
            ]
            conn.executemany("""
                INSERT OR REPLACE INTO paragraphs (
                    para_id, doc_id, chunk_id, para_index, text, minhash, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, para_rows)

    def load_chunk_text(self, doc_id: str, chunk_index: int) -> Optional[str]:
        """加载文本块的原始文本（解压缩）"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT text_blob FROM chunks WHERE doc_id = ? AND chunk_index = ?",
            (doc_id, chunk_index)
        )
        row = cursor.fetchone()
        if row and row[0]:
            return zlib.decompress(row[0]).decode('utf-8')
        return None

    def load_chunk_metadata(self, doc_id: str, chunk_index: int) -> Optional[ChunkMetadata]:
        """加载文本块元数据（不含文本内容）"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT doc_id, chunk_index, start_page, end_page, text_length, simhash, paragraph_count "
            "FROM chunks WHERE doc_id = ? AND chunk_index = ?",
            (doc_id, chunk_index)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return ChunkMetadata(
            doc_id=row[0],
            chunk_index=row[1],
            start_page=row[2],
            end_page=row[3],
            text_length=row[4],
            simhash=row[5] or '',
            paragraph_count=row[6] or 0,
        )

    def load_document_chunks(self, doc_id: str) -> List[ChunkMetadata]:
        """加载文档的所有文本块元数据"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT doc_id, chunk_index, start_page, end_page, text_length, simhash, paragraph_count "
            "FROM chunks WHERE doc_id = ? ORDER BY chunk_index",
            (doc_id,)
        )
        return [
            ChunkMetadata(
                doc_id=row[0], chunk_index=row[1],
                start_page=row[2], end_page=row[3],
                text_length=row[4], simhash=row[5] or '',
                paragraph_count=row[6] or 0,
            )
            for row in cursor.fetchall()
        ]

    def load_paragraph_hashes(self, doc_id: str, chunk_index: int) -> List[str]:
        """加载文本块的段落哈希列表"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT paragraph_hashes_json FROM chunks WHERE doc_id = ? AND chunk_index = ?",
            (doc_id, chunk_index)
        )
        row = cursor.fetchone()
        if row and row[0]:
            return json.loads(row[0])
        return []

    def load_paragraph_text(self, doc_id: str, para_index: int) -> Optional[str]:
        """加载单个段落的文本"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT text FROM paragraphs WHERE doc_id = ? AND para_index = ?",
            (doc_id, para_index)
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def load_all_paragraphs_text(self, doc_id: str) -> List[str]:
        """加载文档所有段落的文本（用于全局 SBERT 编码）

        Returns:
            按 para_index 升序排列的文本列表
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT text FROM paragraphs WHERE doc_id = ? "
            "AND text IS NOT NULL AND text != '' ORDER BY para_index",
            (doc_id,)
        )
        return [row[0] for row in cursor.fetchall()]

    def load_all_paragraph_minhashes(self, doc_id: str) -> Dict[int, str]:
        """加载文档所有段落的 MinHash 签名（用于分析阶段）"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT para_index, minhash FROM paragraphs WHERE doc_id = ? ORDER BY para_index",
            (doc_id,)
        )
        return {row[0]: row[1] for row in cursor.fetchall() if row[1]}

    def get_paragraph_source_map(self, doc_id: str) -> Dict[int, str]:
        """获取段落类型映射（para_index -> 'text'|'ocr'）"""
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "SELECT para_index, source FROM paragraphs "
                "WHERE doc_id = ? ORDER BY para_index",
                (doc_id,)
            )
            return {row[0]: row[1] or 'text' for row in cursor.fetchall()}
        except Exception:
            # 兼容旧库（无 source 列）
            cursor.execute(
                "SELECT para_index FROM paragraphs WHERE doc_id = ? ORDER BY para_index",
                (doc_id,)
            )
            return {row[0]: 'text' for row in cursor.fetchall()}

    def load_paragraphs_in_range(self, doc_id: str, start_idx: int, end_idx: int) -> List[Dict]:
        """加载指定范围的段落数据（含文本和 MinHash）"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT para_index, text, minhash FROM paragraphs "
            "WHERE doc_id = ? AND para_index >= ? AND para_index < ? "
            "ORDER BY para_index",
            (doc_id, start_idx, end_idx)
        )
        return [
            {
                'para_index': row[0],
                'text': row[1] or '',
                'minhash': row[2] or '',
            }
            for row in cursor.fetchall()
        ]

    def get_document_paragraph_count(self, doc_id: str) -> int:
        """获取文档的总段落数"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT MAX(para_index) + 1 FROM paragraphs WHERE doc_id = ?",
            (doc_id,)
        )
        row = cursor.fetchone()
        return row[0] if row[0] else 0

    # ================================================================
    # 嵌入向量 CRUD（Phase 1.5 全局编码 + Phase 3 查表）
    # ================================================================

    def store_paragraph_embeddings(
        self, doc_id: str, para_indices: List[int], embeddings: 'np.ndarray'
    ) -> None:
        """批量存储段落 SBERT 嵌入向量（BLOB 序列化）

        Args:
            doc_id: 文档 ID
            para_indices: 段落索引列表（需与 embeddings 行对应）
            embeddings: shape (n_paras, embedding_dim) 的 numpy 数组
        """
        data = []
        for i, para_idx in enumerate(para_indices):
            blob = embeddings[i].astype(np.float32).tobytes()
            data.append((blob, doc_id, para_idx))
        with self.transaction() as conn:
            conn.executemany(
                "UPDATE paragraphs SET embedding = ? "
                "WHERE doc_id = ? AND para_index = ?",
                data
            )

    def load_paragraph_embeddings(
        self, doc_id: str, para_indices: List[int]
    ) -> Dict[int, 'np.ndarray']:
        """加载指定段落的嵌入向量

        Returns:
            {para_index: ndarray(embedding_dim,)}
        """
        if not para_indices:
            return {}
        cursor = self.conn.cursor()
        placeholders = ','.join('?' * len(para_indices))
        cursor.execute(
            f"SELECT para_index, embedding FROM paragraphs "
            f"WHERE doc_id = ? AND para_index IN ({placeholders}) "
            f"AND embedding IS NOT NULL",
            [doc_id] + list(para_indices)
        )
        result = {}
        for row in cursor.fetchall():
            blob = row[1]
            if blob:
                result[row[0]] = np.frombuffer(blob, dtype=np.float32)
        return result

    def load_all_paragraph_embeddings(
        self, doc_id: str
    ) -> Dict[int, 'np.ndarray']:
        """加载文档所有段落的嵌入向量"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT para_index, embedding FROM paragraphs "
            "WHERE doc_id = ? AND embedding IS NOT NULL ORDER BY para_index",
            (doc_id,)
        )
        result = {}
        for row in cursor.fetchall():
            blob = row[1]
            if blob:
                result[row[0]] = np.frombuffer(blob, dtype=np.float32)
        return result

    def store_document_embedding(self, doc_id: str, embedding: 'np.ndarray') -> None:
        """存储文档级嵌入向量（段落嵌入均值池化）"""
        blob = embedding.astype(np.float32).tobytes()
        self.conn.execute(
            "INSERT OR REPLACE INTO document_embeddings (doc_id, embedding) "
            "VALUES (?, ?)",
            (doc_id, blob)
        )
        self.conn.commit()

    def load_all_document_embeddings(self) -> Dict[str, 'np.ndarray']:
        """加载所有文档级嵌入向量

        Returns:
            {doc_id: ndarray(embedding_dim,)}
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT doc_id, embedding FROM document_embeddings "
            "WHERE embedding IS NOT NULL"
        )
        result = {}
        for row in cursor.fetchall():
            blob = row[1]
            if blob:
                result[row[0]] = np.frombuffer(blob, dtype=np.float32)
        return result

    # ================================================================
    # 元数据指纹 CRUD（Phase 2 元数据候选筛选）
    # ================================================================

    def store_metadata_fingerprint(
        self, doc_id: str, author: str = '', creator: str = '',
        producer: str = '', software_fingerprint: str = '', time_bucket: str = ''
    ) -> None:
        """存储文档元数据指纹"""
        self.conn.execute(
            "INSERT OR REPLACE INTO metadata_fingerprints "
            "(doc_id, author, creator, producer, software_fingerprint, time_bucket) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (doc_id, author, creator, producer, software_fingerprint, time_bucket)
        )
        self.conn.commit()

    def load_metadata_fingerprints(self) -> Dict[str, Dict[str, str]]:
        """加载所有文档的元数据指纹

        Returns:
            {doc_id: {author, creator, producer, software_fingerprint, time_bucket}}
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT doc_id, author, creator, producer, "
            "software_fingerprint, time_bucket FROM metadata_fingerprints"
        )
        result = {}
        for row in cursor.fetchall():
            result[row[0]] = {
                'author': row[1] or '',
                'creator': row[2] or '',
                'producer': row[3] or '',
                'software_fingerprint': row[4] or '',
                'time_bucket': row[5] or '',
            }
        return result

    # ================================================================
    # 图片 OCR 结果 CRUD
    # ================================================================

    def store_image_ocr_result(
        self, doc_id: str, page_num: int, image_hash: str,
        ocr_text: str = '', ocr_words: List[str] = None,
        bboxes: List[Dict] = None, confidence: float = 0.0,
        non_text_hash: str = '',
        image_width: int = 0, image_height: int = 0,
        thumbnail: bytes = b'',
    ) -> None:
        """存储单张图片的 OCR 结果"""
        import json as _json
        self.conn.execute(
            "INSERT OR REPLACE INTO image_ocr_results "
            "(doc_id, page_num, image_hash, ocr_text, ocr_words_json, "
            "text_bboxes_json, confidence, non_text_hash, "
            "image_width, image_height, thumbnail) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                doc_id, page_num, image_hash, ocr_text,
                _json.dumps(ocr_words or [], ensure_ascii=False),
                _json.dumps(bboxes or [], ensure_ascii=False),
                confidence,
                non_text_hash,
                image_width, image_height,
                thumbnail if thumbnail else None,
            )
        )
        self.conn.commit()

    def _get_ocr_col_count(self) -> int:
        """探测 image_ocr_results 表的列数（结果缓存，避免每次调 PRAGMA）"""
        if not hasattr(self, '_ocr_col_count_cache'):
            cursor = self.conn.cursor()
            try:
                cursor.execute("PRAGMA table_info(image_ocr_results)")
                self._ocr_col_count_cache = len(cursor.fetchall())
            except Exception:
                self._ocr_col_count_cache = 6  # v3 最小值
        return self._ocr_col_count_cache

    def load_image_ocr_results(
        self, doc_id: str
    ) -> List[Dict]:
        """加载文档的所有图片 OCR 结果"""
        import json as _json
        cursor = self.conn.cursor()

        # 使用缓存的列数（兼容 v3/v4/v5/v6+）
        col_count = self._get_ocr_col_count()

        if col_count >= 10:
            # v6+: 含 non_text_hash + image_width + image_height + thumbnail
            cursor.execute(
                "SELECT page_num, image_hash, ocr_text, ocr_words_json, "
                "text_bboxes_json, confidence, non_text_hash, "
                "image_width, image_height, thumbnail "
                "FROM image_ocr_results "
                "WHERE doc_id = ? ORDER BY page_num",
                (doc_id,)
            )
        elif col_count >= 9:
            # v5: 含 non_text_hash + image_width + image_height
            cursor.execute(
                "SELECT page_num, image_hash, ocr_text, ocr_words_json, "
                "text_bboxes_json, confidence, non_text_hash, "
                "image_width, image_height "
                "FROM image_ocr_results "
                "WHERE doc_id = ? ORDER BY page_num",
                (doc_id,)
            )
        elif col_count >= 7:
            # v4: 含 non_text_hash
            cursor.execute(
                "SELECT page_num, image_hash, ocr_text, ocr_words_json, "
                "text_bboxes_json, confidence, non_text_hash "
                "FROM image_ocr_results "
                "WHERE doc_id = ? ORDER BY page_num",
                (doc_id,)
            )
        else:
            # v3: 无 non_text_hash
            cursor.execute(
                "SELECT page_num, image_hash, ocr_text, ocr_words_json, "
                "text_bboxes_json, confidence FROM image_ocr_results "
                "WHERE doc_id = ? ORDER BY page_num",
                (doc_id,)
            )

        results = []
        for row in cursor.fetchall():
            entry = {
                'page_num': row[0],
                'image_hash': row[1],
                'ocr_text': row[2] or '',
                'ocr_words': _json.loads(row[3] or '[]'),
                'bboxes': _json.loads(row[4] or '[]'),
                'confidence': row[5] or 0.0,
            }
            if col_count >= 7:
                entry['non_text_hash'] = row[6] or ''
            if col_count >= 9:
                entry['image_width'] = row[7] or 0
                entry['image_height'] = row[8] or 0
            if col_count >= 10:
                entry['thumbnail'] = row[9] if row[9] else b''
            results.append(entry)
        return results

    # ================================================================
    # 候选对 CRUD
    # ================================================================

    def store_candidate_pairs(self, pairs: List[Tuple[str, str, str, float]]) -> None:
        """批量存储候选对

        Args:
            pairs: [(doc_a_id, doc_b_id, method, similarity), ...]
        """
        with self.transaction() as conn:
            rows = [
                ("::".join(sorted([doc_a_id, doc_b_id])),
                 doc_a_id, doc_b_id, method, similarity)
                for doc_a_id, doc_b_id, method, similarity in pairs
            ]
            conn.executemany("""
                INSERT OR IGNORE INTO candidate_pairs (
                    pair_id, doc_a_id, doc_b_id, selection_method, doc_level_similarity
                ) VALUES (?, ?, ?, ?, ?)
            """, rows)

    def get_unprocessed_pairs(self, limit: int = 0) -> List[Tuple[str, str]]:
        """获取尚未分析的候选对"""
        cursor = self.conn.cursor()
        query = "SELECT doc_a_id, doc_b_id FROM candidate_pairs WHERE processed = 0"
        if limit > 0:
            query += f" LIMIT {limit}"
        cursor.execute(query)
        return [(row[0], row[1]) for row in cursor.fetchall()]

    def mark_pair_processed(self, doc_a_id: str, doc_b_id: str) -> None:
        """标记候选对为已处理（不单独 commit，由调用方批量提交）"""
        pair_id = "::".join(sorted([doc_a_id, doc_b_id]))
        self.conn.execute(
            "UPDATE candidate_pairs SET processed = 1 WHERE pair_id = ?",
            (pair_id,)
        )

    def get_total_candidate_pairs(self) -> int:
        """获取候选对总数"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM candidate_pairs")
        return cursor.fetchone()[0]

    def get_processed_pair_count(self) -> int:
        """获取已处理候选对数"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM candidate_pairs WHERE processed = 1")
        return cursor.fetchone()[0]

    # ================================================================
    # 分析结果 CRUD
    # ================================================================

    def store_pairwise_result(self, result: PairwiseResult) -> None:
        """存储单个分析结果"""
        evidence = result.evidence
        text_ev = evidence.text_evidence

        ie = evidence.image_evidence
        evidence_json = json.dumps({
            'metadata_matched_fields': evidence.metadata_evidence.matched_fields,
            'metadata_matched_values': evidence.metadata_evidence.matched_values,
            'same_time_bucket': evidence.metadata_evidence.same_time_bucket,
            # 图片证据完整字段
            'image_common_hashes': ie.common_image_hashes,
            'image_exact_count': ie.exact_image_count,
            'image_near_identical_count': ie.near_identical_count,
            'image_similar_count': ie.similar_image_count,
            'image_risk_score': ie.image_risk_score,
            'image_risk_factors': ie.image_risk_factors,
            'ps_suspicious': ie.ps_suspicious,
            'ps_suspicious_count': ie.ps_suspicious_count,
            'shared_typos': ie.shared_typos,
            'shared_typo_count': ie.shared_typo_count,
            'text_identical_count': ie.text_identical_count,
            'text_similar_count': ie.text_similar_count,
            'matched_image_pairs': ie.matched_image_pairs,
            'matched_text_pairs': ie.matched_text_pairs,
            'ps_detail_list': ie.ps_detail_list,
            # 段落证据
            'detection_summary': text_ev.detection_summary,
            'common_paragraphs': text_ev.common_paragraphs[:10],
            'continuous_clone_blocks': text_ev.continuous_clone_blocks,
        }, ensure_ascii=False)

        with self.transaction() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO pairwise_results (
                    pair_id, doc_a_id, doc_b_id,
                    text_similarity, risk_level, risk_score,
                    risk_factors_json, match_count, clone_block_count,
                    metadata_match_count, image_match_count, evidence_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.pair_id,
                result.doc_a_id,
                result.doc_b_id,
                result.similarity_scores.get('text_local', 0),
                result.risk_level,
                result.risk_score,
                json.dumps(result.risk_factors, ensure_ascii=False),
                len(text_ev.paragraph_matches),
                len(text_ev.continuous_clone_blocks),
                len(evidence.metadata_evidence.matched_fields),
                evidence.image_evidence.common_image_count,
                evidence_json,
            ))

            # 批量存储段落匹配详情（executemany 减少往返）
            if text_ev.paragraph_matches:
                match_rows = [
                    (
                        result.pair_id,
                        match.get('similarity', 0),
                        match.get('paragraph_a_index', 0),
                        match.get('paragraph_b_index', 0),
                        match.get('detection_method', ''),
                        1 if match.get('is_continuous_clone') else 0,
                        match.get('continuous_clone_group_id', ''),
                    )
                    for match in text_ev.paragraph_matches
                ]
                conn.executemany("""
                    INSERT INTO paragraph_matches (
                        pair_id, similarity, para_a_index, para_b_index,
                        detection_method, is_continuous_clone, clone_group_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, match_rows)

    def load_all_results(self) -> List[PairwiseResult]:
        """加载所有分析结果（含段落匹配+完整文本+高亮标记）"""
        from data_structures import (
            PairwiseResult, EvidenceChain, TextEvidence,
            MetadataEvidence, ImageEvidence
        )

        results = []
        cursor = self.conn.cursor()

        # 加载配对结果
        cursor.execute("SELECT * FROM pairwise_results ORDER BY risk_score DESC")
        for row in cursor.fetchall():
            cols = [
                'pair_id', 'doc_a_id', 'doc_b_id', 'text_similarity',
                'risk_level', 'risk_score', 'risk_factors_json',
                'match_count', 'clone_block_count', 'metadata_match_count',
                'image_match_count', 'evidence_json', 'processed_at'
            ]
            data = dict(zip(cols, row))

            # 加载段落匹配
            cursor2 = self.conn.cursor()
            cursor2.execute(
                "SELECT similarity, para_a_index, para_b_index, detection_method, "
                "is_continuous_clone, clone_group_id "
                "FROM paragraph_matches WHERE pair_id = ? ORDER BY similarity DESC",
                (data['pair_id'],)
            )
            para_matches = [
                {
                    'similarity': r[0],
                    'paragraph_a_index': r[1],
                    'paragraph_b_index': r[2],
                    'detection_method': r[3],
                    'is_continuous_clone': bool(r[4]),
                    'continuous_clone_group_id': r[5],
                    'paragraph_a': '',
                    'paragraph_b': '',
                    'highlighted_text_a': '',
                    'highlighted_text_b': '',
                    'common_parts': [],
                }
                for r in cursor2.fetchall()
            ]

            # 填充段落文本并计算高亮标记
            self._fill_paragraph_texts_and_highlights(
                para_matches, data['doc_a_id'], data['doc_b_id']
            )

            # 解析证据 JSON
            evidence_raw = json.loads(data['evidence_json'] or '{}')

            text_evidence = TextEvidence(
                local_similarity=data['text_similarity'] or 0.0,
                common_paragraphs=evidence_raw.get('common_paragraphs', []),
                paragraph_matches=para_matches,
                continuous_clone_blocks=evidence_raw.get('continuous_clone_blocks', []),
                detection_summary=evidence_raw.get('detection_summary', {}),
            )

            metadata_evidence = MetadataEvidence(
                matched_fields=evidence_raw.get('metadata_matched_fields', []),
                matched_values=evidence_raw.get('metadata_matched_values', {}),
                same_time_bucket=evidence_raw.get('same_time_bucket', False),
            )

            image_evidence = ImageEvidence(
                common_image_count=data['image_match_count'] or 0,
                common_image_hashes=evidence_raw.get('image_common_hashes', []),
                exact_image_count=evidence_raw.get('image_exact_count', 0),
                near_identical_count=evidence_raw.get('image_near_identical_count', 0),
                similar_image_count=evidence_raw.get('image_similar_count', 0),
                image_risk_score=evidence_raw.get('image_risk_score', 0),
                image_risk_factors=evidence_raw.get('image_risk_factors', []),
                ps_suspicious=evidence_raw.get('ps_suspicious', False),
                ps_suspicious_count=evidence_raw.get('ps_suspicious_count', 0),
                shared_typos=evidence_raw.get('shared_typos', []),
                shared_typo_count=evidence_raw.get('shared_typo_count', 0),
                text_identical_count=evidence_raw.get('text_identical_count', 0),
                text_similar_count=evidence_raw.get('text_similar_count', 0),
                matched_image_pairs=evidence_raw.get('matched_image_pairs', []),
                matched_text_pairs=evidence_raw.get('matched_text_pairs', []),
                ps_detail_list=evidence_raw.get('ps_detail_list', []),
            )

            results.append(PairwiseResult(
                pair_id=data['pair_id'],
                doc_a_id=data['doc_a_id'],
                doc_b_id=data['doc_b_id'],
                similarity_scores={'text_local': data['text_similarity'] or 0.0},
                risk_level=data['risk_level'] or 'NONE',
                risk_score=data['risk_score'] or 0,
                risk_factors=json.loads(data['risk_factors_json'] or '[]'),
                evidence=EvidenceChain(
                    text_evidence=text_evidence,
                    metadata_evidence=metadata_evidence,
                    image_evidence=image_evidence,
                ),
            ))

        return results

    # ================================================================
    # 辅助方法
    # ================================================================

    @staticmethod
    def _chunk_id(doc_id: str, chunk_index: int) -> str:
        return f"{doc_id}_chunk_{chunk_index:04d}"

    @staticmethod
    def _para_id(doc_id: str, para_index: int) -> str:
        return f"{doc_id}_para_{para_index:06d}"

    # ================================================================
    # 段落文本填充与高亮计算
    # ================================================================

    def _fill_paragraph_texts_and_highlights(
        self, para_matches: List[Dict], doc_a_id: str, doc_b_id: str
    ) -> None:
        """为段落匹配填充完整文本并计算高亮标记和共同部分

        从 paragraphs 表中加载段落文本，然后使用 difflib 计算
        【】高亮标记文本和共同文本片段。

        此方法原地修改 para_matches 列表中的每个字典。
        """
        if not para_matches:
            return

        # 批量加载两个文档的所有段落文本（避免逐条查询）
        texts_a = self._batch_load_paragraph_texts(doc_a_id)
        texts_b = self._batch_load_paragraph_texts(doc_b_id)

        for match in para_matches:
            idx_a = match.get('paragraph_a_index', -1)
            idx_b = match.get('paragraph_b_index', -1)

            # 从批量加载的字典中获取文本
            text_a = texts_a.get(idx_a, '')
            text_b = texts_b.get(idx_b, '')

            if text_a:
                match['paragraph_a'] = text_a
            if text_b:
                match['paragraph_b'] = text_b

            # 计算高亮文本和共同部分
            if text_a and text_b:
                hl_a, hl_b, common = compute_text_diff(text_a, text_b)
                match['highlighted_text_a'] = hl_a
                match['highlighted_text_b'] = hl_b
                match['common_parts'] = common

    def _batch_load_paragraph_texts(self, doc_id: str) -> Dict[int, str]:
        """批量加载文档的所有段落文本

        Returns:
            Dict[para_index, text]
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT para_index, text FROM paragraphs "
            "WHERE doc_id = ? AND text IS NOT NULL AND text != '' "
            "ORDER BY para_index",
            (doc_id,)
        )
        return {row[0]: row[1] for row in cursor.fetchall()}

    def clear_cache(self) -> None:
        """清空所有缓存数据（谨慎使用）"""
        with self.transaction() as conn:
            tables = ['paragraph_matches', 'pairwise_results', 'candidate_pairs',
                      'paragraphs', 'chunks', 'documents', 'pipeline_state']
            for table in tables:
                conn.execute(f"DELETE FROM {table}")
        logger.warning("缓存已清空")

    def vacuum(self) -> None:
        """压缩数据库文件"""
        self.conn.execute("PRAGMA optimize")
        self.conn.execute("VACUUM")
        logger.info("数据库已压缩优化")

    def close(self) -> None:
        """关闭数据库连接"""
        self.conn.close()
        logger.info("SQLite 缓存已关闭")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
