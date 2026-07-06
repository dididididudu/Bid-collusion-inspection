#!/usr/bin/env python3
"""一次性应用所有性能优化补丁"""
import re, os

ROOT = r'c:\dongyuhang\project\Bid collusion inspection'

# ============================================================
# 1. feature_cache.py
# ============================================================
with open(os.path.join(ROOT, 'extraction/feature_cache.py'), encoding='utf-8') as f:
    fc = f.read()

# 1a. 插入 load_all_paragraphs_full
insert_method = '''
    def load_all_paragraphs_full(self, doc_id: str) -> Dict[int, dict]:
        """一次查询返回段落全部数据，替代 3 次独立查询"""
        import json as _json
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT para_index, minhash, text, tokens, source, embedding "
            "FROM paragraphs WHERE doc_id = ? AND text IS NOT NULL AND text != '' "
            "ORDER BY para_index", (doc_id,))
        result = {}
        for row in cursor.fetchall():
            idx, mh, text, tokens_raw, source, emb_bytes = row
            tokens = []
            if tokens_raw:
                try: tokens = _json.loads(tokens_raw)
                except: pass
            embedding = None
            if emb_bytes:
                try: embedding = np.frombuffer(emb_bytes, dtype=np.float32)
                except: pass
            result[idx] = {'minhash': mh or '', 'text': text or '',
                           'tokens': tokens, 'source': source or 'text', 'embedding': embedding}
        return result

    def load_all_paragraphs_with_tokens(self, doc_id: str) -> Dict[int, dict]:
        import json as _json
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT para_index, text, tokens FROM paragraphs WHERE doc_id = ? "
            "AND text IS NOT NULL AND text != '' ORDER BY para_index", (doc_id,))
        result = {}
        for row in cursor.fetchall():
            idx, text, tokens_raw = row
            tokens = []
            if tokens_raw:
                try: tokens = _json.loads(tokens_raw)
                except: pass
            result[idx] = {'text': text, 'tokens': tokens}
        return result

'''
# Insert before load_all_paragraph_minhashes
marker = 'def load_all_paragraph_minhashes(self, doc_id: str) -> Dict[int, str]:'
fc = fc.replace(marker, insert_method + '\n    ' + marker)

# 1b. Fix store_chunk for batch mode
old_store = 'def store_chunk(self, chunk_result: ChunkResult) -> None:\n        """存储文本块（文本内容 zlib 压缩）"""'
new_store = 'def store_chunk(self, chunk_result: ChunkResult, conn=None) -> None:\n        """存储文本块，conn 参数支持批量事务"""'
fc = fc.replace(old_store, new_store)

# Use own transaction only if no external conn
old_tx = 'with self.transaction() as conn:'
new_tx = '''        _conn = conn if conn is not None else self.conn
        _own_tx = conn is None
        if _own_tx:
            _conn.execute("BEGIN IMMEDIATE")
        try:'''
fc = fc.replace(old_tx, new_tx, 1)  # Only first occurrence

# Add commit/rollback at end of store_chunk
old_end = '''            conn.executemany("""
                INSERT OR REPLACE INTO paragraphs (
                    para_id, doc_id, chunk_id, para_index, text, minhash, source, tokens
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, para_rows)'''
new_end = '''            _conn.executemany("""
                INSERT OR REPLACE INTO paragraphs (
                    para_id, doc_id, chunk_id, para_index, text, minhash, source, tokens
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, para_rows)

            if _own_tx:
                _conn.execute("COMMIT")
        except Exception:
            if _own_tx:
                _conn.execute("ROLLBACK")
            raise'''
fc = fc.replace(old_end + '\n', new_end + '\n', 1)

# 1c. Add tokens column to paragraphs table schema
old_schema = 'minhash TEXT DEFAULT "",\n                FOREIGN KEY (doc_id) REFERENCES documents(doc_id),'
new_schema = 'minhash TEXT DEFAULT "",\n                tokens TEXT DEFAULT "",\n                FOREIGN KEY (doc_id) REFERENCES documents(doc_id),'
fc = fc.replace(old_schema, new_schema, 1)

# Also add source column
old_para1 = 'INSERT OR REPLACE INTO paragraphs (\n                    para_id, doc_id, chunk_id, para_index, text, minhash\n                )'
new_para1 = 'INSERT OR REPLACE INTO paragraphs (\n                    para_id, doc_id, chunk_id, para_index, text, minhash, source\n                )'
fc = fc.replace(old_para1, new_para1, 1)
old_para1v = 'VALUES (?, ?, ?, ?, ?, ?)'
new_para1v = 'VALUES (?, ?, ?, ?, ?, ?, ?)'
fc = fc.replace(old_para1v, new_para1v, 1)

with open(os.path.join(ROOT, 'extraction/feature_cache.py'), 'w', encoding='utf-8') as f:
    f.write(fc)
print('[OK] feature_cache.py')

# ============================================================
# 2. paragraph_matcher.py — 合并查询 + 消除回填
# ============================================================
with open(os.path.join(ROOT, 'matching/paragraph_matcher.py'), encoding='utf-8') as f:
    pm = f.read()

# 2a. Use load_all_paragraphs_full
old_load = '''        # Load all paragraph MinHash signatures from SQLite
        minhashes_a = cache.load_all_paragraph_minhashes(doc_a.doc_id)
        minhashes_b = cache.load_all_paragraph_minhashes(doc_b.doc_id)'''
new_load = '''        # 一次性加载全部段落数据（minhash + text + tokens + source）
        para_full_a = cache.load_all_paragraphs_full(doc_a.doc_id)
        para_full_b = cache.load_all_paragraphs_full(doc_b.doc_id)

        minhashes_a = {k: v['minhash'] for k, v in para_full_a.items() if v['minhash']}
        minhashes_b = {k: v['minhash'] for k, v in para_full_b.items() if v['minhash']}'''
pm = pm.replace(old_load, new_load)

# 2b. Filter sources from para_full
old_src = '''        # 过滤跨类型候选对（OCR 段落只和 OCR 段落匹配，文本只和文本匹配）
        source_a = cache.get_paragraph_source_map(doc_a.doc_id)
        source_b = cache.get_paragraph_source_map(doc_b.doc_id)
        before = len(stage1_candidates)
        stage1_candidates = [
            (i, j, sim) for i, j, sim in stage1_candidates
            if source_a.get(i, 'text') == source_b.get(j, 'text')
        ]'''
new_src = '''        # 过滤跨类型候选对（从 para_full 读取 source，无需额外查询）
        before = len(stage1_candidates)
        stage1_candidates = [
            (i, j, sim) for i, j, sim in stage1_candidates
            if para_full_a.get(i, {}).get('source', 'text') == para_full_b.get(j, {}).get('source', 'text')
        ]'''
pm = pm.replace(old_src, new_src)

# 2c. Stage 2 uses para_full instead of load_all_paragraphs_with_tokens
old_stage2load = '''        if self.semantic_matcher.is_available:
            # 一次性加载所有段落的文本和预分词（避免逐个查询 + 重复 jieba）
            para_data_a = cache.load_all_paragraphs_with_tokens(doc_a.doc_id)
            para_data_b = cache.load_all_paragraphs_with_tokens(doc_b.doc_id)

            # 构建索引：段落 → (text, word_set)
            para_texts_a = {}
            para_texts_b = {}
            word_sets_a = {}
            word_sets_b = {}
            import json as _json

            for i, j, _ in stage1_candidates:
                if i not in para_texts_a and i in para_data_a:
                    pd = para_data_a[i]
                    para_texts_a[i] = pd['text']
                    tokens = pd['tokens']
                    if tokens:
                        word_sets_a[i] = {w for w in tokens if len(w) > 1}
                    else:
                        # 兼容旧数据：无预分词时 jieba 补算
                        word_sets_a[i] = {w for w in jieba.cut(pd['text']) if len(w) > 1}
                if j not in para_texts_b and j in para_data_b:
                    pd = para_data_b[j]
                    para_texts_b[j] = pd['text']
                    tokens = pd['tokens']
                    if tokens:
                        word_sets_b[j] = {w for w in tokens if len(w) > 1}
                    else:
                        word_sets_b[j] = {w for w in jieba.cut(pd['text']) if len(w) > 1}'''

new_stage2load = '''        if self.semantic_matcher.is_available:
            # 从 para_full 直接读取（已在 match 开头一次查询加载）
            para_texts_a = {}
            para_texts_b = {}
            word_sets_a = {}
            word_sets_b = {}

            for i, j, _ in stage1_candidates:
                if i not in para_texts_a and i in para_full_a:
                    pd = para_full_a[i]
                    para_texts_a[i] = pd['text']
                    tokens = pd['tokens']
                    word_sets_a[i] = {w for w in tokens if len(w) > 1} if tokens else {w for w in jieba.cut(pd['text']) if len(w) > 1}
                if j not in para_texts_b and j in para_full_b:
                    pd = para_full_b[j]
                    para_texts_b[j] = pd['text']
                    tokens = pd['tokens']
                    word_sets_b[j] = {w for w in tokens if len(w) > 1} if tokens else {w for w in jieba.cut(pd['text']) if len(w) > 1}'''
pm = pm.replace(old_stage2load, new_stage2load)

# 2d. Eliminate per-result text backfill queries
old_backfill = '''            if not result.get('paragraph_a'):
                result['paragraph_a'] = cache.load_paragraph_text(doc_a.doc_id, i) or ''
            if not result.get('paragraph_b'):
                result['paragraph_b'] = cache.load_paragraph_text(doc_b.doc_id, j) or \'\''''
new_backfill = '''            # 从已加载的 para_full 获取（无 SQLite 查询）
            if not result.get('paragraph_a'):
                result['paragraph_a'] = para_full_a.get(i, {}).get('text', '')
            if not result.get('paragraph_b'):
                result['paragraph_b'] = para_full_b.get(j, {}).get('text', '')'''
pm = pm.replace(old_backfill, new_backfill)

with open(os.path.join(ROOT, 'matching/paragraph_matcher.py'), 'w', encoding='utf-8') as f:
    f.write(pm)
print('[OK] paragraph_matcher.py')

# ============================================================
# 3. semantic_matcher.py — vectorize score_pairs_from_cache
# ============================================================
with open(os.path.join(ROOT, 'matching/semantic_matcher.py'), encoding='utf-8') as f:
    sm = f.read()

old_vec = '''        # 自适应阈值
        n_candidates = len(candidates)
        if n_candidates > 300:
            base_threshold_adj = 0.03
        elif n_candidates < 30:
            base_threshold_adj = -0.03
        else:
            base_threshold_adj = 0.0

        results = []
        for i, j, jaccard_sim in candidates:
            emb_a = embs_a.get(i)
            emb_b = embs_b.get(j)
            if emb_a is None or emb_b is None:
                continue

            # 向量化余弦相似度（纯 numpy 点积，极快）
            norm_a = np.linalg.norm(emb_a)
            norm_b = np.linalg.norm(emb_b)
            if norm_a == 0 or norm_b == 0:
                continue
            sim = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))'''

new_vec = '''        # 构建对齐嵌入矩阵（向量化批量余弦相似度）
        valid_pairs = []
        for i, j, jaccard_sim in candidates:
            ea, eb = embs_a.get(i), embs_b.get(j)
            if ea is not None and eb is not None:
                valid_pairs.append((i, j, jaccard_sim, ea, eb))

        if not valid_pairs:
            return []

        emb_a_mat = np.stack([p[3] for p in valid_pairs])
        emb_b_mat = np.stack([p[4] for p in valid_pairs])
        na = np.linalg.norm(emb_a_mat, axis=1); na[na == 0] = 1.0
        nb = np.linalg.norm(emb_b_mat, axis=1); nb[nb == 0] = 1.0
        all_sims = np.sum(emb_a_mat * emb_b_mat, axis=1) / (na * nb)

        # 自适应阈值
        n_candidates = len(candidates)
        if n_candidates > 300:
            base_threshold_adj = 0.03
        elif n_candidates < 30:
            base_threshold_adj = -0.03
        else:
            base_threshold_adj = 0.0

        results = []
        for idx, (i, j, jaccard_sim, _ea, _eb) in enumerate(valid_pairs):
            sim = float(all_sims[idx])'''

if old_vec in sm:
    sm = sm.replace(old_vec, new_vec)
    print('[OK] semantic_matcher.py')
else:
    print('[WARN] semantic_matcher.py — pattern not found (may already be fixed)')

with open(os.path.join(ROOT, 'matching/semantic_matcher.py'), 'w', encoding='utf-8') as f:
    f.write(sm)

print('\nAll fixes applied.')
