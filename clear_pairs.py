import sqlite3
conn = sqlite3.connect('./cache/features.db')

# Clear all Phase 2/3 results to force re-analysis
conn.execute('DELETE FROM candidate_pairs')
conn.execute('DELETE FROM pairwise_results')
conn.execute('DELETE FROM paragraph_matches')
conn.execute('UPDATE documents SET processed = 0')
# Clear pipeline_state to reset Phase 3 progress
conn.execute("DELETE FROM pipeline_state WHERE key LIKE '%phase3%' OR key LIKE '%completed%'")
conn.commit()

# Verify
for table in ['candidate_pairs', 'pairwise_results', 'paragraph_matches']:
    cur = conn.execute(f'SELECT COUNT(*) FROM {table}')
    print(f'{table}: {cur.fetchone()[0]} rows')

cur = conn.execute('SELECT doc_id, doc_minhash, image_hash_count FROM documents')
for row in cur.fetchall():
    mh = row[1] or ''
    print(f'  doc: {row[0][:8]}..., minhash_len={len(mh)}, img_count={row[2]}')
conn.close()
print('Done - ready for re-run')
