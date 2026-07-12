"""清除 Phase 3 缓存，强制重新分析所有候选对"""
import sqlite3
import os

db_path = './cache/features.db'
conn = sqlite3.connect(db_path)

# 删除 Phase 3 结果
conn.execute('DELETE FROM pairwise_results')
conn.execute('DELETE FROM paragraph_matches')

# 重置候选对处理状态
conn.execute('UPDATE candidate_pairs SET processed = 0')

conn.commit()

# 验证
for table in ['pairwise_results', 'paragraph_matches', 'candidate_pairs']:
    cur = conn.execute(f'SELECT COUNT(*) FROM {table}')
    count = cur.fetchone()[0]
    if table == 'candidate_pairs':
        cur2 = conn.execute(f'SELECT COUNT(*) FROM {table} WHERE processed = 0')
        unprocessed = cur2.fetchone()[0]
        print(f'{table}: {count} total, {unprocessed} unprocessed')
    else:
        print(f'{table}: {count} rows')

conn.close()
print('Phase 3 cache cleared - ready for re-run')
