#!/bin/bash
# 清理 Git 历史中的大文件
set -e
cd "$(dirname "$0")/.."

echo "=== 1. 重写历史，移除大文件 ==="
git filter-branch -f --index-filter \
  "git rm --cached --ignore-unmatch -qr input/ cache_old_1783042226/ cache_old_1783042654/ checkpoints_old/" \
  --prune-empty -- --all

echo "=== 2. 清理原始引用 ==="
git for-each-ref --format="%(refname)" refs/original/ | xargs -n 1 git update-ref -d

echo "=== 3. 过期 reflog ==="
git reflog expire --expire=now --all

echo "=== 4. 垃圾回收 ==="
git gc --aggressive --prune=now

echo "=== 完成 ==="
echo ""
echo "=== 验证 ==="
git count-objects -vH
echo ""
git ls-tree -r -l HEAD | awk '{sum+=$4; count++} END{printf "跟踪文件: %d 个, 总计 %.2f MB\n", count, sum/1024/1024}'
