@echo off
cd /d c:\dongyuhang\project\"Bid collusion inspection"

echo Step 0: Committing current changes...
git add -A
git commit -m "chore: stage changes before history rewrite"

echo Step 1: Rewriting history...
set FILTER_BRANCH_SQUELCH_WARNING=1
git filter-branch -f --index-filter "git rm --cached --ignore-unmatch -qr input/ cache_old_1783042226/ cache_old_1783042654/ checkpoints_old/" --prune-empty -- --all

echo Step 2: Cleaning refs...
git for-each-ref --format="%%(refname)" refs/original/ > refs.txt
for /f "delims=" %%i in (refs.txt) do git update-ref -d "%%i"
del refs.txt

echo Step 3: Expiring reflog...
git reflog expire --expire=now --all

echo Step 4: GC...
git gc --aggressive --prune=now

echo Done!
git count-objects -vH
