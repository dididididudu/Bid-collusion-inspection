cd /c/dongyuhang/project/Bid\ collusion\ inspection
CMD="git filter-branch -f --index-filter \"git rm --cached --ignore-unmatch -qr input/ cache_old_1783042226/ cache_old_1783042654/ checkpoints_old/\" --prune-empty -- --all"
eval $CMD
git for-each-ref --format="%(refname)" refs/original/ | xargs -n 1 git update-ref -d
git reflog expire --expire=now --all
git gc --aggressive --prune=now
git count-objects -vH
