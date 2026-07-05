import subprocess, os
os.chdir(r'c:\dongyuhang\project\Bid collusion inspection')
cmd = 'git filter-branch -f --index-filter "git rm --cached --ignore-unmatch -qr input/ cache_old_1783042226/ cache_old_1783042654/ checkpoints_old/" --prune-empty -- --all'
subprocess.run(cmd, shell=True, check=False)
subprocess.run(['git', 'reflog', 'expire', '--expire=now', '--all'], check=False)
subprocess.run(['git', 'gc', '--prune=now'], check=False)
subprocess.run(['git', 'count-objects', '-vH'], check=False)
