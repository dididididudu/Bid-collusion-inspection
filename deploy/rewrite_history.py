#!/usr/bin/env python3
"""Rewrite git history via fast-export/fast-import to remove large directories."""
import subprocess, os, sys

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(REPO)

REMOVE = ('input/', 'cache_old_1783042226/', 'cache_old_1783042654/', 'checkpoints_old/')

def run(cmd, input_data=None):
    kw = {'cwd': REPO, 'capture_output': True, 'text': True}
    if input_data:
        kw['input'] = input_data
    r = subprocess.run(cmd, **kw)
    if r.returncode != 0:
        print(f"ERR: {r.stderr}", file=sys.stderr)
        sys.exit(1)
    return r

def keep(line):
    for prefix in ('M ', 'D ', 'C ', 'R '):
        if line.startswith(prefix):
            parts = line.split('\t')
            if len(parts) >= 2:
                path = parts[-1].strip()
                for rm in REMOVE:
                    if path.startswith(rm):
                        return False
    return True

print("Exporting...")
data = run(['git', 'fast-export', '--all', '--show-original-ids']).stdout

lines = data.splitlines(keepends=True)
print(f"Total lines: {len(lines)}")

kept = [l for l in lines if keep(l)]
removed = len(lines) - len(kept)
print(f"Removed: {removed}, Kept: {len(kept)}")

print("Backing up current HEAD...")
head = run(['git', 'rev-parse', 'HEAD']).stdout.strip()

print("Importing filtered history...")
run(['git', 'update-ref', '-d', 'refs/heads/master'])
proc = subprocess.run(['git', 'fast-import', '--force'],
    input=''.join(kept), cwd=REPO, capture_output=True, text=True)

if proc.returncode != 0:
    print(f"FAILED: {proc.stderr}")
    run(['git', 'update-ref', 'refs/heads/master', head])
    sys.exit(1)

print("Checking out master...")
run(['git', 'checkout', 'master', '-f'])

print("Cleaning up...")
run(['git', 'reflog', 'expire', '--expire=now', '--all'])
run(['git', 'gc', '--aggressive', '--prune=now'])

print("Done! Verifying...")
run(['git', 'count-objects', '-vH'])
r = run(['git', 'ls-tree', '-r', '-l', 'HEAD'])
total, cnt = 0, 0
for line in r.stdout.splitlines():
    parts = line.split()
    if len(parts) >= 4 and parts[1] == 'blob':
        try:
            total += int(parts[3])
            cnt += 1
        except ValueError:
            pass
print(f"Tracked: {cnt} files, {total/1024/1024:.2f} MB")
