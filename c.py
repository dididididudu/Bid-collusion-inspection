import subprocess, os
os.chdir(r'c:\dongyuhang\project\Bid collusion inspection')
subprocess.run(['git','add','-A'], check=True)
subprocess.run(['git','commit','-m','fix gpu'], check=True)
