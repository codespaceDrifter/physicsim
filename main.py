# Run the backend (python -m api in back/) and the frontend dev server (npm run dev in front/) together.
# run from repo root:  python3 main.py   then open http://localhost:5173. ctrl+c stops both.
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent

back = subprocess.Popen([sys.executable, '-m', 'api'], cwd=ROOT / 'back')
front = subprocess.Popen(['npm', 'run', 'dev'], cwd=ROOT / 'front')
print('backend on http://localhost:8080, editor on http://localhost:5173 (ctrl+c to stop)')
try:
    # if either one dies, stop the other too
    while back.poll() is None and front.poll() is None:
        time.sleep(0.5)
    print(f'backend exited with {back.poll()}' if back.poll() is not None else f'frontend exited with {front.poll()}')
except KeyboardInterrupt:
    pass
finally:
    for p in (back, front):
        if p.poll() is None:
            p.terminate()
    for p in (back, front):
        p.wait()
