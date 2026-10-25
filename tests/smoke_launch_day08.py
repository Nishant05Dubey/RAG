"""Smoke-check the real launch command; stop only the server this script starts."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

root = Path(__file__).resolve().parents[1]
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
output = root / 'outputs/day08/web_launch_verification.txt'
env = dict(os.environ, HF_HUB_OFFLINE='1')
with output.open('w', encoding='utf-8') as log:
    proc = subprocess.Popen([sys.executable, str(root/'run_rag.py'), 'web', '--port', str(port)],
                            cwd=root, env=env, stdout=log, stderr=log)
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError('Server exited before readiness; inspect log')
            try:
                with opener.open(f'http://127.0.0.1:{port}/', timeout=2) as response:
                    assert response.status == 200
                    assert 'Blend RAG Assistant' in response.read().decode()
                with opener.open(f'http://127.0.0.1:{port}/api/health', timeout=8) as response:
                    health = json.load(response)
                    assert health['ok'], health
                print('PASS: python run_rag.py web; HTTP / = 200; health = ready; model =', health['ollama']['model'])
                break
            except (OSError, TimeoutError):
                time.sleep(1)
        else:
            raise RuntimeError('Web launch timed out')
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
