"""Fetch an unchanged, pinned official neuralCT source checkout (not vendored)."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'vendor' / 'neuralCT'
URL = 'https://github.com/li012589/neuralCT.git'
COMMIT = 'f5673f185a2b787f748cb72721eadebf907413a4'

def git(*args, cwd=None):
    return subprocess.check_output(['git', *args], cwd=cwd, text=True).strip()

if TARGET.exists():
    if git('rev-parse', 'HEAD', cwd=TARGET) != COMMIT:
        raise SystemExit('Existing upstream checkout differs; refusing to alter it.')
    if git('status', '--porcelain', cwd=TARGET):
        raise SystemExit('Existing upstream checkout has changes; refusing to alter it.')
else:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'clone', '--no-checkout', '--depth', '1', URL, str(TARGET)], check=True)
    if git('rev-parse', 'HEAD', cwd=TARGET) != COMMIT:
        subprocess.run(['git', 'fetch', '--depth', '1', 'origin', COMMIT], cwd=TARGET, check=True)
    subprocess.run(['git', 'checkout', '--detach', COMMIT], cwd=TARGET, check=True)
print('Official neuralCT source verified:', COMMIT)
