"""Offline validation of the publication payload; no training or network."""
from pathlib import Path
import ast
import csv
import json
import re
import struct

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'.git', '.venv', 'venv', 'vendor', 'runs', '__pycache__', 'mpl-cache'}
files = sorted(p for p in ROOT.rglob('*') if p.is_file() and not (set(p.relative_to(ROOT).parts) & EXCLUDED))
for p in files:
    assert p.suffix not in {'.pt', '.npz', '.hdf5', '.saving', '.log'}, p.relative_to(ROOT)
    assert p.stat().st_size < 2_000_000, p.relative_to(ROOT)
    if p.suffix == '.png':
        data = p.read_bytes()
        assert data[:8] == b'\x89PNG\r\n\x1a\n'
        width, height = struct.unpack('>II', data[16:24])
        assert width > 500 and height > 200
        continue
    s = p.read_text()
    for pattern in [r'/(?:Users|home)/[^/\s]+/', r'gh[pousr]_[A-Za-z0-9]{20,}', r'sk-[A-Za-z0-9]{20,}', r'-----BEGIN [A-Z ]*PRIVATE KEY-----']:
        assert not re.search(pattern, s), p.relative_to(ROOT)
    if p.suffix == '.py':
        ast.parse(s)
    if p.suffix == '.md':
        for link in re.findall(r'\]\(([^)]+)\)', s):
            if '://' in link or link.startswith('#'):
                continue
            target = (p.parent / link.split('#')[0]).resolve()
            if ROOT / 'vendor' in target.parents:
                continue  # Official source materializes at setup.
            assert target.exists(), (p.relative_to(ROOT), link)
for label, count in [('harmonic4', 600), ('harmonic32', 5000)]:
    folder = ROOT / 'results' / label
    m = json.loads((folder / 'metrics.json').read_text())
    with (folder / 'loss.csv').open() as h:
        assert len(list(csv.DictReader(h))) == count
    before, after = (m['before'], m['after']) if label == 'harmonic4' else (m['initial'], m['final'])
    assert before['reverse_KL_MC'] > after['reverse_KL_MC'] > 0
assert sum(p.stat().st_size for p in files) < 5_000_000
print(json.dumps({'status': 'passed', 'payload_files': len(files), 'payload_bytes': sum(p.stat().st_size for p in files),
                  'checks': ['Python syntax', 'relative links', 'PNG headers', 'loss row counts', 'result direction',
                             'size/exclusion rules', 'personal-path and common credential-pattern scan']}, indent=2))
