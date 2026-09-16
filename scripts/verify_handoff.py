"""Run after extraction, before regenerating reports or other packaged files."""
from pathlib import Path
import hashlib
import json

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'handoff-manifest.json').read_text(encoding='utf-8'))
errors=[]
for name,expected in manifest['files'].items():
    path=root/name
    if not path.exists(): errors.append(f'missing: {name}'); continue
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != expected: errors.append(f'changed: {name}')
print(json.dumps(dict(status='passed' if not errors else 'failed',files=len(manifest['files']),errors=errors),indent=2))
raise SystemExit(bool(errors))
