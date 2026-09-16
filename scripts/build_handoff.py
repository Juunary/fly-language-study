"""Package only current artifacts; verify every ZIP entry with CRC and SHA256."""
from pathlib import Path
from datetime import datetime,timezone
from importlib.metadata import version
import json
import xml.etree.ElementTree as ET
import zipfile
from flystudy.gates import code_hash,environment
from flystudy.protocol import file_hash,write_json
from flystudy.graph import Graph
from flystudy.model import FlyClassifier

root=Path(__file__).resolve().parents[1]
data=root/'data/draft-v4.3'
graph=Graph.load(root/'artifacts/graphs/real.npz')
audit=json.loads((data/'audit.json').read_text())
token=json.loads((root/'artifacts/tokenizer-v4.3.meta.json').read_text())
power=json.loads((root/'reports/power-summary.json').read_text())
tests=ET.parse(root/'reports/cpu-tests.xml').getroot()
suites=list(tests.iter('testsuite'))
assert all(int(s.get('failures',0))==0 and int(s.get('errors',0))==0 for s in suites)
assert audit['passed'] and audit['examples']==96000 and token['vocab_actual']==4096
g0=json.loads((root/'reports/g0-local.json').read_text())
assert g0['status']=='blocked' and g0['environment']['code_hash']==code_hash()
params=sum(p.numel() for p in FlyClassifier(graph,4096).parameters())
status=dict(packaged_at_utc=datetime.now(timezone.utc).isoformat(),phase='implementation_and_cpu_validation',
    code_hash=code_hash(),dataset_version='draft-v4.3',dataset_hash=audit['dataset_hash'],
    graph_hash=graph.hash,tokenizer_hash=token['tokenizer_hash'],trainable_parameters=params,
    neurons=graph.n,edges=len(graph.src),examples=audit['examples'],
    cpu_tests=dict(tests=sum(int(s.get('tests',0)) for s in suites),failures=0,errors=0,
                   skipped=sum(int(s.get('skipped',0)) for s in suites)),
    local_environment=environment(),cuda_g0='blocked_no_two_sm86_devices',human_review='pending',
    g1='not_run_on_cuda',g2='not_run',g3='not_run',sequential_pilots='not_run',main='not_started',
    power=power,observed_study_gpu_hours=0,
    baseline_notes='Nuisance 50% in all 12 cells; hypothesis-only roles 54.4%, other cells 50%; grammar roles 100%. These are corpus diagnostics, not FlyGPT performance.',
    limitations=['Draft corpus requires two human reviewers per language.',
        'Whole-string BPE creates different train/evaluation lengths; metadata reports all language/task lengths.',
        'Template holdout means outer construction families; embedded grammar is shared.',
        'CUDA compatibility, learnability, power feasibility and wall time remain unmeasured.'])
write_json(root/'reports/implementation-status.json',status)
packages=['numpy','scipy','torch','tokenizers','safetensors','matplotlib','threadpoolctl','pytest']
(root/'requirements-cpu-observed.txt').write_text('# Observed local CPU versions, NOT a CUDA server installation recipe.\n'+'\n'.join(f'{p}=={version(p)}' for p in packages)+'\n',encoding='utf-8')
files=[]
for path in root.rglob('*'):
    if not path.is_file(): continue
    rel=path.relative_to(root)
    if any(part in ('__pycache__','.pytest_cache','.venv','external') or part.endswith('.egg-info') for part in rel.parts): continue
    if rel.parts[0]=='data' and rel.parts[1]!='draft-v4.3': continue
    if rel.parts[0]=='artifacts' and str(rel).replace('\\','/') not in {
        'artifacts/graphs/real.npz','artifacts/graphs/real.json','artifacts/tokenizer-v4.3.json',
        'artifacts/tokenizer-v4.3.meta.json','artifacts/power/final-grid.jsonl','artifacts/power/final-grid.meta.json'}: continue
    if rel.parts[0] not in ('src','tests','scripts','configs','docs','reports','data','artifacts') and len(rel.parts)>1: continue
    if rel.name=='handoff-manifest.json' or rel.suffix in ('.partial','.zip'): continue
    files.append(path)
manifest={str(p.relative_to(root)).replace('\\','/'):file_hash(p) for p in sorted(files)}
write_json(root/'handoff-manifest.json',dict(files=manifest,code_hash=code_hash(),dataset_hash=audit['dataset_hash']))
files.append(root/'handoff-manifest.json')
destination=root.parent/'fly-language-study-handoff.zip'
with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for path in files: z.write(path,str(Path(root.name)/path.relative_to(root)))
with zipfile.ZipFile(destination) as z:
    assert z.testzip() is None
    import hashlib
    for rel,expected in manifest.items():
        assert hashlib.sha256(z.read(f'{root.name}/{rel}')).hexdigest()==expected
report=dict(path=str(destination),files=len(files),bytes=destination.stat().st_size,sha256=file_hash(destination),crc='passed',sha256_per_file='passed')
write_json(destination.with_suffix('.json'),report)
print(json.dumps(report,indent=2))
