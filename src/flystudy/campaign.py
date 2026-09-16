"""Two independent GPU workers, complete seed blocks, stop dispatch on failure."""
from __future__ import annotations
from pathlib import Path
import json
import subprocess
import sys
import time
from .protocol import Protocol, file_hash
from .gates import code_hash
from .graph import Graph


def campaign(protocol, graph, data, tokenizer, manifest, matrix, ledger, output, devices, microbatch=32):
    if len(devices)!=2 or len(set(devices))!=2:
        raise ValueError("Specify two distinct CUDA devices")
    schedule=json.loads(Path(matrix).read_text())
    runs=schedule["runs"]
    expected = dict(protocol_hash=Protocol.load(protocol).hash, graph_hash=Graph.load(graph).hash,
                    tokenizer_hash=file_hash(tokenizer), code_hash=code_hash(),
                    dataset_hash=json.loads((Path(data)/'manifest.json').read_text())['dataset_hash'])
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    for seed in sorted({r["seed"] for r in runs}):
        block=[r for r in runs if r["seed"]==seed]
        if len(block)!=10:
            raise ValueError("Campaign scheduling unit is a complete 10-condition seed block")
        pending=list(block); active={}; failed=False
        while pending or active:
            for device in devices:
                if device in active or not pending or failed: continue
                run=pending.pop(0); folder=output/run["run_id"]
                if (folder/"summary.json").exists():
                    row=json.loads((folder/"summary.json").read_text())
                    if row.get('run_id') != run['run_id'] or any(row.get(k) != v for k,v in expected.items()):
                        raise ValueError('Existing run does not match this campaign')
                    if row["stop_reason"] in ("mastered","administrative_cap"):
                        continue
                    raise RuntimeError(f"Explicit failure recovery required for {run['run_id']}")
                command=[sys.executable,"-m","flystudy","run","--protocol",str(protocol),"--graph",str(graph),
                         "--data",str(data),"--tokenizer",str(tokenizer),"--manifest",str(manifest),"--matrix",str(matrix),
                         "--ledger",str(ledger),"--output",str(folder),"--run-id",run["run_id"],"--device",device,
                         "--microbatch",str(microbatch)]
                log=(output/(run["run_id"]+".console.log")).open("w",encoding="utf-8")
                flags=subprocess.CREATE_NO_WINDOW if sys.platform=="win32" else 0
                active[device]=(subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,creationflags=flags),log,run["run_id"])
            for device,(process,log,rid) in list(active.items()):
                code=process.poll()
                if code is not None:
                    log.close(); del active[device]
                    if code:
                        failed=True
            if failed and not active:
                raise RuntimeError("Campaign stopped after a failed run; incomplete blocks are not censored observations")
            time.sleep(.2)
    return dict(status="completed",runs=len(runs))
