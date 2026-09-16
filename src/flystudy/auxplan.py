"""Outcome-independent complete bundles for optional sensitivity studies."""
from dataclasses import asdict, replace
from pathlib import Path
import json
import numpy as np
from .protocol import Protocol, LANGUAGES, ORDERS, file_hash, write_json
from .graph import Graph
from .gates import code_hash
from .budget import Ledger
from .workflow import read, verified_evidence


def reserve_auxiliary(main_manifest, category, graph_path, other_path, tokenizer_path,
                      dataset, output, ledger_path, hours_per_run):
    main = read(main_manifest)
    if main.get('status') != 'frozen' or main['code_hash'] != code_hash():
        raise ValueError('Freeze the main design before reserving auxiliary bundles')
    evidence = verified_evidence(main)
    evidence[str(Path(main_manifest).resolve())] = file_hash(main_manifest)
    p = Protocol(**main['protocol'])
    graph = Graph.load(graph_path)
    if graph.hash != main['graph_hash'] or file_hash(tokenizer_path) != main['tokenizer_hash']:
        raise ValueError('Main artifacts differ from the frozen design')
    dataset_hash = read(Path(dataset)/'manifest.json')['dataset_hash']
    if dataset_hash != main['dataset_hash']:
        raise ValueError('Auxiliary dataset must match main study')
    seeds = (7001,7002)
    if set(seeds) & (set(p.main_seeds)|set(p.pilot_seeds)):
        raise ValueError('Auxiliary seeds overlap other cohorts')
    if category == 'shuffle':
        other = Graph.load(other_path)
        if other.provenance.get('condition') != 'degree_preserving' or other.provenance.get('source') != graph.hash:
            raise ValueError('Expected a documented shuffle of the real graph')
        for before,after in ((graph.src,other.src),(graph.dst,other.dst)):
            np.testing.assert_array_equal(np.bincount(before,minlength=graph.n),np.bincount(after,minlength=other.n))
        np.testing.assert_array_equal(graph.inputs,other.inputs)
        np.testing.assert_array_equal(graph.outputs,other.outputs)
        variants=[('real',p,graph_path,tokenizer_path),('shuffle',p,other_path,tokenizer_path)]
        conditions=[('sequential',o) for o in ORDERS]
    elif category == 'tokenizer':
        from tokenizers import Tokenizer
        small = read(Path(other_path).with_suffix('.meta.json'))
        if small['dataset_hash'] != dataset_hash or small['tokenizer_hash'] != file_hash(other_path) or Tokenizer.from_file(str(other_path)).get_vocab_size() != 1024:
            raise ValueError('Expected a training-only 1,024-token tokenizer from this dataset')
        variants=[(f'v{v}-m{m}',replace(p,vocab_size=v,microsteps=m),graph_path,t)
                  for v,t in ((1024,other_path),(4096,tokenizer_path)) for m in (2,4)]
        conditions=[('mono',(lang,)) for lang in LANGUAGES]
    else:
        raise ValueError('Unknown auxiliary category')
    root = Path(output)
    if root.exists() and any(root.iterdir()): raise FileExistsError(root)
    runs=[]
    for name,config,g,t in variants:
        config_path=root/f'{name}.json'
        for seed in seeds:
            for mode,order in conditions:
                rid=f'aux-{category}-{name}-{seed}-{mode}-'+ '-'.join(order)
                runs.append(dict(run_id=rid,cohort='auxiliary',seed=seed,mode=mode,order=list(order),variant=name,
                    protocol_path=str(config_path.resolve()),graph_path=str(Path(g).resolve()),
                    tokenizer_path=str(Path(t).resolve()),dataset_path=str(Path(dataset).resolve()),
                    protocol_hash=config.hash,graph_hash=Graph.load(g).hash,tokenizer_hash=file_hash(t),dataset_hash=dataset_hash))
    # Reserve the entire paired bundle before any model is trained.
    Ledger(ledger_path).reserve_bundle([dict(run_id=r['run_id'],category=category,hours=hours_per_run*1.25) for r in runs])
    for name,config,g,t in variants: write_json(root/f'{name}.json',asdict(config))
    manifest=dict(status='auxiliary_reserved',category=category,code_hash=code_hash(),runs=runs,
                  seeds=list(seeds),evidence_files=evidence,selection_uses_outcomes=False,
                  interpretation='Exploratory paired bundle; one graph shuffle realization does not estimate topology-population variance.')
    manifest.update({k: main.get(k) for k in ('review_mode', 'human_reviewed', 'review_limitations', 'protocol_amendment')})
    write_json(root/'bundle.json',manifest)
    return manifest
