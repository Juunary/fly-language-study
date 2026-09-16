from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import csv
import importlib.metadata
import json
import platform
import shutil
import subprocess
import time

import numpy as np
import torch

from .protocol import LANGUAGES, TASKS, ORDERS, UPSTREAM, digest, file_hash, write_json
from .statistics import exact_censor_upper


def code_hash():
    return digest({p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def environment():
    return dict(python=platform.python_version(), system=platform.platform(), torch=str(torch.__version__),
                torch_cuda=torch.version.cuda, nvcc=shutil.which("nvcc"),
                cuda_devices=[dict(index=i, name=torch.cuda.get_device_name(i),
                              capability=list(torch.cuda.get_device_capability(i)),
                              vram=torch.cuda.get_device_properties(i).total_memory)
                              for i in range(torch.cuda.device_count())], code_hash=code_hash())


def verify_kernels_revision():
    try:
        dist = importlib.metadata.distribution("connectome-kernels")
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        revision = direct.get("vcs_info", {}).get("commit_id")
        if revision == UPSTREAM["kernels"]:
            return revision
        if direct.get("dir_info", {}).get("editable"):
            from urllib.parse import unquote, urlparse
            root = Path(unquote(urlparse(direct["url"]).path).lstrip("/") if platform.system() == "Windows" else unquote(urlparse(direct["url"]).path))
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            clean = not subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
            if revision == UPSTREAM["kernels"] and clean:
                return revision
    except (importlib.metadata.PackageNotFoundError, OSError, subprocess.CalledProcessError):
        pass
    raise RuntimeError("Install the exact pinned connectome-kernels revision; unproven builds cannot pass G0")


def g0(graph, output):
    from .graph import synthetic
    from .model import FlyClassifier, train_batch
    from .protocol import Protocol
    from .runtime import immutable_evaluation, equal_state
    report = dict(gate="G0", status="blocked", environment=environment(), graph_hash=graph.hash,
                  graph_condition=graph.provenance.get("condition"), checks=[])
    try:
        if graph.n != 5000 or graph.provenance.get("condition") != "real":
            raise RuntimeError("G0 requires the actual cb5k graph")
        if torch.cuda.device_count() != 2 or any(torch.cuda.get_device_capability(i) != (8, 6) for i in range(2)):
            raise RuntimeError("Two sm_86 CUDA GPUs required; missing hardware is BLOCKED, never PASS")
        report["kernel_revision"] = verify_kernels_revision()
        from connectome_kernels import available
        if not available():
            raise RuntimeError("Compiled kernel unavailable")
        for device_index in range(2):
            device = f"cuda:{device_index}"
            torch.cuda.set_device(device_index)
            for candidate, reference, batches, times in ((synthetic(16), "dense", (4, 32, 40), (7, 13)),
                                                        (graph, "sparse", (1, 4, 32, 128, 256), (7,)),
                                                        (graph, "sparse", (4,), (128,))):
                for batch in batches:
                    for steps in (2, 4):
                        for length in times:
                            a = FlyClassifier(candidate, 32, 8, steps, seed=9, backend="cuda").to(device)
                            b = FlyClassifier(candidate, 32, 8, steps, seed=9, backend=reference).to(device)
                            x = torch.randint(0, 32, (batch, length), device=device)
                            sizes = torch.arange(batch, device=device) % length + 1
                            state_a = torch.randn(batch, candidate.n, device=device, requires_grad=True)
                            state_b = state_a.detach().clone().requires_grad_()
                            ya, yb = a(x, sizes, state_a), b(x, sizes, state_b)
                            torch.testing.assert_close(ya, yb, atol=1e-4, rtol=1e-4)
                            ya.square().sum().backward(); yb.square().sum().backward()
                            for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
                                tolerance = 1e-5*max(float(pb.grad.abs().max()), 1.)
                                torch.testing.assert_close(pa.grad, pb.grad, atol=tolerance, rtol=1e-4)
                            torch.testing.assert_close(state_a.grad, state_b.grad, atol=1e-4, rtol=1e-4)
                            report["checks"].append(dict(device=device, neurons=candidate.n, batch=batch,
                                                         length=length, microsteps=steps, output_and_all_gradients=True))
            # Full classifier accumulation and immutable evaluation on the actual CUDA path.
            tiny = synthetic(16)
            a = FlyClassifier(tiny, 32, 8, 4, 19, "cuda").to(device)
            b = FlyClassifier(tiny, 32, 8, 4, 19, "cuda").to(device)
            p = Protocol()
            oa = torch.optim.AdamW(a.parameter_groups(p), weight_decay=p.weight_decay)
            ob = torch.optim.AdamW(b.parameter_groups(p), weight_decay=p.weight_decay)
            examples = [([1,2,3,4][:i%4+1], i%2) for i in range(11)]
            train_batch(a, oa, examples, 0, 11, p); train_batch(b, ob, examples, 0, 3, p)
            for pa, pb in zip(a.parameters(), b.parameters()):
                torch.testing.assert_close(pa, pb, atol=1e-5, rtol=1e-4)
            with immutable_evaluation(a, oa):
                a(torch.tensor([[1,2,3]], device=device), torch.tensor([3], device=device))
        report["status"] = "passed"
    except Exception as exc:
        report["reason"] = f"{type(exc).__name__}: {exc}"
    write_json(output, report)
    return report


def certify_review(dataset, reviewed_csv, template_attestation, output, ai_evidence=None):
    from .data import audit
    dataset = Path(dataset)
    audit_report = audit(dataset)
    expected = list(csv.DictReader((dataset / "audit-sample.csv").open(encoding="utf-8-sig")))
    expected_ids = {r["id"] for r in expected}
    original_labels = {r["id"]: int(r["label"]) for r in expected}
    item_languages = {r["id"]: r["language"] for r in expected}
    rows = list(csv.DictReader(Path(reviewed_csv).open(encoding="utf-8-sig")))
    attest = json.loads(Path(template_attestation).read_text(encoding="utf-8"))
    provenance = dict(review_mode="human", human_reviewed=True)
    if ai_evidence is None and attest.get("review_mode", "human") != "human":
        raise ValueError("AI judgments require certify-ai-review, not human certification")
    if ai_evidence is not None:
        from .ai_review import validate_ai_evidence
        provenance = validate_ai_evidence(dataset, reviewed_csv, attest, ai_evidence)
    groups = defaultdict(dict)
    for row in rows:
        reviewer = row["reviewer"].strip()
        if not reviewer or reviewer.startswith("reviewer_") or row["id"] not in expected_ids or reviewer in groups[row["id"]]:
            raise ValueError("Missing/duplicate reviewer identity or changed audit sample")
        if row["judged_label"] not in ("0", "1") or row["fluent"].lower() not in ("yes", "true", "1", "no", "false", "0"):
            raise ValueError("Incomplete or unresolved review")
        groups[row["id"]][reviewer] = int(row["judged_label"])
    reviewers, agreements = {}, {}
    if set(groups) != expected_ids:
        raise ValueError("Missing audit items")
    for language in LANGUAGES:
        items=sorted(item for item in groups if item_languages[item] == language)
        people=sorted({name for item in items for name in groups[item]})
        if len(people) != 2 or any(set(groups[item]) != set(people) for item in items):
            raise ValueError("Each language needs two distinct reviewer codes on every sampled item")
        reviewers[language]=people
        labels=np.array([[groups[item][r] for r in people] for item in items])
        agreement=float((labels[:,0] == labels[:,1]).mean())
        chance=sum(float((labels[:,0]==x).mean()*(labels[:,1]==x).mean()) for x in (0,1))
        kappa=(agreement-chance)/(1-chance) if chance < 1 else None
        if len(items) != 200 or agreement < .95 or kappa is None or kappa < .8:
            raise ValueError("Each language requires 200 audited items, agreement >= .95, and kappa >= .8")
        agreements[language]=dict(items=len(items),agreement=agreement,kappa=kappa)
    unresolved={item for item,people in groups.items() if any(label != original_labels[item] for label in people.values())}
    unresolved.update(r["id"] for r in rows if r["fluent"].lower() in ("no","false","0"))
    for item in unresolved:
        resolution=attest.get("resolved_items",{}).get(item,{})
        if resolution.get("label") != original_labels[item] or resolution.get("fluent") is not True or not resolution.get("rationale"):
            raise ValueError("Preserve original judgments and resolve disagreements explicitly; changed labels require a new dataset version")
    if attest.get("reviewers") != reviewers or not attest.get("all_templates_and_forms_checked"):
        raise ValueError("Template/form review is incomplete")
    if attest.get("inventory_hash") != file_hash(dataset / "template-inventory.json"):
        raise ValueError("Template attestation belongs to a different version")
    report = dict(gate="ai_review" if ai_evidence is not None else "human_review", status="passed" if audit_report["passed"] else "failed",
                  dataset_hash=audit_report["dataset_hash"], reviewers=reviewers,
                  sample_items=len(groups), agreement_by_language=agreements,
                  reviewed_csv_hash=file_hash(reviewed_csv), attestation_hash=file_hash(template_attestation), **provenance)
    report.setdefault("evidence_files", {}).update({str(Path(p).resolve()): file_hash(p)
                                                   for p in (reviewed_csv, template_attestation)})
    if report["sample_items"] != 600:
        raise ValueError("Production review requires 200 items per language")
    write_json(output, report)
    return report


def summarize_pilot(paths, protocol, output):
    records = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    if not records or any(r.get("cohort") != "pilot" or r.get("graph_condition") != 'real' or r.get("synthetic") or r.get("seed") in protocol.main_seeds for r in records):
        raise ValueError("Only independent, real-graph pilot runs can inform gates")
    if any(r["stop_reason"] not in ("mastered", "administrative_cap") for r in records):
        raise ValueError("Technical/budget interruptions are not administrative censoring")
    if len({r["run_id"] for r in records}) != len(records):
        raise ValueError("Duplicate pilot observations")
    keys = ("dataset_hash", "training_hash", "graph_hash", "tokenizer_hash", "code_hash")
    for key in keys:
        if len({r[key] for r in records}) != 1:
            raise ValueError(f"Do not pool incompatible pilot versions: {key}")
    result = {key: records[0][key] for key in keys}
    result.update(gate="pilots", status="incomplete", groups={}, evidence_files={str(p): file_hash(p) for p in paths})
    result["eval_interval"] = records[0]["protocol"]["eval_interval"]
    for mode, order in [("mono", (x,)) for x in LANGUAGES]+[("mixed", LANGUAGES)]+[("sequential", x) for x in ORDERS]:
        subset = [r for r in records if r["mode"] == mode and tuple(r["order"]) == order]
        if len({r['seed'] for r in subset}) != len(subset):
            raise ValueError('Repeated seed is not an independent pilot replication')
        mastered = sum(r["stop_reason"] == "mastered" for r in subset)
        costs = [r["seen"] for r in subset]
        result["groups"][f"{mode}/{'-'.join(order)}"] = dict(n=len(subset), successes=mastered,
            censored=len(subset)-mastered, censor_upper_95=exact_censor_upper(len(subset)-mastered, len(subset)),
            restricted_costs=costs, gpu_hours=[r["gpu_hours"] for r in subset])
    groups = result["groups"].values()
    seed_conditions = defaultdict(set)
    for row in records:
        seed_conditions[row['seed']].add((row['mode'],tuple(row['order'])))
    paired = all(len(conditions) == 10 for conditions in seed_conditions.values())
    if paired and all(g["n"] >= 2 for g in groups) and all(g["successes"] >= 2 for k,g in result["groups"].items() if not k.startswith("sequential")):
        result["status"] = "passed"
    sequences=[r for r in records if r["mode"] == "sequential"]
    grouped={}
    for r in sequences:
        grouped.setdefault(r["seed"],{})[tuple(r["order"])]=r["first_global"] or r["cap"]
    result["dispersion_uncertainty"] = dict(status="insufficient",restricted_cv_upper=None)
    if len(grouped) >= 5 and all(set(g) == set(ORDERS) for g in grouped.values()):
        values=np.array([[g[o] for o in ORDERS] for g in grouped.values()],dtype=float)
        if np.all(values.std(0) > 0):
            rng=np.random.default_rng(39001)
            samples=values[rng.integers(0,len(values),(5000,len(values))) ]
            cvs=samples.std(1,ddof=1)/samples.mean(1)
            bound=float(np.quantile(cvs,.975,axis=0).max()*1.25)
            result["dispersion_uncertainty"]=dict(status="estimated",restricted_cv_upper=bound,
                n_seed_blocks=len(values),method="Paired seed bootstrap 97.5% CV bound, inflated by 25%; approximate nuisance envelope, not a power guarantee.")
    result["interpretation"] = "Structural learnability only. Requires a separate uncertainty-aware power and budget decision."
    write_json(output, result)
    return result


def verify_main_launch(launch_path, protocol, graph, dataset, tokenizer, run_id):
    from .data import audit
    launch = json.loads(Path(launch_path).read_text(encoding="utf-8"))
    if launch.get("status") != "frozen" or launch.get("protocol_hash") != protocol.hash:
        raise ValueError("No compatible frozen main-study manifest")
    if launch.get("code_hash") != code_hash() or launch.get("graph_hash") != graph.hash:
        raise ValueError("Code/graph changed after launch freeze")
    report = audit(dataset)
    if not report["passed"] or launch["dataset_hash"] != report["dataset_hash"] or launch["tokenizer_hash"] != file_hash(tokenizer):
        raise ValueError("Dataset/tokenizer changed after launch freeze")
    for path, expected in launch["evidence_files"].items():
        if file_hash(path) != expected:
            raise ValueError("Gate evidence changed after launch freeze")
    runs = [r for r in launch["runs"] if r["run_id"] == run_id]
    if len(runs) != 1:
        raise ValueError("Run is not in the complete prespecified matrix")
    return launch, runs[0]
