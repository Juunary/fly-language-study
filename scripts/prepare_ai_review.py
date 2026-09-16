"""Prepare a Claude evidence workspace, or hash completed response artifacts; never generate judgments."""
import argparse
import json
from pathlib import Path
import shutil

from flystudy.ai_review import SOURCES, rows
from flystudy.protocol import file_hash, write_json


def prepare(data, dist, output):
    data, dist, output = Path(data), Path(dist), Path(output)
    if output.exists():
        raise FileExistsError("Use a new AI evidence directory; do not overwrite existing review records")
    files = sorted(dist.glob("*/items-*.csv"))
    if len(files) != 6:
        raise ValueError("Expected the six existing blinded packages")
    runs = []
    for items in files:
        code = items.parent.name
        language = rows(items)[0]["language"]
        folder = output / code
        folder.mkdir(parents=True)
        inputs = {}
        for kind, name in dict(items=items.name, checklist=f"template-checklist-{code}.csv",
                               noun_forms=f"noun-forms-{language}.csv",
                               construction_examples=f"construction-examples-{language}.csv",
                               inventory="template-inventory.json").items():
            target = folder / name
            shutil.copyfile(items.parent / name, target)
            inputs[kind] = dict(path=str(target.relative_to(output)), sha256=file_hash(target))
        prompt = folder / "prompt.md"
        shutil.copyfile(Path(__file__).resolve().parents[1] / "docs" / "CLAUDE_REVIEW_PROMPT.md", prompt)
        def pending(name):
            return dict(path=str((folder / name).relative_to(output)), sha256="")
        runs.append(dict(reviewer=code, language=language, provider="anthropic", model_id="", model_version="",
                         interface="", session_id="", executed_at_utc="", settings={}, fresh_context=None,
                         answer_key_exposed=None, inputs=inputs,
                         prompt=dict(path=str(prompt.relative_to(output)), sha256=file_hash(prompt)),
                         transcript=pending("transcript.txt"), items_response=pending("items-response.csv"),
                         checklist_response=pending("checklist-response.csv"), prior_artifacts=[]))
    manifest = dict(schema_version=1, review_mode="claude_only", data_origin="pending",
                    dataset_hash=json.loads((data / "manifest.json").read_text(encoding="utf-8"))["dataset_hash"],
                    source_hashes={name: file_hash(data / name) for name in SOURCES}, runs=runs)
    write_json(output / "evidence.json", manifest)
    return output / "evidence.json"


def seal(path):
    """Create a separate snapshot. Existing input hashes must match; output hashes are computed once."""
    path = Path(path)
    target = path.with_name(path.stem + ".sealed.json")
    if target.exists():
        raise FileExistsError("Sealed evidence is immutable; use a new versioned record")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("data_origin") != "actual_claude_responses":
        raise ValueError("Confirm actual Claude provenance before sealing")
    for run in manifest["runs"] + ([manifest["adjudication"]] if manifest.get("adjudication") else []):
        refs = list(run.get("inputs", {}).values()) + [run["prompt"], run["transcript"]]
        refs += [run[k] for k in ("items_response", "checklist_response", "response") if k in run]
        refs += run.get("prior_artifacts", [])
        for ref in refs:
            source = path.parent / ref["path"]
            if not source.is_file() or source.stat().st_size == 0:
                raise ValueError(f"Missing artifact: {source}")
            current = file_hash(source)
            if ref.get("sha256") and ref["sha256"] != current:
                raise ValueError(f"Changed artifact: {source}")
            ref["sha256"] = current
    write_json(target, manifest)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/draft-v4.3")
    parser.add_argument("--dist", default="review/dist")
    parser.add_argument("--output", default="review/ai/claude-v1")
    parser.add_argument("--seal", help="Write a sealed copy of a completed evidence.json")
    args = parser.parse_args()
    print(seal(args.seal) if args.seal else prepare(args.data, args.dist, args.output))
