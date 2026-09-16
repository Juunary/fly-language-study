"""Per-package session prompts for the Claude-only review are derived deterministically from the evidence workspace:
the generic prompt plus a package section naming the five blinded files by absolute path, the short launcher text a
fresh session receives, and a launch record with file sizes. Nothing here generates or alters judgments."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_claude_sessions", ROOT/"scripts"/"prepare_claude_sessions.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)


def workspace(tmp_path):
    ws = tmp_path/"ai"; runs = []
    for code, lang in (("EN-R1-AAAAA", "en"), ("KO-R2-BBBBB", "ko")):
        folder = ws/code; folder.mkdir(parents=True)
        names = dict(items=f"items-{code}.csv", checklist=f"template-checklist-{code}.csv", noun_forms=f"noun-forms-{lang}.csv",
                     construction_examples=f"construction-examples-{lang}.csv", inventory="template-inventory.json")
        inputs = {}
        for kind, name in names.items():
            (folder/name).write_text(f"{kind} content\n" * (3 if kind == "items" else 1), encoding="utf-8")
            inputs[kind] = dict(path=f"{code}/{name}", sha256="x")
        (folder/"prompt.md").write_text("# Generic prompt\n\nJudge the items.\n", encoding="utf-8")
        runs.append(dict(reviewer=code, language=lang, inputs=inputs, prompt=dict(path=f"{code}/prompt.md", sha256="y")))
    (ws/"evidence.json").write_text(json.dumps(dict(runs=runs)), encoding="utf-8")
    return ws


def test_prompts_and_launchers_name_the_package_files_and_required_output(tmp_path):
    ws = workspace(tmp_path)
    record = mod.prepare_sessions(ws)
    for code, lang in (("EN-R1-AAAAA", "en"), ("KO-R2-BBBBB", "ko")):
        sent = (ws/code/"prompt-sent.md").read_text(encoding="utf-8")
        assert sent.startswith("# Generic prompt") and f"## This session's package (reviewer code {code}, language {lang})" in sent
        for name in (f"items-{code}.csv", f"template-checklist-{code}.csv", f"noun-forms-{lang}.csv", f"construction-examples-{lang}.csv", "template-inventory.json"):
            assert str((ws/code/name).resolve()) in sent
        assert "```csv items-response.csv" in sent and "```csv checklist-response.csv" in sent
        assert "all 2 rows in the input order" in sent and "all 0 rows with area" in sent  # counts come from the package files
        assert "If you cannot complete all rows, say so explicitly instead of omitting rows." in sent
        launcher = (ws/code/"launcher-text.md").read_text(encoding="utf-8")
        assert str((ws/code/"prompt-sent.md").resolve()) in launcher and "fresh, independent AI reviewer session" in launcher
        assert "exactly the two fenced CSV blocks" in launcher and "\n" not in launcher.strip()
        assert record[code]["language"] == lang and record[code]["sizes"]["items"] == len("items content\n") * 3
        assert record[code]["prompt_sent_sha256"] and record[code]["launcher_sha256"] and record[code]["prepared_at_utc"].endswith("+00:00")
    saved = json.loads((ws/"launch-record.json").read_text(encoding="utf-8"))
    assert saved == record
    again = mod.prepare_sessions(ws)  # deterministic apart from the timestamp
    assert {c: r["prompt_sent_sha256"] for c, r in again.items()} == {c: r["prompt_sent_sha256"] for c, r in record.items()}
