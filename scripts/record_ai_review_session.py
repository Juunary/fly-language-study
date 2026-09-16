"""Archive one Claude review session's real transcript and raw response as evidence; never generate or alter judgments.

Reads either the session's agent transcript (JSONL as written by Claude Code, ``--agent-transcript``) or a plain file
holding the raw response text (``--response``), stores the raw material verbatim as the transcript, extracts the two
fenced CSV blocks the prompt asked for, validates them against the blinded inputs (identifiers, sentences and row counts
unchanged; judgments present), copies the items CSV into the submissions inbox and fills the run's metadata in
evidence.json from the transcript itself (model id, timestamps, stop reasons). A response that hit the output limit and
was continued in a later message is reassembled: the blocks are concatenated in order and the single partial row cut
off at the limit is dropped only when the continuation supplies the same review_id again; a partial row whose judgment
conflicts with the continuation is refused. Every assembly step is recorded in evidence.json. Any validation failure
leaves the raw material in place and exits non-zero so that a fresh session can be run and both records kept.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

from flystudy.protocol import file_hash

ITEM_FIELDS = ["review_id", "language", "split", "task", "sentence_a", "sentence_b", "reviewer", "judged_label", "fluent", "comment"]
CHECK_FIELDS = ["area", "language", "family", "split", "task", "checked", "issue", "comment"]
YES, NO = ("yes", "true", "1"), ("no", "false", "0")


def read_csv_text(text):
    return list(csv.DictReader(StringIO(text.lstrip("﻿"))))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_agent_transcript(path):
    """Assistant messages, tool uses, launcher text, model identity and timestamps from a Claude Code agent JSONL."""
    session = dict(model_id="", model_version="", launcher="", started_at_utc="", finished_at_utc="", messages=[], tool_uses=[])
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        d = json.loads(raw)
        if d.get("type") == "attachment" and d.get("attachment", {}).get("type") == "model":
            identity = d["attachment"].get("identity", {})
            session["model_id"], session["model_version"] = identity.get("modelId", ""), identity.get("marketingName", "")
        elif d.get("type") == "user" and isinstance(d.get("message", {}).get("content"), str) and not session["launcher"]:
            session["launcher"], session["started_at_utc"] = d["message"]["content"], d.get("timestamp", "")
        elif d.get("type") == "assistant":
            message = d.get("message", {})
            for block in message.get("content", []):
                if block.get("type") == "text":
                    session["messages"].append(dict(uuid=d.get("uuid"), timestamp=d.get("timestamp", ""), stop_reason=message.get("stop_reason"),
                                                    output_tokens=(message.get("usage") or {}).get("output_tokens"), text=block["text"]))
                elif block.get("type") == "tool_use":
                    session["tool_uses"].append(dict(name=block.get("name"), input=block.get("input")))
            if d.get("timestamp"):
                session["finished_at_utc"] = d["timestamp"]
    if not session["messages"]:
        raise ValueError("agent transcript holds no assistant text")
    return session


def render_transcript(launcher, session):
    parts = ["=== launcher text sent to the session ===", launcher, "",
             "=== prompt file the session was told to read: prompt-sent.md ===", ""]
    if session.get("model_id"):
        parts += [f"=== model identity exposed to the session: {session['model_id']} ({session['model_version']}) ===", ""]
    for name, tool in [(t["name"], t) for t in session.get("tool_uses", [])]:
        parts.append(f"=== tool use: {name} {json.dumps(tool['input'], ensure_ascii=False)} ===")
    if session.get("tool_uses"):
        parts.append("")
    for k, m in enumerate(session["messages"], 1):
        parts += [f"=== assistant message {k}/{len(session['messages'])} (uuid {m.get('uuid')}, timestamp {m.get('timestamp')}, "
                  f"stop_reason {m.get('stop_reason')}, output_tokens {m.get('output_tokens')}) ===", m["text"], ""]
    return "\n".join(parts)


OPEN = re.compile(r"^```csv[ \t]+(?P<name>\S+)(?:[ \t].*)?$")
CLOSE = re.compile(r"^```[ \t]*$")


def fenced_blocks(message):
    """(name, body_lines, terminated) for each fenced csv block of one message, in order."""
    blocks, current = [], None
    for line in message.splitlines():
        if current is None:
            m = OPEN.match(line)
            if m:
                current = [m.group("name"), []]
        elif CLOSE.match(line):
            blocks.append((current[0], current[1], True)); current = None
        else:
            current[1].append(line)
    if current is not None:
        blocks.append((current[0], current[1], False))
    return blocks


def parse_rows(lines):
    """DictReader over assembled lines; rows whose field count differs from the header are partial (cut off) rows."""
    header = next(csv.reader([lines[0]]))
    rows, partial = [], []
    for line in lines[1:]:
        fields = next(csv.reader([line]), [])
        (rows if len(fields) == len(header) else partial).append(dict(zip(header, fields)) if len(fields) == len(header) else line)
    return header, rows, partial


def compare_attempt(attempt_lines, final_lines, compare):
    """How a superseded (cut-off) attempt differs from the final block on the compared fields, keyed by the first column."""
    header, rows, partial = parse_rows(attempt_lines)
    _, final_rows, _ = parse_rows(final_lines)
    key = header[0]
    final_by_key = {r[key]: r for r in final_rows}
    disagreements = {field: [r[key] for r in rows if r[key] in final_by_key and r[field].strip() != final_by_key[r[key]][field].strip()]
                     for field in compare}
    return dict(rows=len(rows) + len(partial), complete_rows=len(rows), disagreements=disagreements)


def extract_csv(messages, name, compare=()):
    """Assemble the fenced blocks named ``name`` across the session's messages, in order, and report the assembly.

    A block that was cut off at the output limit is either continued (the next block starts after the cut, without a
    header) or restarted (the next block starts again with the header): a continuation is concatenated after dropping
    the single partial row that the continuation re-supplies; a complete restart supersedes the cut-off attempt, which
    is kept in the report together with the fields on which its complete rows disagree with the final block.
    """
    found = [(k, body, terminated) for k, message in enumerate(messages) for n, body, terminated in fenced_blocks(message) if n == name]
    if not found:
        raise ValueError(f"No fenced block named {name}")
    info = dict(blocks=len(found), unterminated=sum(not t for _, _, t in found), dropped_partial_rows=[], superseded_attempts=[])
    attempts, lines, terminated, started_in = [], [], False, None
    for message_index, body, block_terminated in found:
        body = list(body)
        header = lines[0] if lines else None
        if lines and body and body[0] == header:  # a restart: the earlier attempt is superseded
            attempts.append((started_in, lines)); lines = []
        elif lines and not terminated:  # a continuation of a cut-off block
            partial = lines[-1]
            partial_id = partial.split(",", 1)[0]
            replacement = next((l for l in body if l.split(",", 1)[0] == partial_id), None)
            if replacement is None:
                raise ValueError(f"unterminated {name} block: partial row {partial_id!r} is not re-supplied by the continuation")
            if partial != replacement[:len(partial)]:
                raise ValueError(f"partial row {partial_id!r} conflicts with its continuation row")
            info["dropped_partial_rows"].append(dict(review_id=partial_id, text=partial))
            lines = lines[:-1]
        elif lines and terminated:
            raise ValueError(f"a second {name} block follows a complete one without restarting from the header")
        if not lines:
            started_in = message_index
        lines += [l for l in body if l.strip() != ""] if lines else body
        terminated = block_terminated
    if not terminated:
        raise ValueError(f"unterminated {name} block without a continuation")
    for message_index, attempt in attempts:
        info["superseded_attempts"].append(dict(message=message_index, **compare_attempt(attempt, lines, compare)))
    return "\n".join(lines).strip("\n") + "\n", info


def write_csv(path, fields, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        w.writeheader(); w.writerows(rows)


def validate_items(response_rows, input_rows, code):
    if [k for k in response_rows[0].keys()] != ITEM_FIELDS:
        raise ValueError(f"items header differs: {list(response_rows[0].keys())}")
    if len(response_rows) != len(input_rows):
        raise ValueError(f"items rows {len(response_rows)} != {len(input_rows)}")
    by_id = {r["review_id"]: r for r in input_rows}
    seen = set()
    for r in response_rows:
        src = by_id.get(r["review_id"])
        if src is None or r["review_id"] in seen:
            raise ValueError(f"unknown or duplicate review_id {r['review_id']!r}")
        seen.add(r["review_id"])
        for k in ("language", "split", "task", "sentence_a", "sentence_b"):
            if r[k].strip() != src[k].strip():
                raise ValueError(f"{k} changed for {r['review_id']}")
        if r["reviewer"].strip() != code:
            raise ValueError(f"reviewer code changed for {r['review_id']}")
        if r["judged_label"].strip() not in ("0", "1"):
            raise ValueError(f"judged_label missing for {r['review_id']}")
        if r["fluent"].strip().lower() not in YES + NO:
            raise ValueError(f"fluent missing for {r['review_id']}")
        if r["fluent"].strip().lower() in NO and not r["comment"].strip():
            raise ValueError(f"fluent=no without comment for {r['review_id']}")


def validate_checklist(response_rows, input_rows):
    if [k for k in response_rows[0].keys()] != CHECK_FIELDS:
        raise ValueError(f"checklist header differs: {list(response_rows[0].keys())}")
    key = lambda r: tuple(r[k].strip() for k in ("area", "language", "family", "split", "task"))
    if [key(r) for r in response_rows] != [key(r) for r in input_rows]:
        raise ValueError("checklist rows differ from the input rows")
    for r in response_rows:
        if r["checked"].strip().lower() not in YES + NO or r["issue"].strip().lower() not in YES + NO:
            raise ValueError(f"checklist row {key(r)} lacks checked/issue")
        if r["issue"].strip().lower() in YES and not r["comment"].strip():
            raise ValueError(f"checklist issue without comment: {key(r)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workspace", required=True); p.add_argument("--code", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--agent-transcript", help="Claude Code agent transcript (JSONL) of the session; copied verbatim")
    src.add_argument("--response", help="file holding the raw response text exactly as returned")
    p.add_argument("--launcher", required=True, help="file holding the exact launcher text sent to the session")
    p.add_argument("--session-id", required=True); p.add_argument("--executed-at", help="UTC ISO timestamp (taken from the transcript when given)")
    p.add_argument("--model-id", help="required with --response; must match the transcript when given"); p.add_argument("--model-version")
    p.add_argument("--interface", default="claude_code"); p.add_argument("--settings", help="JSON object of extra settings to record")
    p.add_argument("--submissions", required=True, help="review/<ver>/submissions/raw")
    a = p.parse_args()
    ws, folder = Path(a.workspace), Path(a.workspace)/a.code
    evidence = json.loads((ws/"evidence.json").read_text(encoding="utf-8"))
    run = next(r for r in evidence["runs"] if r["reviewer"] == a.code)
    launcher = Path(a.launcher).read_text(encoding="utf-8")
    settings = json.loads(a.settings) if a.settings else {}
    if a.agent_transcript:
        session = read_agent_transcript(a.agent_transcript)
        if session["launcher"].strip() != launcher.strip():
            raise ValueError("launcher text on disk differs from the first user message of the transcript")
        if a.model_id and a.model_id != session["model_id"]:
            raise ValueError(f"--model-id {a.model_id!r} differs from the transcript's {session['model_id']!r}")
        model_id, model_version = session["model_id"] or a.model_id or "", session["model_version"] or a.model_version or "not_exposed"
        executed_at, started_at = session["finished_at_utc"], session["started_at_utc"]
        settings.update(stop_reasons=[m["stop_reason"] for m in session["messages"]], output_tokens=[m["output_tokens"] for m in session["messages"]],
                        tool_uses=[t["name"] for t in session["tool_uses"]], sampling="not_exposed")
        session_copy = folder/"session-transcript.jsonl"
        shutil.copyfile(a.agent_transcript, session_copy)
    else:
        if not (a.model_id and a.executed_at):
            raise ValueError("--response needs --model-id and --executed-at")
        session = dict(messages=[dict(uuid=None, timestamp=a.executed_at, stop_reason=None, output_tokens=None,
                                      text=Path(a.response).read_text(encoding="utf-8"))], tool_uses=[], model_id=a.model_id, model_version=a.model_version or "not_exposed")
        model_id, model_version, executed_at, started_at, session_copy = a.model_id, session["model_version"], a.executed_at, "", None
        settings.setdefault("availability", "not_exposed")
    transcript = folder/"transcript.txt"
    transcript.write_text(render_transcript(launcher, session), encoding="utf-8")
    texts = [m["text"] for m in session["messages"]]
    items_text, items_info = extract_csv(texts, "items-response.csv", compare=("judged_label", "fluent"))
    checks_text, checks_info = extract_csv(texts, "checklist-response.csv", compare=("checked", "issue"))
    items, checks = read_csv_text(items_text), read_csv_text(checks_text)
    validate_items(items, read_csv(ws/run["inputs"]["items"]["path"]), a.code)
    validate_checklist(checks, read_csv(ws/run["inputs"]["checklist"]["path"]))
    write_csv(folder/"items-response.csv", ITEM_FIELDS, items)
    write_csv(folder/"checklist-response.csv", CHECK_FIELDS, checks)
    inbox = Path(a.submissions)/a.code
    inbox.mkdir(parents=True, exist_ok=True)
    for old in inbox.glob("*.csv"):
        raise FileExistsError(f"Inbox already holds {old}; keep the first submission and record a new session separately")
    shutil.copyfile(folder/"items-response.csv", inbox/"items-response.csv")
    datetime.fromisoformat(executed_at.replace("Z", "+00:00"))
    rel = lambda path: str(Path(path).resolve().relative_to(ws.resolve()))
    run.update(model_id=model_id, model_version=model_version, interface=a.interface, session_id=a.session_id,
               executed_at_utc=executed_at, session_started_at_utc=started_at, settings=settings, fresh_context=True, answer_key_exposed=False,
               prompt=dict(path=rel(folder/"prompt-sent.md"), sha256=file_hash(folder/"prompt-sent.md")),
               transcript=dict(path=rel(transcript), sha256=file_hash(transcript)),
               items_response=dict(path=rel(folder/"items-response.csv"), sha256=file_hash(folder/"items-response.csv")),
               checklist_response=dict(path=rel(folder/"checklist-response.csv"), sha256=file_hash(folder/"checklist-response.csv")),
               response_assembly={"items-response.csv": items_info, "checklist-response.csv": checks_info},
               prior_artifacts=[dict(path=rel(folder/"prompt.md"), sha256=file_hash(folder/"prompt.md"))] + run.get("prior_artifacts", []))
    if session_copy is not None:
        run["session_transcript"] = dict(path=rel(session_copy), sha256=file_hash(session_copy))
    elif Path(a.response).resolve().is_relative_to(ws.resolve()):
        run["response_raw"] = dict(path=rel(Path(a.response)), sha256=file_hash(a.response))
    (ws/"evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fluent_no = sum(r["fluent"].strip().lower() in NO for r in items)
    issues = sum(r["issue"].strip().lower() in YES for r in checks)
    print(json.dumps(dict(code=a.code, items=len(items), fluent_no=fluent_no, checklist_rows=len(checks), checklist_issues=issues,
                          label_1=sum(r["judged_label"].strip() == "1" for r in items)), indent=1))


if __name__ == "__main__":
    main()
