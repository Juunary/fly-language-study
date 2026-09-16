"""Recording a Claude review session from its real agent transcript: assistant messages are archived verbatim, a
response that hit the output limit and was continued in a second message is reassembled without touching any
judgment, and the run metadata (model id, timestamps) comes from the transcript rather than from estimates."""
import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("record_ai_review_session", ROOT/"scripts"/"record_ai_review_session.py")
rec = importlib.util.module_from_spec(spec); spec.loader.exec_module(rec)

HEADER = "review_id,language,split,task,sentence_a,sentence_b,reviewer,judged_label,fluent,comment"
CODE = "EN-R1-TEST1"
ROWS = [f"aaaa000001,en,dev_a,space,A one.,B one.,{CODE},1,yes,fine",
        f"bbbb000002,en,dev_b,roles,A two.,B two.,{CODE},0,yes,verbs swapped",
        f"cccc000003,en,test,quantity,A three.,B three.,{CODE},1,no,\"awkward phrase, suggest fix\""]
CHECK_HEADER = "area,language,family,split,task,checked,issue,comment"
CHECKS = ["construction,en,assertion,train,roles,yes,no,fine", "noun_forms,en,,,nom,yes,no,fine"]


def line(kind, **fields):
    return json.dumps(dict(type=kind, **fields), ensure_ascii=False)


def transcript(tmp_path, first_text, second_text, first_stop="max_tokens"):
    rows = [line("attachment", attachment=dict(type="model", identity=dict(modelId="claude-opus-5[1m]", marketingName="Opus 5 (1M context)"))),
            line("user", uuid="u1", timestamp="2026-09-16T22:25:10.803Z", message=dict(role="user", content="LAUNCH TEXT")),
            line("assistant", uuid="a0", timestamp="2026-09-16T22:25:12.000Z", message=dict(role="assistant", stop_reason="tool_use",
                 content=[dict(type="text", text="I will read the prompt."), dict(type="tool_use", name="Read", input=dict(file_path="x"))])),
            line("user", uuid="u2", timestamp="2026-09-16T22:25:13.000Z", message=dict(role="user", content=[dict(type="tool_result", content="...")])),
            line("assistant", uuid="a1", timestamp="2026-09-16T22:36:27.115Z", message=dict(role="assistant", stop_reason=first_stop,
                 usage=dict(output_tokens=64000), content=[dict(type="text", text=first_text)])),
            line("assistant", uuid="a2", timestamp="2026-09-16T22:37:40.692Z", message=dict(role="assistant", stop_reason="end_turn",
                 usage=dict(output_tokens=7553), content=[dict(type="text", text=second_text)]))]
    path = tmp_path/"agent.jsonl"; path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def truncated_pair(partial):
    first = "```csv items-response.csv\n" + HEADER + "\n" + ROWS[0] + "\n" + partial
    second = ("```csv items-response.csv (continued from the truncated row bbbb000002)\n" + ROWS[1] + "\n" + ROWS[2] + "\n```\n\n"
              "```csv checklist-response.csv\n" + CHECK_HEADER + "\n" + "\n".join(CHECKS) + "\n```")
    return first, second


def test_agent_transcript_yields_messages_model_and_timestamps(tmp_path):
    first, second = truncated_pair(ROWS[1][:40])
    session = rec.read_agent_transcript(transcript(tmp_path, first, second))
    assert session["model_id"] == "claude-opus-5[1m]" and session["model_version"] == "Opus 5 (1M context)"
    assert session["launcher"] == "LAUNCH TEXT"
    assert session["started_at_utc"] == "2026-09-16T22:25:10.803Z" and session["finished_at_utc"] == "2026-09-16T22:37:40.692Z"
    assert [m["stop_reason"] for m in session["messages"]] == ["tool_use", "max_tokens", "end_turn"]
    assert [m["text"] for m in session["messages"]] == ["I will read the prompt.", first, second]
    assert session["tool_uses"] == [dict(name="Read", input=dict(file_path="x"))]


def test_continuation_block_is_reassembled_and_partial_row_dropped(tmp_path):
    first, second = truncated_pair(ROWS[1][:40])
    text, info = rec.extract_csv([first, second], "items-response.csv")
    assert text == HEADER + "\n" + "\n".join(ROWS) + "\n"
    assert info == dict(blocks=2, unterminated=1, dropped_partial_rows=[dict(review_id="bbbb000002", text=ROWS[1][:40])], superseded_attempts=[])


def test_single_terminated_block_has_no_assembly(tmp_path):
    whole = "```csv items-response.csv\n" + HEADER + "\n" + "\n".join(ROWS) + "\n```\n"
    text, info = rec.extract_csv([whole], "items-response.csv")
    assert text == HEADER + "\n" + "\n".join(ROWS) + "\n"
    assert info == dict(blocks=1, unterminated=0, dropped_partial_rows=[], superseded_attempts=[])


def test_partial_row_with_a_conflicting_judgment_is_refused():
    conflicting = ROWS[1].replace(f"{CODE},0,yes", f"{CODE},1,yes")  # complete row whose label differs from the continuation
    first, second = truncated_pair(conflicting)
    with pytest.raises(ValueError, match="conflict"):
        rec.extract_csv([first, second], "items-response.csv")


def restart_pair(draft_rows, final_rows):
    first = "```csv items-response.csv\n" + HEADER + "\n" + "\n".join(draft_rows)
    second = ("```csv items-response.csv\n" + HEADER + "\n" + "\n".join(final_rows) + "\n```\n\n"
              "```csv checklist-response.csv\n" + CHECK_HEADER + "\n" + "\n".join(CHECKS) + "\n```")
    return first, second


def test_complete_restart_supersedes_a_truncated_attempt_and_is_compared_to_it():
    draft = [ROWS[0], ROWS[1].replace(f"{CODE},0,yes", f"{CODE},1,yes"), ROWS[2][:30]]  # label differs on row 2; row 3 cut off
    first, second = restart_pair(draft, ROWS)
    text, info = rec.extract_csv([first, second], "items-response.csv", compare=("judged_label", "fluent"))
    assert text == HEADER + "\n" + "\n".join(ROWS) + "\n"
    assert info == dict(blocks=2, unterminated=1, dropped_partial_rows=[],
                        superseded_attempts=[dict(message=0, rows=3, complete_rows=2, disagreements={"judged_label": ["bbbb000002"], "fluent": []})])


def test_restart_that_is_itself_unterminated_is_refused():
    first = "```csv items-response.csv\n" + HEADER + "\n" + ROWS[0]
    second = "```csv items-response.csv\n" + HEADER + "\n" + "\n".join(ROWS)
    with pytest.raises(ValueError, match="unterminated"):
        rec.extract_csv([first, second], "items-response.csv")


def test_unterminated_block_without_continuation_is_refused():
    first = "```csv items-response.csv\n" + HEADER + "\n" + ROWS[0] + "\n" + ROWS[1][:40]
    with pytest.raises(ValueError, match="unterminated"):
        rec.extract_csv([first], "items-response.csv")


def workspace(tmp_path):
    ws = tmp_path/"ws"; folder = ws/CODE; folder.mkdir(parents=True)
    (folder/f"items-{CODE}.csv").write_text("﻿" + HEADER + "\n" + "\n".join(r.rsplit(",", 3)[0] + ",,," for r in ROWS) + "\n", encoding="utf-8")
    (folder/f"checklist-{CODE}.csv").write_text("﻿" + CHECK_HEADER + "\n" + "\n".join(c.rsplit(",", 3)[0] + ",,," for c in CHECKS) + "\n", encoding="utf-8")
    (folder/"prompt.md").write_text("prior prompt", encoding="utf-8"); (folder/"prompt-sent.md").write_text("prompt sent", encoding="utf-8")
    (folder/"launcher-text.md").write_text("LAUNCH TEXT", encoding="utf-8")
    run = dict(reviewer=CODE, language="en", provider="anthropic", model_id="", model_version="", interface="", session_id="",
               executed_at_utc="", settings={}, fresh_context=None, answer_key_exposed=None,
               inputs=dict(items=dict(path=f"{CODE}/items-{CODE}.csv"), checklist=dict(path=f"{CODE}/checklist-{CODE}.csv")),
               prompt=dict(path=f"{CODE}/prompt.md"), prior_artifacts=[])
    (ws/"evidence.json").write_text(json.dumps(dict(schema_version=1, runs=[run])), encoding="utf-8")
    return ws


def test_main_records_session_from_agent_transcript(tmp_path, monkeypatch, capsys):
    ws = workspace(tmp_path)
    first, second = truncated_pair(ROWS[1][:40])
    jsonl = transcript(tmp_path, first, second)
    monkeypatch.setattr(sys, "argv", ["rec", "--workspace", str(ws), "--code", CODE, "--agent-transcript", str(jsonl),
                                      "--launcher", str(ws/CODE/"launcher-text.md"), "--session-id", "review-" + CODE,
                                      "--submissions", str(tmp_path/"inbox")])
    rec.main()
    folder = ws/CODE
    with (folder/"items-response.csv").open(encoding="utf-8-sig", newline="") as f:
        items = list(csv.DictReader(f))
    assert [r["review_id"] for r in items] == ["aaaa000001", "bbbb000002", "cccc000003"]
    assert [r["judged_label"] for r in items] == ["1", "0", "1"] and items[2]["fluent"] == "no"
    assert (tmp_path/"inbox"/CODE/"items-response.csv").read_bytes() == (folder/"items-response.csv").read_bytes()
    assert (folder/"session-transcript.jsonl").read_bytes() == jsonl.read_bytes()
    archived = (folder/"transcript.txt").read_text(encoding="utf-8")
    assert "LAUNCH TEXT" in archived and first in archived and second in archived and "max_tokens" in archived
    run = json.loads((ws/"evidence.json").read_text(encoding="utf-8"))["runs"][0]
    assert run["model_id"] == "claude-opus-5[1m]" and run["model_version"] == "Opus 5 (1M context)"
    assert run["interface"] == "claude_code" and run["executed_at_utc"] == "2026-09-16T22:37:40.692Z"
    assert run["session_started_at_utc"] == "2026-09-16T22:25:10.803Z"
    assert run["fresh_context"] is True and run["answer_key_exposed"] is False and run["session_id"] == "review-" + CODE
    assert run["settings"]["stop_reasons"] == ["tool_use", "max_tokens", "end_turn"]
    assert run["response_assembly"]["items-response.csv"] == dict(blocks=2, unterminated=1, dropped_partial_rows=[dict(review_id="bbbb000002", text=ROWS[1][:40])], superseded_attempts=[])
    assert run["response_assembly"]["checklist-response.csv"] == dict(blocks=1, unterminated=0, dropped_partial_rows=[], superseded_attempts=[])
    assert run["transcript"]["sha256"] and run["items_response"]["sha256"] and run["session_transcript"]["path"] == f"{CODE}/session-transcript.jsonl"
    assert run["prior_artifacts"][0]["path"] == f"{CODE}/prompt.md"
    out = json.loads(capsys.readouterr().out)
    assert out["items"] == 3 and out["fluent_no"] == 1 and out["checklist_rows"] == 2
    with pytest.raises(FileExistsError):
        rec.main()
