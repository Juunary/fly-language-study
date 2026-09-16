"""Adjudication of unresolved Claude-review items (v4-ai-review-1): a blinded package (no original label) and prompt are
derived from the merge's disagreement list, and a fresh session's JSON verdicts are recorded verbatim into evidence.json
and copied into the attestation. The tool never writes a verdict itself."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("adjudicate_ai_review", ROOT/"scripts"/"adjudicate_ai_review.py")
adj = importlib.util.module_from_spec(spec); spec.loader.exec_module(adj)

DISAGREEMENT_FIELDS = ["id", "language", "split", "task", "sentence_a", "sentence_b", "original_label",
                       "reviewer_1", "judged_label_1", "fluent_1", "comment_1", "reviewer_2", "judged_label_2", "fluent_2", "comment_2", "reason"]


def fixture(tmp_path):
    merged = tmp_path/"merged"; merged.mkdir()
    rows = [dict(id="item1", language="de", split="dev_a", task="space", sentence_a="A eins.", sentence_b="B eins.", original_label="1",
                 reviewer_1="DE-R1-X", judged_label_1="1", fluent_1="no", comment_1="stilted phrase", reviewer_2="DE-R2-Y", judged_label_2="1", fluent_2="yes", comment_2="fine", reason="fluent_no"),
            dict(id="item2", language="ko", split="test", task="roles", sentence_a="가.", sentence_b="나.", original_label="0",
                 reviewer_1="KO-R1-X", judged_label_1="0", fluent_1="yes", comment_1="ok", reviewer_2="KO-R2-Y", judged_label_2="1", fluent_2="yes", comment_2="hmm", reason="reviewers_differ;differs_from_original")]
    with (merged/"disagreements.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DISAGREEMENT_FIELDS); w.writeheader(); w.writerows(rows)
    (merged/"attestation-draft.json").write_text(json.dumps(dict(review_mode="claude_only", reviewers={}, resolved_items={"item1": {}, "item2": {}})), encoding="utf-8")
    ws = tmp_path/"ai"; ws.mkdir()
    (ws/"evidence.json").write_text(json.dumps(dict(schema_version=1, runs=[])), encoding="utf-8")
    return merged, ws


def transcript(tmp_path, text, model="claude-opus-5[1m]"):
    lines = [json.dumps(dict(type="attachment", attachment=dict(type="model", identity=dict(modelId=model, marketingName="Opus 5 (1M context)")))),
             json.dumps(dict(type="user", uuid="u", timestamp="2026-09-17T01:00:00.000Z", message=dict(role="user", content="LAUNCH"))),
             json.dumps(dict(type="assistant", uuid="a", timestamp="2026-09-17T01:05:00.000Z", message=dict(role="assistant", stop_reason="end_turn", usage=dict(output_tokens=500), content=[dict(type="text", text=text)])))]
    p = tmp_path/"agent.jsonl"; p.write_text("\n".join(lines) + "\n", encoding="utf-8"); return p


def test_prepare_builds_a_blinded_package_prompt_and_evidence_skeleton(tmp_path):
    merged, ws = fixture(tmp_path)
    info = adj.prepare(merged, ws)
    folder = ws/"adjudication"
    with (folder/"items-adjudication.csv").open(encoding="utf-8-sig", newline="") as f:
        items = list(csv.DictReader(f))
    assert [r["id"] for r in items] == ["item1", "item2"] and "original_label" not in items[0] and items[0]["comment_1"] == "stilted phrase"
    prompt = (folder/"prompt-sent.md").read_text(encoding="utf-8")
    assert str((folder/"items-adjudication.csv").resolve()) in prompt and "```json adjudication-response.json" in prompt and "resolved_items" in prompt
    launcher = (folder/"launcher-text.md").read_text(encoding="utf-8")
    assert str((folder/"prompt-sent.md").resolve()) in launcher and "\n" not in launcher.strip()
    e = json.loads((ws/"evidence.json").read_text(encoding="utf-8"))["adjudication"]
    assert e["provider"] == "anthropic" and e["model_id"] == "" and e["inputs"]["items"]["sha256"] and e["prompt"]["path"] == "adjudication/prompt-sent.md"
    assert e["response"]["path"] == "adjudication/response.json" and e["transcript"]["path"] == "adjudication/transcript.txt"
    assert info == dict(items=2, languages={"de": 1, "ko": 1})
    (ws/"adjudication"/"launcher-text.md").write_text("LAUNCH", encoding="utf-8")


def good_response():
    return ('```json adjudication-response.json\n' + json.dumps(dict(resolved_items={
        "item1": dict(label="1", fluent=True, rationale="Converse relation preserved; phrasing acceptable."),
        "item2": dict(label="0", fluent=True, rationale="Verbs are exchanged, so roles differ.")}), ensure_ascii=False, indent=1) + '\n```')


def test_record_stores_the_verdicts_verbatim_and_copies_them_into_the_attestation(tmp_path, capsys):
    merged, ws = fixture(tmp_path); adj.prepare(merged, ws)
    (ws/"adjudication"/"launcher-text.md").write_text("LAUNCH", encoding="utf-8")
    jsonl = transcript(tmp_path, good_response())
    adj.record(ws, jsonl, ws/"adjudication"/"launcher-text.md", "agent-adj-1", merged/"attestation-draft.json")
    folder = ws/"adjudication"
    response = json.loads((folder/"response.json").read_text(encoding="utf-8"))
    assert set(response["resolved_items"]) == {"item1", "item2"} and response["resolved_items"]["item1"]["fluent"] is True
    attest = json.loads((merged/"attestation-draft.json").read_text(encoding="utf-8"))
    assert attest["resolved_items"] == response["resolved_items"]
    e = json.loads((ws/"evidence.json").read_text(encoding="utf-8"))["adjudication"]
    assert e["model_id"] == "claude-opus-5[1m]" and e["session_id"] == "agent-adj-1" and e["executed_at_utc"] == "2026-09-17T01:05:00.000Z"
    assert e["fresh_context"] is True and e["answer_key_exposed"] is False and e["interface"] == "claude_code"
    assert e["response"]["sha256"] and e["transcript"]["sha256"] and (folder/"session-transcript.jsonl").exists()
    assert "LAUNCH" in (folder/"transcript.txt").read_text(encoding="utf-8")
    out = json.loads(capsys.readouterr().out)
    assert out["items"] == 2 and out["label_matches_original"] == 2 and out["fluent_true"] == 2
    with pytest.raises(FileExistsError):
        adj.record(ws, jsonl, ws/"adjudication"/"launcher-text.md", "agent-adj-2", merged/"attestation-draft.json")


@pytest.mark.parametrize("bad", [
    '```json adjudication-response.json\n{"resolved_items": {"item1": {"label": "1", "fluent": true, "rationale": "x"}}}\n```',  # missing item2
    '```json adjudication-response.json\n{"resolved_items": {"item1": {"label": 1, "fluent": true, "rationale": "x"}, "item2": {"label": "0", "fluent": true, "rationale": "y"}}}\n```',  # label not a string
    '```json adjudication-response.json\n{"resolved_items": {"item1": {"label": "1", "fluent": "yes", "rationale": "x"}, "item2": {"label": "0", "fluent": true, "rationale": "y"}}}\n```',  # fluent not boolean
    '```json adjudication-response.json\n{"resolved_items": {"item1": {"label": "1", "fluent": true, "rationale": ""}, "item2": {"label": "0", "fluent": true, "rationale": "y"}}}\n```',  # empty rationale
])
def test_invalid_verdicts_are_refused_without_writing(tmp_path, bad):
    merged, ws = fixture(tmp_path); adj.prepare(merged, ws)
    (ws/"adjudication"/"launcher-text.md").write_text("LAUNCH", encoding="utf-8")
    with pytest.raises(ValueError):
        adj.record(ws, transcript(tmp_path, bad), ws/"adjudication"/"launcher-text.md", "agent-adj-1", merged/"attestation-draft.json")
    assert not (ws/"adjudication"/"response.json").exists()
    assert json.loads((merged/"attestation-draft.json").read_text(encoding="utf-8"))["resolved_items"] == {"item1": {}, "item2": {}}
