"""Derive each Claude review session's prompt and launcher text from the evidence workspace; no judgments involved.

For every run in ``<workspace>/evidence.json`` this writes ``<code>/prompt-sent.md`` (the generic prompt copied by
prepare_ai_review.py plus a package section naming the five blinded input files by absolute path and the required
output format), ``<code>/launcher-text.md`` (the one-paragraph message a fresh session receives) and a
``launch-record.json`` with file sizes and hashes. The texts are fixed here so that every session of a review round
receives the same instructions and the record can be reproduced.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from flystudy.protocol import file_hash

PACKAGE_SECTION = """

## This session's package (reviewer code {code}, language {language})

Read ONLY these five files with your file-reading tool, in this order, and nothing else. Do not run shell
commands, do not search or open any other file or directory, do not use the repository, and do not look for
labels, IDs or other sessions' judgments. The files contain no answer key.

1. items: {items}
2. checklist: {checklist}
3. noun forms: {noun_forms}
4. construction examples: {construction_examples}
5. template inventory: {inventory}

The construction examples have two renderings per held-out split: `primary` (the training-frame declarative
sentence, which is what the study's primary evaluation scores) and `auxiliary` (the outer frame owned by the split:
question, reported clause, conditional). The item sentences you judge are primary-rendering sentences (train and
held-out). Check both renderings' constructions in the checklist.

## Required output

Return, in your final message and nothing else, exactly two fenced code blocks:

```csv items-response.csv
<the complete items CSV: header row exactly as in the input, then all {item_count} rows in the input order, with
review_id, language, split, task, sentence_a, sentence_b, reviewer copied unchanged and judged_label (0 or 1),
fluent (yes or no) and comment filled; quote any field containing a comma, quote or newline with standard CSV quoting>
```

```csv checklist-response.csv
<the complete checklist CSV: header row exactly as in the input, then all {check_count} rows with area, language, family,
split, task copied unchanged and checked (yes/no), issue (yes/no), comment filled>
```

Keep each comment to one short clause (about twelve words at most) so that the complete CSV fits in one message.
If your message is cut off before the blocks are complete, continue in your next message from the row that was cut
off, without repeating the rows already returned.
If you cannot complete all rows, say so explicitly instead of omitting rows.
"""

LAUNCHER = ("You are a fresh, independent AI reviewer session for a language-data assessment. Your complete instructions are "
            "in the file {prompt}. Read that file first with your file-reading tool and follow it exactly. Read only the six "
            "files it names (the prompt file and the five package files); do not run shell commands, do not open any other "
            "file or directory, and do not use any other tool. Your final message must contain exactly the two fenced CSV "
            "blocks the instructions specify and nothing else.")


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        return max(sum(1 for line in f if line.strip()) - 1, 0)


def prepare_sessions(workspace):
    ws = Path(workspace).resolve()
    evidence = json.loads((ws/"evidence.json").read_text(encoding="utf-8"))
    record = {}
    for run in evidence["runs"]:
        code, folder = run["reviewer"], ws/run["reviewer"]
        paths = {kind: (ws/ref["path"]).resolve() for kind, ref in run["inputs"].items()}
        generic = (ws/run["prompt"]["path"]).read_text(encoding="utf-8").rstrip("\n")
        section = PACKAGE_SECTION.format(code=code, language=run["language"], item_count=csv_rows(paths["items"]),
                                         check_count=csv_rows(paths["checklist"]), **{k: str(p) for k, p in paths.items()})
        sent, launcher = folder/"prompt-sent.md", folder/"launcher-text.md"
        sent.write_text(generic + "\n" + section, encoding="utf-8")
        launcher.write_text(LAUNCHER.format(prompt=sent), encoding="utf-8")
        record[code] = dict(language=run["language"], sizes={k: p.stat().st_size for k, p in paths.items()},
                            prompt_sent_sha256=file_hash(sent), launcher_sha256=file_hash(launcher),
                            prepared_at_utc=datetime.now(timezone.utc).isoformat())
    (ws/"launch-record.json").write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workspace", required=True, help="review/<ver>/ai/<name> holding evidence.json")
    a = p.parse_args()
    record = prepare_sessions(a.workspace)
    print(json.dumps({code: r["sizes"] for code, r in record.items()}, indent=1))


if __name__ == "__main__":
    main()
