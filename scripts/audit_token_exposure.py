"""Read-only token exposure audit. Re-render dev scenes in the training frame, without training a model."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

from tokenizers import Tokenizer

from flystudy.data import encode_pair, load_rows, render
from flystudy.protocol import LANGUAGES, file_hash, write_json


def assertion_view(row):
    """Preserve held-out scenes, IDs and labels; change only the outer rendering frame."""
    return {**row, "sentence_a": render(row["scene"], row["language"], "train"),
            "sentence_b": render(row["scene"], row["language"], "train", inverse=True, foil=not row["label"])}


def audit_exposure(data, tokenizer_path):
    data = Path(data)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    training = {language: Counter() for language in LANGUAGES}
    for row in load_rows(data, "train"):
        training[row["language"]].update(encode_pair(tokenizer, row))
    pooled = set().union(*(set(counter) for counter in training.values()))
    result = dict(kind="exploratory_token_exposure_diagnostic", model_evaluation_performed=False,
                  gpu_used=False, tokenizer_hash=file_hash(tokenizer_path),
                  source_hashes={f"{split}.jsonl": file_hash(data / f"{split}.jsonl")
                                 for split in ("train", "dev_a", "dev_b")}, cells=[])
    for split in ("dev_a", "dev_b"):
        counts, sizes = defaultdict(Counter), Counter()
        for row in load_rows(data, split):
            for view, item in (("original", row), ("assertion", assertion_view(row))):
                key = (row["language"], view)
                counts[key].update(encode_pair(tokenizer, item))
                sizes[key] += 1
        for (language, view), counter in sorted(counts.items()):
            total = sum(counter.values())
            result["cells"].append(dict(split=split, language=language, view=view, items=sizes[(language, view)],
                tokens=total, mean_tokens=total/sizes[(language, view)],
                unseen_in_language_count=sum(n for token, n in counter.items() if token not in training[language]),
                unseen_in_any_training_count=sum(n for token, n in counter.items() if token not in pooled),
                unseen_in_any_training_fraction=sum(n for token, n in counter.items() if token not in pooled)/total))
    result["interpretation"] = (
        "Token-occurrence fractions, including SEP/EOS; not vocabulary-type fractions or UNK rates. "
        "Original evaluation combines frame shift, length shift, and token IDs absent from all training inputs. "
        "Assertion views retain held-out semantic scenes but relax outer-frame holdout. This is a diagnostic, "
        "not a replacement confirmatory evaluation or evidence that this shift caused all performance failures.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/draft-v4.3")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer-v4.3.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError("Keep previous diagnostics; use a new output path")
    report = audit_exposure(args.data, args.tokenizer)
    write_json(args.output, report)
    for cell in report["cells"]:
        print(f"{cell['split']}/{cell['language']}/{cell['view']}: "
              f"mean_length={cell['mean_tokens']:.2f}, unseen_token_occurrences={cell['unseen_in_any_training_fraction']:.2%}")
