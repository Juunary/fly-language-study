"""Draft controlled-language corpus. Generation is NOT human validation.

Split ownership is assigned to a semantic orbit before rendering: exchanging
entities, counts, polarity or relation direction cannot move a foil to another split.
Outer construction families are held out; core lexical/grammatical rules are shared.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import csv
import json
import random
import re
import unicodedata

from .protocol import LANGUAGES, TASKS, digest, file_hash, write_json

# en singular/plural, de nominative/accusative/dative/plural, gender, ko
NOUNS = [
    ("dog", "dogs", "Hund", "Hund", "Hund", "Hunde", "m", "개"),
    ("fox", "foxes", "Fuchs", "Fuchs", "Fuchs", "Füchse", "m", "여우"),
    ("bear", "bears", "Bär", "Bären", "Bären", "Bären", "m", "곰"),
    ("lion", "lions", "Löwe", "Löwen", "Löwen", "Löwen", "m", "사자"),
    ("cat", "cats", "Katze", "Katze", "Katze", "Katzen", "f", "고양이"),
    ("mouse", "mice", "Maus", "Maus", "Maus", "Mäuse", "f", "쥐"),
    ("cow", "cows", "Kuh", "Kuh", "Kuh", "Kühe", "f", "소"),
    ("duck", "ducks", "Ente", "Ente", "Ente", "Enten", "f", "오리"),
    ("horse", "horses", "Pferd", "Pferd", "Pferd", "Pferde", "n", "말"),
    ("sheep", "sheep", "Schaf", "Schaf", "Schaf", "Schafe", "n", "양"),
    ("hen", "hens", "Huhn", "Huhn", "Huhn", "Hühner", "n", "닭"),
    ("rabbit", "rabbits", "Kaninchen", "Kaninchen", "Kaninchen", "Kaninchen", "n", "토끼"),
]
COLORS = [("red", "rot", "빨간"), ("blue", "blau", "파란"), ("green", "grün", "초록색"),
          ("yellow", "gelb", "노란"), ("black", "schwarz", "검은"), ("white", "weiß", "하얀")]
SIZES = [("small", "klein", "작은"), ("large", "groß", "큰"),
         ("young", "jung", "어린"), ("old", "alt", "늙은")]
VERBS = [("sees", "see", "seen", "sieht", "gesehen", "본다"),
         ("greets", "greet", "greeted", "grüßt", "gegrüßt", "맞이한다"),
         ("follows", "follow", "followed", "verfolgt", "verfolgt", "뒤따른다")]
NUMBERS = {"en": ("two", "three", "four", "five", "six", "seven", "eight"),
           "de": ("zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht"),
           "ko": ("두", "세", "네", "다섯", "여섯", "일곱", "여덟")}
SPLITS = ("train", "dev_a", "dev_b", "test")
EVALUATION_SPLITS = SPLITS[1:]
# Auxiliary (outer-frame) family owned by each split. The primary rendering of every
# evaluation item uses the training frame by design; only the auxiliary families are held out.
FAMILIES = dict(zip(SPLITS, ("assertion", "truth_question", "reported_clause", "conditional")))
PRIMARY_FAMILY = FAMILIES["train"]


def coordinate_ko(sentence):
    for end, replacement in (("않는다", "않으며"), ("맞이한다", "맞이하며"),
                             ("뒤따른다", "뒤따르며"), ("본다", "보며"), ("있다", "있으며")):
        if sentence.endswith(end):
            return sentence[:-len(end)] + replacement
    raise ValueError("Missing Korean coordination rule")


def join_clauses(left, right, lang):
    if lang == "ko":
        # Nominalize BOTH finite clauses identically. Asymmetric connective
        # inflection would reveal which clause carries negation from unigram overlap.
        return tuple(f"{left[i]}는 것과 {right[i]}는 것이 모두 사실이다" for i in (0, 1))
    join = " and " if lang == "en" else " und "
    return tuple(left[i] + join + right[i] for i in (0, 1))


def particle(word, pair):
    final = ord(word[-1])
    coda = 0xAC00 <= final <= 0xD7A3 and (final - 0xAC00) % 28 != 0
    return word + pair[0 if coda else 1]


def noun_phrase(entity, lang, case="nom", count=None):
    n, color, size = entity
    noun, col, adj = NOUNS[n], COLORS[color], SIZES[size]
    index = LANGUAGES.index(lang)
    if lang == "en":
        return f"{'the' if count is None else NUMBERS[lang][count]} {adj[0]} {col[0]} {noun[0 if count is None else 1]}"
    if lang == "ko":
        base = f"{adj[2]} {col[2]} {noun[7]}"
        return base if count is None else f"{base} {NUMBERS[lang][count]} 마리"
    if count is not None:
        return f"{NUMBERS[lang][count]} {adj[1]}e {col[1]}e {noun[5]}"
    gender = noun[6]
    article = {"nom": dict(m="der", f="die", n="das"),
               "acc": dict(m="den", f="die", n="das"),
               "dat": dict(m="dem", f="der", n="dem")}[case][gender]
    ending = "en" if case == "dat" or (case == "acc" and gender == "m") else "e"
    form = noun[{"nom": 2, "acc": 3, "dat": 4}[case]]
    return f"{article} {adj[1]}{ending} {col[1]}{ending} {form}"


def clause(scene, lang, inverse=False, foil=False):
    a, b = scene["a"], scene["b"]
    task = scene["task"]
    if task == "roles":
        v1, v2 = scene["verb"], scene["verb2"]
        if foil:
            v1, v2 = v2, v1
        one = {**scene, "task": "roles_atom", "verb": v1}
        two = {**scene, "task": "roles_atom", "a": b, "b": a, "verb": v2}
        return join_clauses(clause(one, lang, inverse), clause(two, lang, inverse), lang)
    if task == "negation":
        one = {**scene, "task": "roles_atom", "neg": scene["neg"] ^ foil}
        two = {**scene, "task": "roles_atom", "a": b, "b": a, "neg": not one["neg"]}
        left, right = clause(one, lang, inverse), clause(two, lang, inverse)
        return join_clauses(left, right, lang)
    if task == "space" and "c" in scene:
        one = {k: v for k, v in scene.items() if k not in ("c", "d")}
        two = {**one, "a": scene["c"], "b": scene["d"], "direction": not scene["direction"]}
        left, right = clause(one, lang, inverse, foil), clause(two, lang, inverse, foil)
        return join_clauses(left, right, lang)
    if task == "roles" and foil:
        a, b = b, a
    verb = VERBS[scene["verb"]]
    neg = scene["neg"] ^ (foil and task == "negation")
    # Return a main clause and German subordinate clause.
    if task in ("roles", "roles_atom", "negation"):
        if lang == "en":
            if inverse:
                s = f"{noun_phrase(b, lang)} is {'not ' if neg else ''}{verb[2]} by {noun_phrase(a, lang)}"
            else:
                s = f"{noun_phrase(a, lang)} {'does not ' + verb[1] if neg else verb[0]} {noun_phrase(b, lang)}"
            return s, s
        if lang == "de":
            if inverse:
                subject = noun_phrase(b, lang)
                rest = f"{'nicht ' if neg else ''}von {noun_phrase(a, lang, 'dat')} {verb[4]}"
                return f"{subject} wird {rest}", f"{subject} {rest} wird"
            subject, obj = noun_phrase(a, lang), noun_phrase(b, lang, "acc")
            rest = f"{obj}{' nicht' if neg else ''}"
            return f"{subject} {verb[3]} {rest}", f"{subject} {rest} {verb[3]}"
        subject = particle(noun_phrase(a, lang), ("이", "가"))
        obj = particle(noun_phrase(b, lang), ("을", "를"))
        ending = ("보지 않는다", "맞이하지 않는다", "뒤따르지 않는다")[scene["verb"]] if neg else verb[5]
        s = f"{obj} {subject} {ending}" if inverse else f"{subject} {obj} {ending}"
        return s, s
    if task == "space":
        axis = scene["axis"]
        direction = scene["direction"] ^ foil
        if inverse:
            a, b = b, a
            direction = not direction
        en = (("to the left of", "to the right of"), ("above", "below"), ("in front of", "behind"))[axis][int(direction)]
        de = (("links von", "rechts von"), ("oberhalb von", "unterhalb von"), ("vor", "hinter"))[axis][int(direction)]
        ko = (("왼쪽", "오른쪽"), ("위", "아래"), ("앞", "뒤"))[axis][int(direction)]
        if lang == "en":
            s = f"{noun_phrase(a, lang)} is {en} {noun_phrase(b, lang)}"
            return s, s
        if lang == "de":
            s, rest = noun_phrase(a, lang), f"{de} {noun_phrase(b, lang, 'dat')}"
            return f"{s} ist {rest}", f"{s} {rest} ist"
        s = f"{particle(noun_phrase(a, lang), ('은','는'))} {noun_phrase(b, lang)}의 {ko}에 있다"
        return s, s
    c1, c2 = scene["counts"]
    if foil:
        c1, c2 = c2, c1  # same bag of number words in true and false examples
    parts = [noun_phrase(a, lang, count=c1), noun_phrase(b, lang, count=c2)]
    if inverse:
        parts.reverse()
    if lang == "en":
        s = f"there are {parts[0]} and {parts[1]}"
        return s, s
    if lang == "de":
        tail = f"{parts[0]} und {parts[1]}"
        return f"es gibt {tail}", f"es {tail} gibt"
    s = f"{particle(parts[0], ('과','와'))} {particle(parts[1], ('이','가'))} 있다"
    return s, s


def render(scene, lang, split, inverse=False, foil=False):
    main, sub = clause(scene, lang, inverse, foil)
    if lang == "en":
        text = {"train": f"{main}.", "dev_a": f"Is it true that {main}?",
                "dev_b": f"Someone says that {main}.", "test": f"If {main}, a bell rings."}[split]
    elif lang == "de":
        text = {"train": f"{main}.", "dev_a": f"Stimmt es, dass {sub}?",
                "dev_b": f"Jemand sagt, dass {sub}.", "test": f"Wenn {sub}, klingelt eine Glocke."}[split]
    else:
        # Copular 이다 takes 이라는/이라고/이라면, unlike 있다/본다.
        quoted = main[:-2]+'이라는' if main.endswith('이다') else main+'는'
        reported = main[:-2]+'이라고' if main.endswith('이다') else main+'고'
        conditional = main[:-2]+'이라면' if main.endswith('이다') else main+'면'
        text = {"train": f"{main}.", "dev_a": f"{quoted} 것이 사실인가?",
                "dev_b": f"{reported} 누군가 말한다.", "test": f"만약 {conditional} 종이 울린다."}[split]
    return unicodedata.normalize("NFC", text[0].upper() + text[1:])


def example_record(orbit, split, task, scene, pair, label, lang):
    """Primary rendering (training frame) plus, for held-out splits, the auxiliary outer-frame rendering of the same item."""
    record = dict(id=f"{orbit}:{label}:{lang}", meaning_id=orbit, foil_group=orbit,
                  template_family=PRIMARY_FAMILY, split=split, task=task, language=lang,
                  label=label, gender_pair=pair, scene=scene,
                  sentence_a=render(scene, lang, "train"),
                  sentence_b=render(scene, lang, "train", inverse=True, foil=not label))
    if split != "train":
        record["auxiliary"] = dict(template_family=FAMILIES[split], sentence_a=render(scene, lang, split),
                                   sentence_b=render(scene, lang, split, inverse=True, foil=not label))
    return record


def dataset_manifest(version, seed, files, train_per_task, panel_per_task, independent_test, **extra):
    manifest = dict(version=version, seed=seed, files=files, train_per_task=train_per_task, panel_per_task=panel_per_task,
                    human_review="pending", families=FAMILIES,
                    evaluation=dict(
                        primary=dict(template_family=PRIMARY_FAMILY, fields=["sentence_a", "sentence_b"], shared_with_training=True,
                                     purpose="held-out meaning combinations rendered in the training (declarative) frame; "
                                             "scheduled panels, mastery and the independent test use this rendering"),
                        auxiliary=dict(field="auxiliary", template_families={s: FAMILIES[s] for s in EVALUATION_SPLITS},
                                       same_items_as_primary=True,
                                       purpose="the same held-out items rendered in a held-out outer frame; outer-frame transfer only, never mastery"),
                        note="Primary and auxiliary are two renderings of one item: same id, meaning_id, scene and label."),
                    independent_test=independent_test,
                    note="Semantic orbits, foils and translations are split-exclusive. Outer construction families are held out "
                         "only in the auxiliary rendering; embedded grammar rules and the declarative frame are shared. "
                         "Animal colors form a controlled fictional domain.", **extra)
    manifest["dataset_hash"] = digest(files)
    return manifest


def write_split(path, records):
    with Path(path).open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return file_hash(path)


def generate(root, train_per_task=5000, panel_per_task=1000, seed=1729, version=None):
    root = Path(root)
    if (root / "manifest.json").exists():
        raise FileExistsError("Use a new version directory; do not overwrite a dataset")
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows = {split: [] for split in SPLITS}
    capacities = {split: train_per_task if split == "train" else panel_per_task for split in SPLITS}
    # A label pair shares the orbit, so foil leakage is checked explicitly.
    if any(n % 2 for n in capacities.values()):
        raise ValueError("Each task requires an even number of examples")
    for task in TASKS:
        counts = Counter()
        used = set()
        attempt = 0
        while any(counts[s] < capacities[s] for s in SPLITS):
            attempt += 1
            if attempt > 2000000:
                raise RuntimeError("Insufficient semantic orbits")
            # Cycle gender pairs separately inside each split through quotas below.
            a = (rng.randrange(12), rng.randrange(6), rng.randrange(4))
            b = (rng.randrange(12), rng.randrange(6), rng.randrange(4))
            if a == b:
                continue
            scene = dict(task=task, a=a, b=b, verb=rng.randrange(3), neg=bool(rng.randrange(2)),
                         axis=rng.randrange(3), direction=bool(rng.randrange(2)),
                         counts=rng.sample(range(7), 2))
            if task == "roles":
                scene["neg"] = False
                scene["verb2"] = rng.choice([v for v in range(len(VERBS)) if v != scene["verb"]])
            key = dict(task=task, entities=sorted([a, b]))
            if task == "roles":
                key["verbs"] = sorted([scene["verb"], scene["verb2"]])
            if task == "negation":
                key["verb"] = scene["verb"]
            if task == "space":
                extras = []
                while len(extras) < 2:
                    candidate = (rng.randrange(12), rng.randrange(6), rng.randrange(4))
                    if candidate not in [a, b] + extras:
                        extras.append(candidate)
                scene["c"], scene["d"] = extras
                key["entities"] = sorted([a, b] + extras)
                key["axis"] = scene["axis"]
            orbit = digest(key)
            split = SPLITS[int(orbit[:8], 16) % 4]
            if orbit in used or counts[split] >= capacities[split]:
                continue
            pair = NOUNS[a[0]][6] + NOUNS[b[0]][6]
            quota_index = (counts[split] // 2) % 9
            target_pair = "mfn"[quota_index // 3] + "mfn"[quota_index % 3]
            if pair != target_pair:
                continue
            used.add(orbit)
            for label in (0, 1):
                for lang in LANGUAGES:
                    rows[split].append(example_record(orbit, split, task, scene, pair, label, lang))
                counts[split] += 1
    files = {f"{split}.jsonl": write_split(root / f"{split}.jsonl", records) for split, records in rows.items()}
    manifest = dataset_manifest(version or root.name, seed, files, train_per_task, panel_per_task,
                                independent_test=dict(confirmatory=True, note="Fresh test orbits; apply once at a run's terminal state."))
    write_json(root / "manifest.json", manifest)
    export_review(root, rows)
    return audit(root)


def revise_evaluation(source, output, test_used_in_exploration=True):
    """Derive a new version from an existing one: identical training bytes and split ownership; every
    held-out item gains a primary (training-frame) rendering while its former sentences become the auxiliary."""
    import shutil
    source, output = Path(source), Path(output)
    if output.resolve() == source.resolve() or (output / "manifest.json").exists():
        raise FileExistsError("Use a new version directory; do not overwrite a dataset")
    origin = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    for name, sha in origin["files"].items():
        if file_hash(source / name) != sha:
            raise ValueError(f"Source file changed since its manifest: {name}")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / "train.jsonl", output / "train.jsonl")
    rows = {"train": load_rows(source, "train")}
    for split in EVALUATION_SPLITS:
        rows[split] = []
        for r in load_rows(source, split):
            record = example_record(r["meaning_id"], split, r["task"], r["scene"], r["gender_pair"], r["label"], r["language"])
            former = r.get("auxiliary", r)
            if record["id"] != r["id"] or (record["auxiliary"]["sentence_a"], record["auxiliary"]["sentence_b"]) != (former["sentence_a"], former["sentence_b"]):
                raise ValueError(f"Source row cannot be re-rendered from its scene: {r['id']}")
            rows[split].append(record)
        write_split(output / f"{split}.jsonl", rows[split])
    files = {f"{split}.jsonl": file_hash(output / f"{split}.jsonl") for split in SPLITS}
    independent_test = (dict(confirmatory=False, reason="test orbits were already evaluated in exploratory runs of the source version; "
                                                        "create a new confirmatory test set before the main experiment")
                        if test_used_in_exploration else dict(confirmatory=True, note="Test orbits unused so far."))
    manifest = dataset_manifest(output.name, origin["seed"], files, origin["train_per_task"], origin["panel_per_task"], independent_test,
                                derived_from=dict(version=origin["version"], dataset_hash=origin["dataset_hash"], files=origin["files"],
                                                  training_split_identical=files["train.jsonl"] == origin["files"]["train.jsonl"],
                                                  orbit_ownership_preserved=True))
    write_json(output / "manifest.json", manifest)
    export_review(output, rows)
    return audit(output)


def require_confirmatory_test(manifest):
    """Study cohorts must not reuse a test split that exploration already consumed."""
    status = manifest.get("independent_test", {})
    if status.get("confirmatory") is False:
        raise ValueError(f"The test split is not confirmatory ({status.get('reason', 'consumed before the study')}); "
                         "create a new confirmatory test set first")


def load_rows(root, split):
    with (Path(root) / f"{split}.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def export_review(root, rows):
    records = [r for split in SPLITS for r in rows[split]]
    fields = ["id", "language", "split", "task", "sentence_a", "sentence_b", "label", "reviewer", "judged_label", "fluent", "comment"]
    with (root / "audit-sample.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for lang in LANGUAGES:
            chosen = []
            groups = defaultdict(list)
            for r in records:
                if r["language"] == lang:
                    groups[(r["split"], r["task"], r["label"])].append(r)
            for key in groups:
                groups[key].sort(key=lambda r: digest(r["id"]))
            target = min(200, sum(map(len, groups.values())))
            while len(chosen) < target:
                progressed = False
                for group in groups.values():
                    if group and len(chosen) < 200:
                        chosen.append(group.pop())
                        progressed = True
                if not progressed:
                    break
            for r in chosen:
                for reviewer in ("reviewer_1", "reviewer_2"):
                    writer.writerow({**r, "reviewer": reviewer})
    write_json(root / "template-inventory.json", dict(nouns=NOUNS, colors=COLORS, sizes=SIZES,
               verbs=VERBS, families=FAMILIES, rendering_source_hash=file_hash(__file__), review_status="pending"))


def audit(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    errors, owners, family_owners, identifiers = [], {}, {}, set()
    counts, labels, genders, views = Counter(), Counter(), Counter(), Counter()
    translations = defaultdict(set)
    train_pairs = {(r["sentence_a"], r["sentence_b"]) for r in load_rows(root, "train")}
    two_renderings = "evaluation" in manifest
    for split in SPLITS:
        path = root / f"{split}.jsonl"
        if file_hash(path) != manifest["files"][path.name]:
            errors.append(f"hash mismatch: {split}")
        for r in load_rows(root, split):
            if r["id"] in identifiers:
                errors.append("duplicate example id")
            identifiers.add(r["id"])
            for key in (r["meaning_id"], r["foil_group"]):
                if owners.setdefault(key, split) != split:
                    errors.append("semantic/foil leakage")
            # The held-out outer frame lives in the auxiliary rendering; the primary frame is shared by design.
            family = r["auxiliary"]["template_family"] if "auxiliary" in r else r["template_family"]
            if family_owners.setdefault(family, split) != split:
                errors.append("template family leakage")
            if r["split"] != split or r["task"] not in TASKS or r["language"] not in LANGUAGES:
                errors.append("invalid record metadata")
            if split != "train":
                if (r["sentence_a"], r["sentence_b"]) in train_pairs:
                    errors.append("primary sentence leakage")
                    views["primary_pairs_in_train"] += 1
                if "auxiliary" in r:
                    views["auxiliary_rows"] += 1
                    if r["template_family"] != PRIMARY_FAMILY:
                        errors.append("primary rendering is not the training frame")
                    expected = example_record(r["meaning_id"], split, r["task"], r["scene"], r["gender_pair"], r["label"], r["language"])
                    if (r["sentence_a"], r["sentence_b"]) == (expected["sentence_a"], expected["sentence_b"]):
                        views["primary_rerendered"] += 1
                    else:
                        errors.append("primary rendering mismatch")
                    if r["auxiliary"] == expected["auxiliary"]:
                        views["auxiliary_rerendered"] += 1
                    else:
                        errors.append("auxiliary/primary mismatch")
                elif two_renderings:
                    errors.append("missing auxiliary rendering")
            counts[(split, r["language"], r["task"])] += 1
            labels[(split, r["language"], r["task"], r["label"])] += 1
            genders[(split, r["task"], r["gender_pair"])] += r["language"] == "de"
            translations[(r["meaning_id"], r["label"])].add(r["language"])
    for split in SPLITS:
        expected = manifest["train_per_task"] if split == "train" else manifest["panel_per_task"]
        for lang in LANGUAGES:
            for task in TASKS:
                if counts[(split, lang, task)] != expected:
                    errors.append(f"wrong panel size: {split}/{lang}/{task}")
                if labels[(split, lang, task, 0)] != labels[(split, lang, task, 1)]:
                    errors.append("label imbalance")
        for task in TASKS:
            values = [genders[(split, task, a+b)] for a in "mfn" for b in "mfn"]
            if max(values) - min(values) > 2:
                errors.append("German gender quota violation")
    if any(group != set(LANGUAGES) for group in translations.values()):
        errors.append("missing aligned translation")
    report = dict(passed=not errors, errors=sorted(set(errors)), dataset_hash=manifest["dataset_hash"],
                  examples=len(identifiers), counts={"/".join(k): v for k, v in counts.items()},
                  gender_counts={"/".join(k): v for k, v in genders.items()}, human_review="pending",
                  view_checks={k: views[k] for k in ("auxiliary_rows", "primary_rerendered", "auxiliary_rerendered", "primary_pairs_in_train")})
    write_json(root / "audit.json", report)
    return report


def train_tokenizer(root, output, vocab_size=4096):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers
    root, output = Path(root), Path(output)
    rows = load_rows(root, "train")
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.NFC()
    # Whole-string byte BPE: the controlled lexicon exhausts word-local merges
    # before 1,024; explicit cross-space merges make 1,024/4,096 distinct policies.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2,
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                 special_tokens=["[PAD]", "[UNK]", "[SEP]", "[EOS]"])
    tokenizer.train_from_iterator((s for r in rows for s in (r["sentence_a"], r["sentence_b"])), trainer)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output))
    lengths = defaultdict(list)
    for split in SPLITS:
        for row in load_rows(root, split):
            lengths[f"{split}/{row['language']}/{row['task']}"].append(len(encode_pair(tokenizer, row)))
    import numpy as np
    if tokenizer.get_vocab_size() != vocab_size:
        raise ValueError("Corpus cannot support the requested vocabulary; do not silently relabel a smaller tokenizer")
    report = dict(vocab_requested=vocab_size, vocab_actual=tokenizer.get_vocab_size(), boundary_policy="whole_string_byte_BPE_use_regex_false",
                  tokenizer_hash=file_hash(output), training_file_hash=file_hash(root / "train.jsonl"),
                  dataset_hash=json.loads((root / "manifest.json").read_text())["dataset_hash"],
                  excluded_examples=0, lengths={k: dict(mean=float(np.mean(v)), max=max(v),
                      p95=float(np.quantile(v, .95)), over_128=sum(n > 128 for n in v)) for k, v in lengths.items()})
    write_json(output.with_suffix(".meta.json"), report)
    return report


def encode_pair(tokenizer, row):
    return (tokenizer.encode(row["sentence_a"]).ids + [tokenizer.token_to_id("[SEP]")] +
            tokenizer.encode(row["sentence_b"]).ids + [tokenizer.token_to_id("[EOS]")])


def nuisance_features(row):
    a, b = row["sentence_a"], row["sentence_b"]
    wa, wb = re.findall(r"\w+", a.lower()), re.findall(r"\w+", b.lower())
    sa, sb = set(wa), set(wb)
    return [len(a), len(b), len(a)-len(b), len(wa), len(wb), len(sa & sb)/max(1, len(sa | sb)),
            float(a.lower() == b.lower())]


def baselines(root):
    """Small logistic nuisance-only classifier; high scores trigger review, not an automatic language ranking."""
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import expit
    train, heldout = load_rows(root, "train"), load_rows(root, "dev_a")
    report = {}
    for lang in LANGUAGES:
        for task in TASKS:
            tr = [r for r in train if r["language"] == lang and r["task"] == task]
            te = [r for r in heldout if r["language"] == lang and r["task"] == task]
            x, z = np.array([nuisance_features(r) for r in tr]), np.array([nuisance_features(r) for r in te])
            mean, std = x.mean(0), x.std(0).clip(.001)
            x = np.column_stack([np.ones(len(x)), (x-mean)/std])
            z = np.column_stack([np.ones(len(z)), (z-mean)/std])
            y = np.array([r["label"] for r in tr])
            def objective(w):
                logit = x @ w
                loss = (np.logaddexp(0, logit)-y*logit).mean() + .001*(w[1:]**2).sum()
                gradient = x.T @ (expit(logit)-y)/len(y)
                gradient[1:] += .002*w[1:]
                return loss, gradient
            fit = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B")
            score = float(np.mean((z @ fit.x >= 0) == np.array([r["label"] for r in te])))
            report[f"{lang}/{task}"] = dict(accuracy=score, audit_required=score > .55,
                                             interpretation="nuisance features only", optimizer_success=bool(fit.success))
    write_json(Path(root) / "nuisance-baselines.json", report)
    write_json(Path(root) / "nuisance-baselines.meta.json", dict(
        data_files={split: file_hash(Path(root)/f'{split}.jsonl') for split in ('train','dev_a')},
        report_hash=file_hash(Path(root)/'nuisance-baselines.json')))
    return report
