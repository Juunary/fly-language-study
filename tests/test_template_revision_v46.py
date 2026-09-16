"""Template revision after the Claude-only review of draft-v4.5 (v4-ai-review-1 on protocol v5): the reviewers' template
issues are fixed in the renderer and catalog before any new data version is generated. Semantic content, orbits, foils
and the generation contract are untouched; only surface forms and review-catalog labels change."""
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from flystudy.data import AGES, COLORS, NOUNS, SIZES, VERBS, coordinate_ko, generate, noun_phrase, particle, render

ROOT = Path(__file__).resolve().parents[1]
DOG, CAT, COW, DUCK = (0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/"scripts"/f"{name}.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def space_scene(axis, direction=False):
    return dict(task="space", a=DOG, b=CAT, c=COW, d=DUCK, verb=0, neg=False, axis=axis, direction=direction, counts=[0, 1])


def roles_scene(verb=1, verb2=0):
    return dict(task="roles", a=DOG, b=CAT, verb=verb, verb2=verb2, neg=False, axis=0, direction=False, counts=[0, 1])


def test_korean_space_frame_marks_the_embedded_subject_with_nominative_not_topic():
    subjects = [noun_phrase(e, "ko") for e in (DOG, CAT, COW, DUCK)]
    for split in ("train", "dev_a", "dev_b", "test"):
        for text in (render(space_scene(1), "ko", split), render(space_scene(1), "ko", split, inverse=True)):
            assert sum(particle(s, ("이", "가")) + " " in text for s in subjects) == 2, text
            assert not any(particle(s, ("은", "는")) + " " in text for s in subjects), text


def test_korean_truth_question_does_not_repeat_the_copula():
    text = render(space_scene(0), "ko", "dev_a")
    assert text.endswith("모두 사실인가?") and "사실이라는 것이 사실인가" not in text
    quantity = dict(task="quantity", a=DOG, b=CAT, verb=0, neg=False, axis=0, direction=False, counts=[0, 1])
    assert render(quantity, "ko", "dev_a").endswith("있다는 것이 사실인가?")
    assert render(space_scene(0), "ko", "dev_b").endswith("모두 사실이라고 누군가 말한다.")
    assert render(space_scene(0), "ko", "test").endswith("모두 사실이라면 종이 울린다.")


def test_german_vertical_relations_use_ueber_and_unter_with_the_dative():
    above, below = render(space_scene(1), "de", "train"), render(space_scene(1, True), "de", "train")
    assert " über dem " in above or " über der " in above
    assert " unter dem " in below or " unter der " in below
    for text in (above, below, render(space_scene(1), "de", "dev_a")):
        assert "oberhalb" not in text and "unterhalb" not in text


def test_hen_is_glossed_consistently_as_chicken():
    entry = next(n for n in NOUNS if n[7] == "닭")
    assert entry[:2] == ("chicken", "chickens") and entry[2] == "Huhn"


def test_korean_greets_uses_bangida_in_every_form():
    assert VERBS[1][5] == "반긴다"
    scene = roles_scene(verb=1, verb2=0)
    assert "반긴다" in render(scene, "ko", "train") and "맞이" not in render(scene, "ko", "train")
    negated = dict(task="negation", a=DOG, b=CAT, verb=1, neg=True, axis=0, direction=False, counts=[0, 1])
    assert "반기지 않는다" in render(negated, "ko", "train")
    assert coordinate_ko("작은 개가 큰 고양이를 반긴다") == "작은 개가 큰 고양이를 반기며"


def test_korean_colours_share_one_form_class():
    assert [c[2] for c in COLORS] == ["빨간색", "파란색", "초록색", "노란색", "검은색", "하얀색"]


def test_sizes_and_ages_are_separate_attribute_lists():
    assert [s[0] for s in SIZES] == ["small", "large"] and [a[0] for a in AGES] == ["young", "old"]
    assert render(roles_scene(), "en", "train").startswith("The small red dog")


def test_generated_inventory_and_catalog_use_the_revised_labels(tmp_path):
    generate(tmp_path/"data", 18, 2, version="draft-test")
    inventory = json.loads((tmp_path/"data"/"template-inventory.json").read_text(encoding="utf-8"))
    assert [s[0] for s in inventory["sizes"]] == ["small", "large"] and [a[0] for a in inventory["ages"]] == ["young", "old"]
    subprocess.run([sys.executable, str(ROOT/"scripts"/"export_review_catalog.py"), "--data", str(tmp_path/"data")], check=True)
    with (tmp_path/"data"/"noun-forms.csv").open(encoding="utf-8-sig", newline="") as f:
        forms = list(csv.DictReader(f))
    assert set(forms[0].keys()) == {"language", "noun", "color", "attribute", "case", "count_index", "surface"}
    assert {r["case"] for r in forms if r["language"] == "ko"} == {"nom", "acc", "gen", "plural"}
    assert {r["case"] for r in forms if r["language"] == "de"} == {"nom", "acc", "dat", "plural"}
    assert any(r["language"] == "ko" and r["case"] == "gen" and r["surface"].endswith("의") for r in forms)
    assert all(r["noun"] in {n[0] for n in NOUNS} and r["attribute"] in {"small", "large", "young", "old"} for r in forms)


def test_checklist_rows_follow_the_language_specific_case_inventory():
    build = load("build_review_packages")
    cases = lambda lang: [r["task"] for r in build.checklist_rows(lang) if r["area"] == "noun_forms"]
    assert cases("ko") == ["nom", "acc", "gen", "plural"] and cases("de") == ["nom", "acc", "dat", "plural"] and cases("en") == ["nom", "acc", "dat", "plural"]


def test_cue_baseline_recognises_the_revised_korean_verb_forms():
    from flystudy.cues import verb_matcher
    pattern, variants = verb_matcher("ko")
    assert variants["반긴다"] == 1 and variants["반기지"] == 1 and "맞이하지" not in variants
    assert pattern.search("작은 빨간색 개가 큰 파란색 여우를 반기지 않는다")
