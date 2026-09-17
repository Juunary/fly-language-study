"""Second review-driven template revision (after the Claude-only review of draft-v4.6): the third verb is 'chases' in
all three languages (DE-R1 flagged 'verfolgt' as pursue rather than neutral follow) and the two horizontal spatial axes
use cardinal directions, whose converses are frame-independent (KO-R2 flagged left/right and front/behind as ambiguous
between viewer-relative and intrinsic readings). Semantic content and the generation contract are untouched."""
import pytest

from flystudy.data import VERBS, coordinate_ko, render

# Candidate v4.7 / protocol v5.2 (commit 5b955e4) is preserved but NOT adopted (operator decision 2026-09-17):
# the study runs on data v4.6 / protocol v5.1, whose renderer is the default. These checks apply only when the
# candidate renderer is active, so they are skipped rather than deleted.
pytestmark = pytest.mark.skipif(VERBS[2][0] != "chases", reason="candidate v4.7 renderer not adopted; default renderer is v4.6 (protocol v5.1)")

DOG, CAT, COW, DUCK = (0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)


def space_scene(axis, direction=False):
    return dict(task="space", a=DOG, b=CAT, c=COW, d=DUCK, verb=0, neg=False, axis=axis, direction=direction, counts=[0, 1])


def test_third_verb_is_chase_in_every_language():
    assert VERBS[2][:3] == ("chases", "chase", "chased") and VERBS[2][3:5] == ("jagt", "gejagt") and VERBS[2][5] == "쫓는다"
    scene = dict(task="negation", a=DOG, b=CAT, verb=2, neg=True, axis=0, direction=False, counts=[0, 1])
    assert "does not chase" in render(scene, "en", "train") and "is not chased by" in render(scene, "en", "train", inverse=True)
    assert "jagt" in render(scene, "de", "train") and "gejagt" in render(scene, "de", "train", inverse=True)
    assert "쫓지 않는다" in render(scene, "ko", "train") and coordinate_ko("작은 개가 큰 고양이를 쫓는다") == "작은 개가 큰 고양이를 쫓으며"
    for lang in ("en", "de", "ko"):
        assert "verfolg" not in render(scene, lang, "train") and "follow" not in render(scene, lang, "train") and "뒤따" not in render(scene, lang, "train")


def test_horizontal_axes_use_cardinal_directions_with_exact_converses():
    west, east = render(space_scene(0), "en", "train"), render(space_scene(0), "en", "train", inverse=True)
    assert " is west of " in west and " is east of " in east
    north, south = render(space_scene(2), "en", "train"), render(space_scene(2), "en", "train", inverse=True)
    assert " is north of " in north and " is south of " in south
    assert " is above " in render(space_scene(1), "en", "train")  # the vertical axis is unchanged
    de = render(space_scene(0), "de", "train")
    assert " ist westlich von " in de and "östlich von" in render(space_scene(0), "de", "train", inverse=True)
    assert " ist nördlich von " in render(space_scene(2), "de", "train") and "südlich von" in render(space_scene(2), "de", "train", inverse=True)
    ko = render(space_scene(0), "ko", "train")
    assert "의 서쪽에 있다" in ko and "의 동쪽에 있다" in render(space_scene(0), "ko", "train", inverse=True)
    assert "의 북쪽에 있다" in render(space_scene(2), "ko", "train") and "의 남쪽에 있다" in render(space_scene(2), "ko", "train", inverse=True)
    for lang in ("en", "de", "ko"):
        for axis in (0, 2):
            for text in (render(space_scene(axis), lang, "train"), render(space_scene(axis), lang, "dev_a")):
                assert not any(w in text for w in ("left", "right", "front", "behind", "links", "rechts", "vor ", "hinter", "왼쪽", "오른쪽", "앞에", "뒤에")), text
