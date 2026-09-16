"""Word-boundary byte BPE: merges never cross whitespace; a vocabulary shortfall is recorded, never hidden or padded."""
import json

import pytest
from tokenizers import Tokenizer

from flystudy.data import generate, train_tokenizer


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("boundary")
    generate(root/"data", 18, 2)
    return root


def test_word_boundary_tokenizer_never_merges_across_spaces(data):
    report = train_tokenizer(data/"data", data/"word.json", vocab_size=320, boundary="word", strict_vocab=False)
    tok = Tokenizer.from_file(str(data/"word.json"))
    tokens = [tok.id_to_token(i) for i in range(tok.get_vocab_size())]
    assert all("Ġ" not in t[1:] for t in tokens if not t.startswith("["))  # a space may only lead a token
    assert report["boundary_policy"] == "word_boundary_byte_BPE_use_regex_true"
    assert report["vocab_actual"] == tok.get_vocab_size() <= 320
    assert report["vocab_shortfall"] == 320-report["vocab_actual"] >= 0
    meta = json.loads((data/"word.meta.json").read_text(encoding="utf-8"))
    assert meta["vocab_actual"] == report["vocab_actual"] and meta["boundary_policy"] == report["boundary_policy"]


def test_whole_string_tokenizer_keeps_its_recorded_policy(data):
    report = train_tokenizer(data/"data", data/"whole.json", vocab_size=320)
    assert report["boundary_policy"] == "whole_string_byte_BPE_use_regex_false" and report["vocab_shortfall"] == 0
    tok = Tokenizer.from_file(str(data/"whole.json"))
    assert any("Ġ" in tok.id_to_token(i)[1:] for i in range(tok.get_vocab_size()))  # cross-space merges exist here


def test_vocabulary_shortfall_is_refused_unless_explicitly_allowed(data):
    with pytest.raises(ValueError, match="vocabulary"):
        train_tokenizer(data/"data", data/"strict.json", vocab_size=100000, boundary="word")
    with pytest.raises(ValueError):
        train_tokenizer(data/"data", data/"bad.json", vocab_size=320, boundary="sentence")
