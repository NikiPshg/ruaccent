"""Pipeline logic with stubbed models: no downloads, no ONNX."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from ruaccent import RUAccent


def _entities(text, labels):
    """Fake token-classifier output: one entity per whitespace-separated chunk with
    punctuation split off like a BERT tokenizer would do it."""
    import re

    entities = []
    chunks = [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"\w+|[^\w\s]", text)]
    assert len(chunks) == len(labels), (chunks, labels)
    for (start, end, word), label in zip(chunks, labels):
        entities.append({"entity": label, "score": 1.0, "word": word, "start": start, "end": end})
    return entities


class FakeStressUsage:
    def __init__(self):
        self.calls = 0
        self.lock = threading.Lock()

    def predict_stress_usage(self, text):
        with self.lock:
            self.calls += 1
        import re

        labels = ["PUNCT" if not re.match(r"\w", w) else "STRESS" for w in re.findall(r"\w+|[^\w\s]", text)]
        return _entities(text, labels)


class FakeYo:
    def predict_yo_homographs(self, text):
        import re

        labels = ["PUNCT" if not re.match(r"\w", w) else "NO_YO" for w in re.findall(r"\w+|[^\w\s]", text)]
        return _entities(text, labels)


class FakeAccentModel:
    def put_accent(self, word):
        # stress the first vowel
        for i, ch in enumerate(word):
            if ch.lower() in "аеёиоуыэюя":
                return word[:i] + "+" + word[i:]
        return word


class FakeOmograph:
    def classify(self, texts, hypotheses, num_hypotheses):
        # always pick the first hypothesis of each group
        out, i = [], 0
        for n in num_hypotheses:
            out.append(hypotheses[i])
            i += n
        return out


@pytest.fixture
def acc():
    a = RUAccent(cache_size=64)
    a.stress_usage_predictor = FakeStressUsage()
    a.yo_homograph_model = FakeYo()
    a.accent_model = FakeAccentModel()
    a.omograph_model = FakeOmograph()
    a.accents = {"замок": "зам+ок", "банк": "б+анк", "о": "+о"}
    a.omographs = {"мука": ["м+ука", "мук+а"]}
    a.yo_words = {"еще": "ещё"}
    a.yo_homographs = {}
    a.loaded = True
    return a


def test_dictionary_and_model_stress(acc):
    # "дверь" has a single vowel: never sent to the accent model, stays as is
    assert acc.process_all("Банк и замок, и дверь.") == "Б+анк и зам+ок, и дверь."


def test_non_destructive_punctuation(acc):
    src = 'Он сказал: "замок сломан", 20% и №5 — ок… (в скобках) [и] {тут} https://a.b/c?x=1'
    out = acc.process_all(src)
    assert out.replace("+", "") == src


def test_alignment_survives_merged_punctuation(acc):
    # "»," is one token for us but two for the tokenizer: words after it must still be stressed
    out = acc.process_all("«Сбербанк», как и банк, закрыл договор.")
    assert "б+анк" in out and "з+акрыл" in out  # fake model stresses the first vowel


def test_premarked_stress_is_kept(acc):
    assert acc.process_all("Зам+ок уже стоит.") == "Зам+ок +уже ст+оит."
    assert acc.process_all("+Он пришёл.") == "+Он пр+ишёл."


def test_yo_restoration(acc):
    assert acc.process_all("еще раз") == "+ещё раз"  # "раз": one vowel, untouched


def test_omograph_choice(acc):
    assert acc.process_all("мука") == "м+ука"


def test_whitespace_and_sentences_preserved(acc):
    src = "  Первое.   Второе?\nТретье  "
    out = acc.process_all(src)
    assert out.replace("+", "") == src


def test_control_chars_removed_only(acc):
    assert acc.process_all("сл​ово\x07").replace("+", "") == "слово"


def test_cache_key_ignores_surrounding_whitespace(acc):
    acc.process_all("Первое предложение. Второе предложение.")
    info = acc.cache_info()
    assert info.misses == 2
    acc.process_all("Второе предложение.")
    assert acc.cache_info().hits == 1
    acc.process_all("   Второе предложение.   ")
    assert acc.cache_info().hits == 2
    acc.clear_cache()
    assert acc.cache_info().currsize == 0


def test_cache_is_per_instance(acc):
    other = RUAccent()
    acc.process_all("Банк.")
    assert acc.cache_info().currsize == 1
    assert other.cache_info().currsize == 0


def test_skip_regex(acc):
    assert acc.process_all("банк [skip банк] банк", skip_regex=r"\[.*?\]") == "б+анк [skip банк] б+анк"


def test_thread_safety_and_determinism(acc):
    sentences = [f"Банк номер {i} и замок." for i in range(50)]
    expected = {s: acc.process_all(s) for s in sentences}
    acc.clear_cache()
    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(acc.process_all, sentences * 5))
    assert results == [expected[s] for s in sentences * 5]


def test_empty_and_no_word_inputs(acc):
    assert acc.process_all("") == ""
    assert acc.process_all("   ") == "   "
    assert acc.process_all("!!! ...") == "!!! ..."


def test_process_yo_only(acc):
    assert acc.process_yo("еще раз, еще") == "ещё раз, ещё"


def test_unknown_omograph_model_name():
    with pytest.raises(ValueError):
        RUAccent._asset_patterns("nope", True, False)


def test_default_workdir_is_not_package_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("RUACCENT_WORKDIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert RUAccent.default_workdir() == str(tmp_path / "ruaccent")
    monkeypatch.setenv("RUACCENT_WORKDIR", "/srv/ruaccent")
    assert RUAccent.default_workdir() == "/srv/ruaccent"
