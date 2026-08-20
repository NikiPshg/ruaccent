"""End-to-end checks with the real models (``pytest --models``). Downloads ~0.5 GB
into RUACCENT_WORKDIR / ~/.cache/ruaccent on first run."""

import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from ruaccent import RUAccent

pytestmark = pytest.mark.models


@pytest.fixture(scope="module")
def acc():
    return RUAccent().load(
        omograph_model_size=os.environ.get("RUACCENT_TEST_MODEL", "turbo3.1"),
        use_dictionary=True,
        tiny_mode=False,
        num_threads=1,
    )


def test_basic(acc):
    assert acc.process_all("на двери висит замок.") == "на двер+и вис+ит зам+ок."


def test_omographs_and_yo(acc):
    out = acc.process_all("Я не знал, что замок такой старый. Все ели ёлку и еще пели.")
    assert "з+амок" in out and "ещ+ё" in out


def test_alignment_after_merged_punctuation(acc):
    out = acc.process_all("«Сбербанк», как и банк, закрыл договор.")
    assert "б+анк" in out and "закр+ыл" in out


def test_non_destructive(acc):
    src = 'Скидка 20% на товар №5 — это "хорошо"… Сайт https://example.com/путь?x=1 не работает.'
    assert acc.process_all(src).replace("+", "") == src


def test_cache_and_threads(acc):
    acc.clear_cache()
    sentences = [f"Клиент номер {i} внёс платёж по договору." for i in range(40)]
    expected = {s: acc.process_all(s) for s in sentences}
    acc.clear_cache()
    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(acc.process_all, sentences * 3))
    assert results == [expected[s] for s in sentences * 3]
    assert acc.cache_info().hits > 0
