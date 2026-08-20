import pytest

from ruaccent.text_preprocessor import TextPreprocessor


@pytest.mark.parametrize(
    "text",
    [
        "Привет, мир!",
        "«Сбербанк», как и банк, закрыл договор.",
        "Скидка 20% на товар №5 — это хорошо…",
        "  ведущие пробелы и хвост  ",
        "зам+ок и +он",
        "https://example.com/путь?x=1",
        "",
        "   ",
        "!!!",
    ],
)
def test_tokenize_roundtrip(text):
    tokens, gaps, spans = TextPreprocessor.tokenize(text)
    assert len(gaps) == len(tokens) + 1
    assert "".join(g + t for g, t in zip(gaps, tokens)) + gaps[-1] == text
    for token, (start, end) in zip(tokens, spans):
        assert text[start:end] == token


def test_tokenize_keeps_plus_inside_words():
    tokens, _, _ = TextPreprocessor.tokenize("Зам+ок, +он и слово+")
    assert tokens == ["Зам+ок", ",", "+он", "и", "слово", "+"]


def test_tokenize_groups_punctuation_runs():
    tokens, _, _ = TextPreprocessor.tokenize("«Сбер», банк")
    assert tokens == ["«", "Сбер", "»,", "банк"]


def test_split_by_words_compat():
    words, gaps = TextPreprocessor.split_by_words("Привет, мир")
    assert words == ["Привет", ",", "мир"]
    assert gaps == ["", "", " ", ""]
    assert TextPreprocessor.split_by_words("") == ([], ["", ""])


def test_split_by_sentences_preserves_text():
    text = "Первое предложение. Второе предложение!  Третье?"
    sentences = TextPreprocessor.split_by_sentences(text)
    assert "".join(sentences) == text
    assert sentences[1].startswith(" ")
