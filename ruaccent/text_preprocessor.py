import re
from razdel import sentenize
from razdel.substring import Substring


class TextPreprocessor:
    # A word is a run of word characters that may carry "+" stress marks inside or in
    # front of it ("зам+ок", "+он"); everything that is neither a word character nor
    # whitespace is a punctuation run.
    TOKEN_RE = re.compile(r"\w*(?:\+\w+)+|\w+|[^\w\s]+")

    @staticmethod
    def tokenize(string):
        """Split a sentence into (tokens, gaps, spans).

        ``tokens`` are words and punctuation runs, ``spans`` their ``(start, end)``
        offsets in ``string`` and ``gaps`` the ``len(tokens) + 1`` pieces of text
        between them, so that ``"".join(g + t for g, t in zip(gaps, tokens)) + gaps[-1]``
        reproduces ``string`` exactly.
        """
        tokens, spans, gaps = [], [], []
        last = 0
        for match in TextPreprocessor.TOKEN_RE.finditer(string):
            tokens.append(match.group(0))
            spans.append((match.start(), match.end()))
            gaps.append(string[last:match.start()])
            last = match.end()
        gaps.append(string[last:])
        return tokens, gaps, spans

    @staticmethod
    def split_by_words(string):
        """Backwards compatible ``(words, remaining_text)`` view of :meth:`tokenize`."""
        tokens, gaps, _ = TextPreprocessor.tokenize(string)
        if not tokens:
            return tokens, ["", ""]
        return tokens, gaps

    @staticmethod
    def split_by_sentences(string):
        sentences = list(sentenize(string))
        if len(sentences) == 0:
            return []
        result = [string[l.stop:r.start] + r.text if l.stop != r.start else r.text for l,r in zip([Substring(0,0, "")] + sentences, sentences)]
        result[-1] = result[-1] + string[sentences[-1].stop:]
        return result
