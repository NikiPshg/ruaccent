import gzip
import json
import logging
import os
import pathlib
import re
from functools import lru_cache
from os.path import join as join_path

from huggingface_hub import snapshot_download

from .accent_model import AccentModel
from .omograph_model import OmographModel
from .stress_usage_model import StressUsagePredictorModel
from .text_postprocessor import fix_capital
from .text_preprocessor import TextPreprocessor
from .yo_homograph_model import YoHomographModel

logger = logging.getLogger(__name__)

DEFAULT_REPO = "ruaccent/accentuator"
# Pinned snapshot of the public model repository. Pass ``revision="main"`` to follow
# upstream updates instead of this reproducible default.
DEFAULT_REVISION = "b78ae5ea1e62beaf138bed1865cd8c3b0b5ca855"

OMOGRAPH_MODELS = {
    "big_poetry": "nn/nn_omograph/big_poetry",
    "medium_poetry": "nn/nn_omograph/medium_poetry",
    "small_poetry": "nn/nn_omograph/small_poetry",
    "turbo": "nn/nn_omograph/turbo",
    "turbo2": "nn/nn_omograph/turbo2",
    "turbo3": "nn/nn_omograph/turbo3",
    "turbo3.1": "nn/nn_omograph/turbo3.1",
    "tiny": "nn/nn_omograph/tiny",
    "tiny2": "nn/nn_omograph/tiny2",
    "tiny2.1": "nn/nn_omograph/tiny2.1",
}

ACCENT_MODEL_DIR = "nn/nn_accent"
YO_MODEL_DIR = "nn/nn_yo_homograph_resolver"
STRESS_USAGE_MODEL_DIR = "nn/nn_stress_usage_predictor"

# The accentuator is non-destructive: the only characters it removes from the input
# are control and zero-width characters that neither the tokenizers nor a TTS
# front-end want to see. Punctuation, symbols, Latin text and digits pass through
# untouched; the output is the input plus "+" stress marks (and "ё" restoration).
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b-\u200d\u2060\ufeff]")
_VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
_PUNCTUATION = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"


class RUAccent:
    def __init__(self, cache_size=4096):
        self.omograph_model = OmographModel()
        self.accent_model = AccentModel()
        self.stress_usage_predictor = StressUsagePredictorModel()
        self.yo_homograph_model = YoHomographModel()
        self.letters_accent = {"о": "+о", "О": "+О"}
        self.tiny_mode = False
        self.accents = {}
        self.omographs = {}
        self.yo_words = {}
        self.yo_homographs = {}
        self.workdir = None
        self.loaded = False
        self._cache_size = cache_size
        self._process_sentence_cached = lru_cache(maxsize=cache_size)(self._process_sentence)

    # ------------------------------------------------------------------ loading
    @staticmethod
    def default_workdir():
        """Directory the model assets are downloaded to when ``workdir`` is not given.

        ``RUACCENT_WORKDIR`` wins; otherwise assets that an older release already
        placed next to the package are reused; otherwise ``$XDG_CACHE_HOME/ruaccent``
        (``~/.cache/ruaccent``). The package directory itself is never written to, so
        the library works from read-only ``site-packages`` and as a non-root user.
        """
        env = os.environ.get("RUACCENT_WORKDIR")
        if env:
            return env
        legacy = pathlib.Path(__file__).resolve().parent
        if (legacy / "dictionary").is_dir():
            return str(legacy)
        cache_home = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
        return os.path.join(cache_home, "ruaccent")

    @staticmethod
    def _asset_patterns(omograph_model_size, use_dictionary, tiny_mode):
        model_dir = OMOGRAPH_MODELS.get(omograph_model_size)
        if model_dir is None:
            raise ValueError(
                f"unknown omograph model {omograph_model_size!r}; choose one of {sorted(OMOGRAPH_MODELS)}"
            )
        accents_file = "dictionary/accents_nn.json.gz" if (tiny_mode or not use_dictionary) else "dictionary/accents.json.gz"
        patterns = [
            "dictionary/omographs.json.gz",
            "dictionary/yo_words.json.gz",
            "dictionary/yo_homographs.json.gz",
            accents_file,
            f"{ACCENT_MODEL_DIR}/*",
            f"{YO_MODEL_DIR}/*",
            f"{model_dir}/*",
        ]
        required = [
            "dictionary/omographs.json.gz",
            "dictionary/yo_words.json.gz",
            "dictionary/yo_homographs.json.gz",
            accents_file,
            f"{ACCENT_MODEL_DIR}/model.onnx",
            f"{YO_MODEL_DIR}/model.onnx",
            f"{model_dir}/model.onnx",
        ]
        if not tiny_mode:
            patterns.append(f"{STRESS_USAGE_MODEL_DIR}/*")
            required.append(f"{STRESS_USAGE_MODEL_DIR}/model.onnx")
        return model_dir, accents_file, patterns, required

    def _ensure_assets(self, repo, revision, workdir, patterns, required, token, local_files_only):
        missing = [path for path in required if not os.path.isfile(join_path(workdir, path))]
        if local_files_only:
            if missing:
                raise FileNotFoundError(f"ruaccent assets are missing from {workdir}: {missing}")
            return
        try:
            snapshot_download(
                repo_id=repo,
                revision=revision,
                allow_patterns=patterns,
                local_dir=workdir,
                token=token,
            )
        except Exception as exc:  # network, auth, offline mode, ...
            if missing:
                raise RuntimeError(
                    f"cannot download ruaccent assets from {repo}@{revision} into {workdir} "
                    f"and the local copy is incomplete ({missing})"
                ) from exc
            logger.warning(
                "ruaccent: using existing assets in %s, refresh from %s@%s failed with %s",
                workdir, repo, revision, type(exc).__name__,
            )

    @staticmethod
    def _load_gz_json(workdir, relative_path):
        with gzip.open(join_path(workdir, relative_path), "rt", encoding="utf-8") as handle:
            return json.load(handle)

    def load(
        self,
        omograph_model_size="turbo2",
        use_dictionary=False,
        custom_dict=None,
        custom_homographs=None,
        providers=None,
        repo=DEFAULT_REPO,
        revision=DEFAULT_REVISION,
        workdir=None,
        tiny_mode=False,
        num_threads=None,
        session_options=None,
        token=None,
        local_files_only=False,
    ):
        """Download (if needed) and load all models.

        ``providers`` are ONNX Runtime execution providers (default CPU).
        ``num_threads`` caps the intra-op threads of every ONNX session (recommended
        ``1`` when the accentuator shares a machine with other workloads; latency per
        sentence is the same, CPU usage is an order of magnitude lower). Pass a
        ``session_options`` object instead for full control.
        ``revision`` pins the Hugging Face snapshot of ``repo``; assets go to
        ``workdir`` (see :meth:`default_workdir`) and are reused on later loads, also
        without network access.
        """
        self.tiny_mode = tiny_mode
        self.workdir = workdir or self.default_workdir()
        os.makedirs(self.workdir, exist_ok=True)
        providers = providers or ["CPUExecutionProvider"]

        model_dir, accents_file, patterns, required = self._asset_patterns(
            omograph_model_size, use_dictionary, tiny_mode
        )
        self._ensure_assets(repo, revision, self.workdir, patterns, required, token, local_files_only)

        sess_options = session_options
        if sess_options is None and num_threads is not None:
            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = int(num_threads)
            sess_options.inter_op_num_threads = 1

        self.omographs = self._load_gz_json(self.workdir, "dictionary/omographs.json.gz")
        self.omographs.update({"коса": ["к+оса", "кос+а"]})
        self.omographs.update(custom_homographs or {})
        self.omograph_model.load(join_path(self.workdir, model_dir), providers=providers, sess_options=sess_options)

        self.yo_words = self._load_gz_json(self.workdir, "dictionary/yo_words.json.gz")
        self.yo_homographs = self._load_gz_json(self.workdir, "dictionary/yo_homographs.json.gz")
        self.accent_model.load(join_path(self.workdir, ACCENT_MODEL_DIR), providers=providers, sess_options=sess_options)
        self.yo_homograph_model.load(join_path(self.workdir, YO_MODEL_DIR), providers=providers, sess_options=sess_options)

        self.accents = {}
        self.accents.update(self._load_gz_json(self.workdir, accents_file))
        self.accents.update(custom_dict or {})
        self.accents.update(self.letters_accent)

        if not tiny_mode:
            self.stress_usage_predictor.load(
                join_path(self.workdir, STRESS_USAGE_MODEL_DIR), providers=providers, sess_options=sess_options
            )

        self.clear_cache()
        self.loaded = True
        return self

    # -------------------------------------------------------------------- cache
    def cache_info(self):
        """``functools`` cache statistics of the per-sentence result cache."""
        return self._process_sentence_cached.cache_info()

    def clear_cache(self):
        self._process_sentence_cached.cache_clear()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def count_vowels(text):
        return sum(1 for char in text if char in _VOWELS)

    @staticmethod
    def has_punctuation(text):
        return any(char in _PUNCTUATION for char in text)

    @staticmethod
    def delete_spaces_before_punc(text):
        punc = "!\"#$%&'()*,./:;<=>?@[\\]^_`{|}-"
        for char in punc:
            if char == "-":
                text = text.replace(" " + char, char).replace(char + " ", char)
            text = text.replace(" " + char, char)
        return text.replace("~", "-")

    @staticmethod
    def _plus_prefix_counts(sentence):
        """``counts[i]`` is the number of "+" characters in ``sentence[:i]``."""
        counts = [0] * (len(sentence) + 1)
        total = 0
        for index, char in enumerate(sentence):
            counts[index] = total
            if char == "+":
                total += 1
        counts[len(sentence)] = total
        return counts

    @staticmethod
    def _labels_by_span(entities, spans, plus_before, default):
        """Map token-classifier entities onto our tokens by character offsets.

        The predictors receive the sentence without "+" marks and report
        ``start``/``end`` offsets in that string; ``plus_before`` translates our token
        offsets into it. Tokens that no entity covers (or whose entity carries no
        offsets) get ``default``. This keeps stress/ё predictions attached to the
        right word whatever the tokenizer does with punctuation runs such as ``»,``.
        """
        ranges = [
            (entity["start"], entity["end"], entity["entity"])
            for entity in entities
            if entity.get("start") is not None and entity.get("end") is not None
        ]
        labels = []
        index = 0
        for start, _end in spans:
            position = start - plus_before[start]
            while index < len(ranges) and ranges[index][1] <= position:
                index += 1
            if index < len(ranges) and ranges[index][0] <= position < ranges[index][1]:
                labels.append(ranges[index][2])
            else:
                labels.append(default)
        return labels

    # ---------------------------------------------------------------- pipeline
    def _process_yo(self, words, lower_core, spans, plus_before):
        yo_labels = None
        if "е" in lower_core:
            entities = self.yo_homograph_model.predict_yo_homographs(lower_core)
            yo_labels = self._labels_by_span(entities, spans, plus_before, "NO_YO")
        for i, word in enumerate(words):
            lower_word = word.lower()
            words[i] = fix_capital(word, self.yo_words.get(lower_word, word))
            if yo_labels and yo_labels[i] == "YO":
                words[i] = fix_capital(word, self.yo_homographs.get(lower_word, word))
        return words

    def _process_omographs(self, words):
        found = []
        hypotheses = []
        for i, word in enumerate(words):
            variants = self.omographs.get(word)
            if variants:
                found.append((i, variants))
                hypotheses.append(variants)
        if not found:
            return words

        texts_batch = []
        for position, variants in found:
            marked = list(words)
            marked[position] = " <w>" + words[position] + "</w> "
            # Context text in the exact shape the homograph classifier was trained on.
            context = self.delete_spaces_before_punc(" ".join(marked).replace(" - ", " ~ "))
            texts_batch.extend([context] * len(variants))
        hypotheses_batch = [variant for variants in hypotheses for variant in variants]
        num_hypotheses = [len(variants) for variants in hypotheses]
        classified = self.omograph_model.classify(texts_batch, hypotheses_batch, num_hypotheses)
        for (position, _variants), choice in zip(found, classified):
            words[position] = choice
        return words

    def _process_accent(self, words, stress_usages):
        for i, word in enumerate(words):
            if "+" in word:
                continue
            if stress_usages[i] != "STRESS":
                continue
            lower_word = word.lower()
            stressed_word = self.accents.get(lower_word, lower_word)
            if stressed_word == lower_word and not self.has_punctuation(lower_word) and self.count_vowels(lower_word) > 1:
                words[i] = self.accent_model.put_accent(word)
            else:
                word_fixed = list(word)
                for j, match in enumerate(re.finditer(r"\+", stressed_word)):
                    word_fixed = word_fixed[: match.start() + j] + ["+"] + list(word)[match.end() - 1 :]
                words[i] = "".join(word_fixed)
        return words

    def _process_sentence(self, sentence):
        """Accentuate one stripped sentence. Results are memoised per sentence."""
        words, gaps, spans = TextPreprocessor.tokenize(sentence)
        if not words:
            return sentence
        plus_before = self._plus_prefix_counts(sentence)
        core = sentence.replace("+", "")

        if self.tiny_mode:
            stress_usages = ["STRESS"] * len(words)
        else:
            entities = self.stress_usage_predictor.predict_stress_usage(core)
            stress_usages = self._labels_by_span(entities, spans, plus_before, "STRESS")

        words = self._process_yo(words, core.lower(), spans, plus_before)
        words = self._process_omographs(words)
        words = self._process_accent(words, stress_usages)
        return "".join(gap + word for gap, word in zip(gaps, words)) + gaps[-1]

    def _process_sentence_keyed(self, sentence):
        """Cache on the stripped sentence so leading/trailing whitespace does not
        fragment the cache, then restore that whitespace around the result."""
        stripped = sentence.strip()
        if not stripped:
            return sentence
        lead = sentence[: len(sentence) - len(sentence.lstrip())]
        trail = sentence[len(sentence.rstrip()) :]
        return lead + self._process_sentence_cached(stripped) + trail

    def process_all_internal(self, text):
        text = _CONTROL_CHARS.sub("", text)
        sentences = TextPreprocessor.split_by_sentences(text)
        if not sentences:
            return text
        return "".join(self._process_sentence_keyed(sentence) for sentence in sentences)

    def process_yo(self, text):
        """Only restore "ё" (dictionary + homograph model); no stress marks."""
        text = _CONTROL_CHARS.sub("", text)
        outputs = []
        for sentence in TextPreprocessor.split_by_sentences(text):
            words, gaps, spans = TextPreprocessor.tokenize(sentence)
            if not words:
                outputs.append(sentence)
                continue
            plus_before = self._plus_prefix_counts(sentence)
            core = sentence.replace("+", "")
            words = self._process_yo(words, core.lower(), spans, plus_before)
            outputs.append("".join(gap + word for gap, word in zip(gaps, words)) + gaps[-1])
        return "".join(outputs) if outputs else text

    def process_all(self, text, skip_regex=None):
        """Put stress marks ("+" before the stressed vowel) into ``text``.

        ``skip_regex`` protects matching spans from processing (they are copied to
        the output verbatim). Text that already carries "+" marks keeps them: marked
        words are never re-stressed.
        """
        if not skip_regex:
            return self.process_all_internal(text)

        pattern = re.compile(skip_regex)
        indices = [(match.start(), match.end()) for match in pattern.finditer(text)]
        if not indices:
            return self.process_all_internal(text)
        skipped = [text[left:right] for left, right in indices]

        elems = [text[: indices[0][0]]]
        for left, right in zip(indices, indices[1:]):
            elems.append(text[left[1] : right[0]])
        elems.append(text[indices[-1][1] :])

        results = [self.process_all_internal(elem) if elem else elem for elem in elems]
        return "".join([results[0]] + [left + right for left, right in zip(skipped, results[1:])])
