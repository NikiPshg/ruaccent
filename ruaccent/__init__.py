"""Russian accentizer: puts "+" stress marks into Russian text using ONNX models."""

__version__ = "1.6.0"

from .ruaccent import DEFAULT_REPO, DEFAULT_REVISION, OMOGRAPH_MODELS, RUAccent

__all__ = ["RUAccent", "DEFAULT_REPO", "DEFAULT_REVISION", "OMOGRAPH_MODELS", "__version__"]
