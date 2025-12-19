r"""Contain the main features of the ``coola`` package."""

__all__ = [
    "BaseSummarizer",
    "Reduction",
    "Summarizer",
    "__version__",
    "objects_are_allclose",
    "objects_are_allclose",
    "objects_are_equal",
    "set_summarizer_options",
    "summarizer_options",
    "summary",
]

from importlib.metadata import PackageNotFoundError, version

from coola.comparison import objects_are_allclose, objects_are_equal
from coola.reduction import Reduction
from coola.summarization import summary
from coola.summarizers import (
    BaseSummarizer,
    Summarizer,
    set_summarizer_options,
    summarizer_options,
)

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    # Package is not installed, fallback if needed
    __version__ = "0.0.0"
