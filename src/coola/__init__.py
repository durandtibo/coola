r"""Contain the main features of the ``coola`` package."""

__all__ = [
    "BaseSummarizer",
    "Reduction",
    "Summarizer",
    "objects_are_allclose",
    "objects_are_allclose",
    "objects_are_equal",
    "set_summarizer_options",
    "summarizer_options",
    "summary",
]

from coola.comparison import objects_are_allclose, objects_are_equal
from coola.reduction import Reduction
from coola.summarization import summary
from coola.summarizers import (
    BaseSummarizer,
    Summarizer,
    set_summarizer_options,
    summarizer_options,
)
