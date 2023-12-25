r"""This package contains the summarizer implementations."""

__all__ = [
    "BaseSummarizer",
    "Summarizer",
    "set_summarizer_options",
    "summarizer_options",
]

from coola.summarizers.base import BaseSummarizer
from coola.summarizers.summarizer import (
    Summarizer,
    set_summarizer_options,
    summarizer_options,
)
