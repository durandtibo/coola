r"""Top-level package for ``coola``.

Use this package to compare nested objects, summarize complex
structures, and work with helper utilities for recursive transformations
and iteration.
"""

__all__ = ["__version__"]

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    # Package is not installed, fallback if needed
    __version__ = "0.0.0"
