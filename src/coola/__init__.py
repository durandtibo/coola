r"""Top-level package for ``coola``.

``coola`` provides functionality to compare nested objects, summarize
complex structures, and work with helper utilities for recursive
transformations and iteration.

This top-level module only exposes ``__version__``; it does not re-
export the package's entry points (e.g. ``objects_are_equal``,
``objects_are_allclose``, ``summary``). Import them from their
respective submodules instead, e.g. ``from coola.equality import
objects_are_equal`` or ``from coola.summary import summary``.
"""

__all__ = ["__version__"]

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    # Package is not installed, fallback if needed
    __version__ = "0.0.0"
