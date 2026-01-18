r"""Contain the main features of the ``coola`` package.

Quick Start
-----------

The most common operations are available at the top level for
convenience:

Compare objects for equality:
    >>> from coola import objects_are_equal
    >>> objects_are_equal([1, 2, 3], [1, 2, 3])
    True

Compare objects with numerical tolerance:
    >>> from coola import objects_are_allclose
    >>> objects_are_allclose([1.0, 2.0], [1.0 + 1e-9, 2.0])
    True

Generate summaries of complex data:
    >>> from coola import summarize
    >>> print(summarize({"a": [1, 2, 3], "b": "hello"}))
    <class 'dict'> (length=2)
      (a): [1, 2, 3]
      (b): hello

For more advanced usage, see the documentation at:
https://durandtibo.github.io/coola/
"""

__all__ = ["__version__", "objects_are_allclose", "objects_are_equal", "summarize"]

from importlib.metadata import PackageNotFoundError, version

from coola.equality import objects_are_allclose, objects_are_equal
from coola.summary import summarize

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    # Package is not installed, fallback if needed
    __version__ = "0.0.0"
