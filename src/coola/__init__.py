r"""Contain the main features of the ``coola`` package."""

__all__ = [
    "Reduction",
    "__version__",
    "objects_are_allclose",
    "objects_are_allclose",
    "objects_are_equal",
    "summary",
]

from importlib.metadata import PackageNotFoundError, version

from coola.comparison import objects_are_allclose, objects_are_equal
from coola.reduction import Reduction

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    # Package is not installed, fallback if needed
    __version__ = "0.0.0"
