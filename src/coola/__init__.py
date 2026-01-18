r"""Contain the main features of the ``coola`` package."""

__all__ = ["__version__"]

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    # Package is not installed, fallback if needed
    __version__ = "0.0.0"
