r"""Top-level package for ``coola``.

``coola`` provides functionality to compare nested objects, summarize
complex structures, and work with helper utilities for recursive
transformations and iteration.

This top-level module only exposes ``__version__``; it does not re-
export the package's entry points (e.g. ``objects_are_equal``,
``objects_are_allclose``, ``summarize``). Import them from their
respective submodules instead, e.g. ``from coola.equality import
objects_are_equal`` or ``from coola.summary import summarize``.
"""

# Design choice: the top-level package deliberately exposes only ``__version__``. The main
# functions (``objects_are_equal``, ``hash_object``, ``summarize``, ...) are not re-exported
# here, so each one is imported from its own subpackage (``coola.equality``,
# ``coola.hashing``, ``coola.summary``) and the subpackage is the documented path.
__all__ = ["__version__"]

from typing import Any


def __getattr__(name: str) -> Any:
    # PEP 562: the version is read from the metadata of the installed package the first time
    # it is used, because ``importlib.metadata`` is slow to import and every subpackage of
    # ``coola`` would pay for it.
    if name == "__version__":
        from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

        try:
            value = version(__name__)
        except PackageNotFoundError:  # pragma: no cover
            # Package is not installed, fallback if needed
            value = "0.0.0"
        globals()["__version__"] = value
        return value
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return sorted([*globals(), "__version__"])
