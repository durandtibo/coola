r"""Top-level package for ``coola``.

``coola`` provides functionality to compare nested objects, hash and
summarize complex structures, and work with helper utilities for
recursive transformations and iteration.

The main entry points are available from the top-level package, for
example ``coola.objects_are_equal``, ``coola.hash_object`` and
``coola.summarize``. They are imported the first time they are used, so
``import coola`` stays cheap. The other functions and classes are
documented under their subpackage (``coola.equality``,
``coola.hashing``, ``coola.summary``, ...).

Example:
    ```pycon
    >>> import coola
    >>> coola.objects_are_equal({"a": [1, 2]}, {"a": [1, 2]})
    True
    >>> coola.objects_are_allclose([1.0], [1.0 + 1e-9])
    True

    ```
"""

__all__ = [
    "__version__",
    "assert_objects_allclose",
    "assert_objects_equal",
    "compare",
    "hash_object",
    "objects_are_allclose",
    "objects_are_equal",
    "summarize",
]

import importlib
from typing import Any

# The module that defines each entry point. The module is imported the first time the name is
# used, so ``import coola`` does not import the subpackages.
_LAZY_EXPORTS = {
    "assert_objects_allclose": "coola.equality",
    "assert_objects_equal": "coola.equality",
    "compare": "coola.equality",
    "hash_object": "coola.hashing",
    "objects_are_allclose": "coola.equality",
    "objects_are_equal": "coola.equality",
    "summarize": "coola.summary",
}


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
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is not None:
        value = getattr(importlib.import_module(module_name), name)
        globals()[name] = value
        return value
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return sorted({*globals(), "__version__", *_LAZY_EXPORTS})
