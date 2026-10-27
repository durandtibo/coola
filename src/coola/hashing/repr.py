r"""Define the repr hasher."""

from __future__ import annotations

__all__ = ["ReprHasher"]

import sys
from typing import Any

from coola.display import InlineDisplayMixin
from coola.hashing.base import BaseHasher, Encoding, tagged
from coola.utils.integer import int_repr
from coola.utils.introspection import get_qualified_class_name


def _numpy_scalar_to_builtin(data: Any) -> Any:
    """Convert a NumPy scalar to the equivalent builtin value.

    The ``repr`` of a NumPy scalar depends on the NumPy version (``1.0``
    with NumPy 1, ``np.float32(1.0)`` with NumPy 2), so the scalars are
    converted before hashing. The other values are returned unchanged.
    """
    # A NumPy scalar cannot exist before NumPy is imported, so NumPy is not imported here.
    np = sys.modules.get("numpy")
    if np is None or not isinstance(data, np.generic):
        return data
    if isinstance(data, np.bool_):
        return bool(data)
    if isinstance(data, np.integer):
        return int(data)
    if isinstance(data, np.floating):
        return float(data)
    if isinstance(data, np.complexfloating):
        return complex(data)
    return data


def _builtin_repr(base: Any, data: Any) -> str:
    """Return the ``repr`` of ``data`` as the builtin type ``base``.

    ``-0.0`` is normalised to ``0.0`` (also for the components of a
    complex number) because the two values are equal.
    """
    if base is float:
        return float.__repr__(float(data) + 0.0)
    if base is complex:
        return complex.__repr__(complex(float(data.real) + 0.0, float(data.imag) + 0.0))
    if base is int:
        return int_repr(data)
    return base.__repr__(data)


class ReprHasher(InlineDisplayMixin, BaseHasher[Any]):
    r"""Hasher for objects whose ``repr()`` is a reliable canonical
    representation.

    This hasher converts the object to its ``repr()`` string and hashes
    it under a type tag. Instances of ``bool``, ``int``, ``float`` and
    ``complex`` (including subclasses) and the NumPy scalars use the
    builtin ``repr`` and the builtin type name as tag, so they hash like
    the equivalent builtin value (``np.float32(1.5)`` hashes like ``1.5``,
    ``np.int64(3)`` like ``3``). Negative zero is normalised to zero,
    since ``-0.0 == 0.0``. It is preferable to
    ``StrHasher`` for numeric types (``int``, ``float``, ``complex``,
    ``bool``) because ``repr()`` guarantees round-trip accuracy for
    floating point values, whereas ``str()`` may lose precision on some
    platforms.

    For the other types, the tag is the module and the qualified name of
    the class by default, so a user class never shares the tag of
    another class with the same name. Pass ``tag`` to register a
    constant tag for a type (``ReprHasher("timedelta")``): a subclass
    then hashes like its base.

    Args:
        tag: The type tag used for the values that are not a ``bool``,
            an ``int``, a ``float`` or a ``complex``. If ``None``, the
            tag is derived from the class of the hashed object.
            Defaults to ``None``.

    Example:
        ```pycon
        >>> from coola.hashing import ReprHasher
        >>> hasher = ReprHasher()
        >>> hasher
        ReprHasher()
        >>> hasher.encode(1234)
        b'int\x001234'

        ```
    """

    def __init__(self, tag: str | None = None) -> None:
        self._tag = tag

    def _display_fields(self) -> dict[str, Any]:
        return {} if self._tag is None else {"tag": self._tag}

    def encode(self, data: Any) -> Encoding:
        r"""Describe an object by its ``repr()``.

        Args:
            data: The object to describe.

        Returns:
            The leaf ``tagged(tag, repr(data))``. A ``bool``, an
            ``int``, a ``float`` or a ``complex`` (and the NumPy
            scalars) use the builtin type name as tag and the builtin
            ``repr``.
        """
        data = _numpy_scalar_to_builtin(data)
        for base in (bool, int, float, complex):
            if isinstance(data, base):
                # Use the builtin repr so subclasses and the NumPy scalars
                # hash like the builtin value, independent of library versions.
                return tagged(base.__name__, _builtin_repr(base, data))
        tag = self._tag or get_qualified_class_name(type(data))
        return tagged(tag, repr(data))
