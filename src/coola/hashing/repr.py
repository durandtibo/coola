r"""Define the repr hasher."""

from __future__ import annotations

__all__ = ["ReprHasher"]

from enum import Enum
from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin, NoArgsDisplayMixin
from coola.hashing.base import BaseHasher
from coola.hashing.string import hash_enum, hash_tagged
from coola.utils.imports import is_numpy_available
from coola.utils.integer import int_repr

if TYPE_CHECKING or is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    from coola.utils.fallback.numpy import numpy as np

if TYPE_CHECKING:
    from coola.hashing.registry import HasherRegistry


def _numpy_scalar_to_builtin(data: Any) -> Any:
    """Convert a NumPy scalar to the equivalent builtin value.

    The ``repr`` of a NumPy scalar depends on the NumPy version (``1.0``
    with NumPy 1, ``np.float32(1.0)`` with NumPy 2), so the scalars are
    converted before hashing. The other values are returned unchanged.
    """
    if not is_numpy_available() or not isinstance(data, np.generic):
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


class ReprHasher(InlineDisplayMixin, NoArgsDisplayMixin, BaseHasher[Any]):
    r"""Hasher for objects whose ``repr()`` is a reliable canonical
    representation.

    This hasher converts the object to its ``repr()`` string and hashes
    it under a type tag. Instances of ``bool``, ``int``, ``float`` and
    ``complex`` (including subclasses) and the NumPy scalars use the
    builtin ``repr`` and the builtin type name as tag, so they hash like
    the equivalent builtin value (``np.float32(1.5)`` hashes like ``1.5``,
    ``np.int64(3)`` like ``3``). Negative zero is normalised to zero,
    since ``-0.0 == 0.0``. Enum members are hashed with
    ``hash_enum`` instead. It is preferable to
    ``StrHasher`` for numeric types (``int``, ``float``, ``complex``,
    ``bool``) because ``repr()`` guarantees round-trip accuracy for
    floating point values, whereas ``str()`` may lose precision on some
    platforms.

    Example:
        ```pycon
        >>> from coola.hashing import ReprHasher, HasherRegistry
        >>> registry = HasherRegistry()
        >>> hasher = ReprHasher()
        >>> hasher
        ReprHasher()
        >>> len(hasher.hash(1234, registry=registry))
        64

        ```
    """

    def hash(
        self,
        data: Any,
        registry: HasherRegistry,  # noqa: ARG002
        *,
        length: int = 64,
        ignore_unhashable: bool = False,  # noqa: ARG002
    ) -> str:
        r"""Compute a deterministic hash of ``repr(data)``.

        Args:
            data: The object to hash.
            registry: The hasher registry. Unused by this hasher since
                the ``repr()`` string is hashed directly with no need
                to dispatch to another hasher for nested data; accepted
                only to satisfy the common ``BaseHasher`` interface.
            length: The desired length of the returned hex string. See
                ``hash_string`` for constraints. Defaults to 64.
            ignore_unhashable: Unused by this hasher; accepted only to
                satisfy the common ``BaseHasher`` interface.

        Returns:
            A lowercase hexadecimal string of exactly ``length``
            characters.

        Raises:
            ValueError: If ``length`` is not an even number between 2
                and 128.
        """
        if isinstance(data, Enum):
            # ``(int, Enum)`` mixins reach this hasher through the MRO.
            return hash_enum(data, length=length)
        data = _numpy_scalar_to_builtin(data)
        for base in (bool, int, float, complex):
            if isinstance(data, base):
                # Use the builtin repr so subclasses and the NumPy scalars
                # hash like the builtin value, independent of library versions.
                return hash_tagged(base.__name__, _builtin_repr(base, data), length=length)
        cls = type(data)
        # The module keeps a user class named ``Decimal`` or ``Path`` out of
        # the tag namespace of the other hashers; the builtins keep the bare
        # name (``NoneType``).
        tag = (
            cls.__qualname__
            if cls.__module__ == "builtins"
            else f"{cls.__module__}.{cls.__qualname__}"
        )
        return hash_tagged(tag, repr(data), length=length)
