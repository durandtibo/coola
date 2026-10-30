r"""Define the hashers of the scalar values: numbers, text, bytes and
enum members."""

from __future__ import annotations

__all__ = ["BytesHasher", "EnumHasher", "ReprHasher", "StrHasher", "StringHasher"]

import sys
from collections import UserString
from enum import Enum
from typing import Any

from coola.display import InlineDisplayMixin
from coola.hashing.base import BaseHasher, Encoding, digest, tagged
from coola.utils.integer import int_repr
from coola.utils.introspection import get_qualified_class_name


class StringHasher(InlineDisplayMixin, BaseHasher[str | UserString]):
    r"""Hasher for string objects.

    This hasher describes the string under the ``str`` type tag, without
    any intermediate conversion. ``str`` subclasses hash like plain
    strings (the enum members are dispatched to ``EnumHasher`` by the
    registry). A ``UserString`` is described by its text under the
    ``UserString`` tag, so it differs from a ``str``.

    Example:
        ```pycon
        >>> from coola.hashing import StringHasher
        >>> hasher = StringHasher()
        >>> hasher
        StringHasher()
        >>> hasher.encode("Meowwwwww")
        b'str\x00Meowwwwww'

        ```
    """

    def encode(self, data: str | UserString) -> Encoding:
        r"""Describe a string.

        A lone surrogate (for example in the result of ``os.fsdecode``
        for a file name that is not valid UTF-8) is encoded with the
        ``surrogatepass`` error handler.

        Args:
            data: The string to describe.

        Returns:
            The leaf ``tagged("str", data)``, or
            ``tagged("UserString", str(data))`` for a ``UserString``.
        """
        if isinstance(data, UserString):
            return tagged("UserString", str(data))
        return tagged("str", data)


class BytesHasher(InlineDisplayMixin, BaseHasher[bytes | bytearray | memoryview]):
    r"""Hasher for ``bytes``, ``bytearray`` and ``memoryview`` objects.

    This hasher computes the hash of the bytes namespaced by the type
    tag (``bytes``, ``bytearray`` or ``memoryview``), so a ``bytes``, a
    ``bytearray`` and a ``memoryview`` with the same content hash
    differently.

    Example:
        ```pycon
        >>> from coola.hashing import BytesHasher
        >>> hasher = BytesHasher()
        >>> hasher
        BytesHasher()
        >>> len(hasher.encode(b"hello"))
        70

        ```
    """

    def encode(self, data: bytes | bytearray | memoryview) -> Encoding:
        r"""Describe a ``bytes``, a ``bytearray`` or a ``memoryview``.

        The content is reduced with ``digest`` so that the encoding is
        small, whatever the size of the buffer.

        Args:
            data: The bytes to describe.

        Returns:
            The leaf ``tagged(tag, digest(content))``.
        """
        if isinstance(data, bytearray):
            tag = "bytearray"
        elif isinstance(data, memoryview):
            tag = "memoryview"
        else:
            tag = "bytes"
        return tagged(tag, digest(bytes(data)))


class EnumHasher(InlineDisplayMixin, BaseHasher[Enum]):
    r"""Hasher for ``enum.Enum`` members.

    The hash depends on the qualified class name and the member name,
    not on the member value. ``IntEnum`` and ``StrEnum`` members are
    therefore not collapsed into ``int`` and ``str``.

    Example:
        ```pycon
        >>> from enum import Enum
        >>> from coola.hashing import EnumHasher
        >>> class Color(Enum):
        ...     RED = 1
        ...
        >>> hasher = EnumHasher()
        >>> hasher
        EnumHasher()
        >>> hasher.encode(Color.RED).startswith(b"Enum")
        True

        ```
    """

    def encode(self, data: Enum) -> Encoding:
        r"""Describe an enum member.

        The encoding depends on the qualified name of the enum class and
        on the member name, not on the member value, so it does not
        collapse ``IntEnum``/``StrEnum`` members into ``int``/``str``.
        A combined flag has no name: its value is used instead.

        Args:
            data: The enum member to describe.

        Returns:
            The leaf ``tagged("Enum", "module.Class.NAME")``.
        """
        cls = type(data)
        name = data.name if data.name is not None else repr(data.value)
        return tagged("Enum", f"{cls.__module__}.{cls.__qualname__}.{name}")


def _numpy_scalar_to_builtin(data: Any) -> Any:
    r"""Convert a NumPy scalar to the equivalent builtin value.

    The ``repr`` of a NumPy scalar depends on the NumPy version (``1.0``
    with NumPy 1, ``np.float32(1.0)`` with NumPy 2), so the scalars are
    converted before hashing. The other values are returned unchanged.

    Args:
        data: The value.

    Returns:
        The builtin value if ``data`` is a NumPy scalar, otherwise
            ``data``.
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
    r"""Return the ``repr`` of ``data`` as the builtin type ``base``.

    ``-0.0`` is normalised to ``0.0`` (also for the components of a
    complex number) because the two values are equal.

    Args:
        base: The builtin type: ``float``, ``complex``, ``int`` or
            another.
        data: The value.

    Returns:
        The ``repr`` of ``data`` as ``base``.
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


class StrHasher(InlineDisplayMixin, BaseHasher[Any]):
    r"""Hasher for objects whose ``str()`` is a reliable canonical
    representation.

    This hasher converts the object to its ``str()`` string and then
    computes the hash of that string. Unlike ``ReprHasher``, which uses
    ``repr()``, this hasher uses ``str()``, which is more human-readable
    but does not guarantee round-trip accuracy for floating point values.

    The tag identifies the type family. By default it is ``str:`` followed
    by the module and the qualified name of the class, so a user class
    never shares the tag of another class with the same name. Pass
    ``tag`` to register a constant tag for a type (``StrHasher("UUID")``):
    a subclass then hashes like its base.

    Args:
        tag: The type tag mixed into the hash. If ``None``, the tag is
            derived from the class of the hashed object. Defaults to
            ``None``.

    Example:
        ```pycon
        >>> from coola.hashing import StrHasher
        >>> hasher = StrHasher()
        >>> hasher
        StrHasher()
        >>> hasher.encode("hello")
        b'str:str\x00hello'

        ```
    """

    def __init__(self, tag: str | None = None) -> None:
        self._tag = tag

    def _display_fields(self) -> dict[str, Any]:
        return {} if self._tag is None else {"tag": self._tag}

    def encode(self, data: Any) -> Encoding:
        r"""Describe an object by its ``str()``.

        Args:
            data: The object to describe.

        Returns:
            The leaf ``tagged(tag, str(data))``.
        """
        tag = self._tag or f"str:{get_qualified_class_name(type(data))}"
        return tagged(tag, str(data))
