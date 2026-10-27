r"""Define the engine that computes the digest of a nested object.

The engine owns the invariants of the hashes: the dispatch by type (once
per type and per call), the composition of the digests, the order rule,
the cycles, the shared containers, the depth and the placeholder of the
unhashable types. The hashers only describe the content of a type (see
``BaseHasher``). The state (stack, cache, memo) is local to the call, so
threads and re-entrant calls are independent.
"""

from __future__ import annotations

__all__ = ["digest_of", "hash_hex", "validate_hash_length"]

import warnings
from hashlib import blake2b
from typing import TYPE_CHECKING, Any

from coola.hashing.base import PERSON, Node, digest, tagged
from coola.registry import TypeNotRegisteredError
from coola.utils.introspection import get_qualified_class_name

if TYPE_CHECKING:
    from coola.hashing.base import Hasher
    from coola.hashing.registry import HasherRegistry

_LEAF = b"\x00"
_NODE = b"\x01"


def validate_hash_length(length: int) -> None:
    r"""Check that ``length`` is a valid length for a hex digest.

    Args:
        length: The length of the hex digest to validate.

    Raises:
        ValueError: If ``length`` is not an even ``int`` between 2 and
            128.

    Example:
        ```pycon
        >>> from coola.hashing.engine import validate_hash_length
        >>> validate_hash_length(64)

        ```
    """
    # The first test is the fast path; a ``bool`` is an ``int`` but is not a valid length.
    is_int = type(length) is int or (isinstance(length, int) and not isinstance(length, bool))
    if not is_int or length % 2 or not 2 <= length <= 128:
        msg = f"length must be an even number between 2 and 128, got {length!r}."
        raise ValueError(msg)


def hash_hex(
    data: object,
    registry: HasherRegistry,
    *,
    length: int = 64,
    ignore_unhashable: bool = False,
    stacklevel: int = 1,
) -> str:
    r"""Return the first ``length`` characters of the hex digest of
    ``data``.

    Args:
        data: The object to hash.
        registry: The registry that finds the hasher of each type.
        length: The number of characters of the output: an even
            ``int`` between 2 and 128.
        ignore_unhashable: If ``True``, the objects with no hasher are
            replaced by a placeholder that only depends on the type.
        stacklevel: The frame blamed by the warning of a placeholder,
            counted from the caller of this function (``1`` is the
            caller).

    Returns:
        A lowercase hex string of ``length`` characters.

    Raises:
        ValueError: If ``length`` is not valid, or if a container
            contains itself.
        TypeNotRegisteredError: If a type has no hasher and
            ``ignore_unhashable`` is ``False``.

    Example:
        ```pycon
        >>> from coola.hashing import get_default_registry
        >>> from coola.hashing.engine import hash_hex
        >>> len(hash_hex([1, 2], get_default_registry(), length=16))
        16

        ```
    """
    validate_hash_length(length)
    return digest_of(
        data, registry, ignore_unhashable=ignore_unhashable, stacklevel=stacklevel + 1
    ).hex()[:length]


def _resolve(
    registry: HasherRegistry, data_type: type, ignore_unhashable: bool, stacklevel: int
) -> Hasher[Any] | None:
    r"""Find the hasher of a type, or ``None`` for the placeholder."""
    try:
        return registry.find_hasher(data_type)
    except KeyError as exc:
        if ignore_unhashable:
            warnings.warn(
                f"No hasher registered for {data_type!r}: using a type-only placeholder, "
                "so all objects of this type hash the same.",
                UserWarning,
                stacklevel=stacklevel,
            )
            return None
        if isinstance(exc, TypeNotRegisteredError):
            msg = (
                f"{exc}\nTo hash this type, register a hasher "
                f"(`register_hashers({{{data_type.__name__}: ...}})`), "
                "give the class a `__coola_hash__` method (see `SupportsCoolaHash`), "
                "or pass `ignore_unhashable=True` to hash a placeholder that only "
                "depends on the type."
            )
            raise TypeNotRegisteredError(msg) from exc
        raise


class _Walker:
    r"""The state of one call: the hasher of each type, the digests of
    the containers (or ``None`` while they are in progress), and the
    options."""

    def __init__(self, registry: HasherRegistry, ignore_unhashable: bool, warn_level: int) -> None:
        self._registry = registry
        self._ignore_unhashable = ignore_unhashable
        self._warn_level = warn_level
        self._cache: dict[type, Hasher[Any] | None] = {}  # type -> hasher (None: placeholder)
        # id -> (object, digest). The object is kept alive so that its id cannot be reused by a
        # temporary child during the call.
        self._memo: dict[int, tuple[object, bytes | None]] = {}
        self._tags: dict[str, bytes] = {}

    def enter(self, obj: object) -> _Frame | bytes:
        r"""Return a frame for a container to walk, and the digest
        otherwise."""
        data_type = type(obj)
        if data_type not in self._cache:
            self._cache[data_type] = _resolve(
                self._registry, data_type, self._ignore_unhashable, self._warn_level
            )
        hasher = self._cache[data_type]
        if hasher is None:
            return digest(_LEAF + tagged("unhashable", get_qualified_class_name(data_type)))
        return self.encode(obj, hasher)

    def encode(self, obj: object, hasher: Hasher[Any]) -> _Frame | bytes:
        r"""Encode an object with its hasher: return a frame for a
        container to walk, and the digest otherwise."""
        encoding = hasher.encode(obj)
        if isinstance(encoding, bytes):
            return blake2b(_LEAF + encoding, digest_size=64, person=PERSON).digest()
        return self.container(obj, encoding)

    def container(self, obj: object, node: Node) -> _Frame | bytes:
        r"""Return a frame for a container, or its digest if it was
        already hashed in this call (a shared container)."""
        seen = self._memo.get(id(obj))
        if seen is not None:
            if seen[1] is None:
                msg = f"Cannot hash a self-referencing {type(obj)!r}."
                raise ValueError(msg)
            return seen[1]
        self._memo[id(obj)] = (obj, None)
        return _Frame(obj, node)

    def fill(self, frame: _Frame) -> _Frame | None:
        r"""Collect the digests of the children of a frame, and return
        the frame of the first child that is a container to walk (the
        frame is resumed afterwards), or ``None`` when all the children
        are done."""
        cache = self._cache
        append = frame.digests.append
        for child in frame.children:  # resumes where it stopped
            hasher = cache.get(type(child))
            if hasher is None:  # first time (or placeholder) for this type
                entered = self.enter(child)
            else:
                encoding = hasher.encode(child)
                if isinstance(encoding, bytes):  # a leaf, the common case
                    append(blake2b(_LEAF + encoding, digest_size=64, person=PERSON).digest())
                    continue
                entered = self.container(child, encoding)
            if isinstance(entered, bytes):
                append(entered)
            else:
                return entered
        return None

    def leave(self, frame: _Frame) -> bytes:
        r"""Compute and remember the digest of a container whose
        children are done."""
        node = frame.node
        digests = frame.digests
        if not node.ordered:
            group = node.group
            if group == 1:
                digests.sort()
            else:
                digests = sorted(
                    b"".join(digests[i : i + group]) for i in range(0, len(digests), group)
                )
        result = blake2b(
            _NODE + self._tag(node.tag) + b"".join(digests), digest_size=64, person=PERSON
        ).digest()
        self._memo[id(frame.obj)] = (frame.obj, result)
        return result

    def _tag(self, tag: str) -> bytes:
        r"""Return the encoded tag of a node followed by a NUL byte."""
        encoded = self._tags.get(tag)
        if encoded is None:
            encoded = self._tags[tag] = tag.encode(errors="surrogatepass") + b"\0"
        return encoded


class _Frame:
    r"""A container being walked: its children are visited one after the
    other and their digests are collected."""

    __slots__ = ("children", "digests", "node", "obj")

    def __init__(self, obj: object, node: Node) -> None:
        self.obj = obj
        self.node = node
        self.children = iter(node.children)
        self.digests: list[bytes] = []


def digest_of(
    data: object, registry: HasherRegistry, *, ignore_unhashable: bool = False, stacklevel: int = 1
) -> bytes:
    r"""Compute the 64-byte digest of ``data``.

    The nesting is walked with an explicit stack, so the depth is
    bounded by the memory only. A container that is reached by several
    paths is hashed once, and a container that contains itself raises a
    ``ValueError``.

    Args:
        data: The object to hash.
        registry: The registry that finds the hasher of each type.
        ignore_unhashable: If ``True``, the objects with no hasher are
            replaced by a placeholder that only depends on the type, and
            a ``UserWarning`` is emitted once per type.
        stacklevel: The frame blamed by the warning of a placeholder,
            counted from the caller of this function (``1`` is the
            caller).

    Returns:
        The digest of ``data``.

    Raises:
        ValueError: If a container contains itself.
        TypeNotRegisteredError: If a type has no hasher and
            ``ignore_unhashable`` is ``False``.

    Example:
        ```pycon
        >>> from coola.hashing import get_default_registry
        >>> from coola.hashing.engine import digest_of
        >>> len(digest_of([1, 2], get_default_registry()))
        64

        ```
    """
    # _resolve -> _Walker.enter -> _Walker.fill -> digest_of -> caller
    walker = _Walker(registry, ignore_unhashable, stacklevel + 4)
    # The root is the only child of a frame that is never finished, so every object is entered
    # from ``fill`` and the warning of a placeholder has the same depth for all of them.
    stack = [_Frame(None, Node("", [data]))]
    while True:
        child = walker.fill(stack[-1])
        if child is not None:
            stack.append(child)
            continue
        frame = stack.pop()
        if not stack:
            return frame.digests[0]
        stack[-1].digests.append(walker.leave(frame))
