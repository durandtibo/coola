r"""Contain helpers shared across the hashing unit tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from coola.hashing import BaseHasher, Node, digest, get_default_registry

if TYPE_CHECKING:
    from coola.hashing import Encoding, Hasher, HasherRegistry

__all__ = ["FixedHasher", "Unhashable", "encode_leaf", "encode_node", "hash_with", "leaf_hash"]


class Unhashable:
    r"""A type with no registered hasher."""


def hash_with(
    hasher: Hasher[Any],
    data: object,
    registry: HasherRegistry | None = None,
    *,
    length: int = 64,
    ignore_unhashable: bool = False,
) -> str:
    r"""Hash ``data`` with ``hasher`` for its type.

    The hasher is registered for ``type(data)`` in a copy of
    ``registry`` (the default registry if ``None``), so the engine finds
    it for the root and the registry finds the hashers of the children.
    """
    registry = (get_default_registry() if registry is None else registry).copy()
    registry.register(type(data), hasher, exist_ok=True)
    return registry.hash(data, length=length, ignore_unhashable=ignore_unhashable)


def leaf_hash(encoding: bytes, length: int = 64) -> str:
    r"""Return the hash of a leaf: the hex digest of its encoding."""
    return digest(b"\x00" + encoding).hex()[:length]


def encode_leaf(hasher: Hasher[Any], data: object) -> bytes:
    r"""Return the encoding of ``data``, which must be a leaf.

    Raises:
        AssertionError: If the hasher describes ``data`` as a node.
    """
    encoding = hasher.encode(data)
    assert isinstance(encoding, bytes), f"{hasher!r} encodes {data!r} as a node"
    return encoding


def encode_node(hasher: Hasher[Any], data: object) -> Node:
    r"""Return the encoding of ``data``, which must be a node.

    Raises:
        AssertionError: If the hasher describes ``data`` as a leaf.
    """
    encoding = hasher.encode(data)
    assert isinstance(encoding, Node), f"{hasher!r} encodes {data!r} as a leaf"
    return encoding


class FixedHasher(BaseHasher[Any]):
    r"""A hasher that returns the same encoding for every value."""

    def __init__(self, encoding: Encoding) -> None:
        self._encoding = encoding

    def encode(self, data: Any) -> Encoding:  # noqa: ARG002
        return self._encoding
