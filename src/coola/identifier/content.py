r"""Provide a content-addressed identifier for nested data.

This is an alternative to ``generate_stable_uuid``: instead of wrapping
the digest in a ``uuid.uuid5`` value (which is bounded by SHA-1's
128-bit output regardless of the strength of the underlying hash),
``generate_content_id`` returns the ``hash_object`` digest directly.
This keeps the full collision resistance and configurable length of the
underlying hash, at the cost of not being a valid UUID string (so it
cannot fill a UUID-typed database column, for example).
"""

from __future__ import annotations

__all__ = ["generate_content_id"]

from typing import TYPE_CHECKING

from coola.hashing import hash_object

if TYPE_CHECKING:
    from coola.hashing import HasherRegistry


def generate_content_id(
    data: object,
    registry: HasherRegistry | None = None,
    length: int = 64,
    ignore_unhashable: bool = False,
) -> str:
    r"""Compute a content-addressed identifier for a nested data
    structure.

    Unlike ``generate_stable_uuid``, the returned identifier is the raw
    ``hash_object`` digest: it is not reshaped into a UUID, so its
    collision resistance and length are exactly those of the
    underlying hash rather than being bounded by ``uuid.uuid5``'s
    SHA-1 pass.

    Args:
        data: The data to derive an identifier from. Can be a nested
            structure such as a ``list``, ``dict``, or ``tuple``.
        registry: The registry used to resolve hashers for each data
            type, forwarded to ``hash_object``. If ``None``, the
            default registry is used.
        length: The desired length of the returned hex string,
            forwarded to ``hash_object``. Must be an even number
            between 2 and 128 inclusive. Defaults to ``64``.
        ignore_unhashable: Forwarded to ``hash_object``. If ``True``,
            objects for which no hasher is registered are replaced by a
            deterministic placeholder hash instead of raising an error.
            Defaults to ``False``, which raises a ``KeyError`` when an
            unhashable object is encountered.

    Returns:
        A lowercase hex string identifier of the requested ``length``.

    Raises:
        KeyError: If ``data`` (or a nested object within it) has a type
            for which no hasher is registered and ``ignore_unhashable``
            is ``False``.

    Example:
        ```pycon
        >>> from coola.identifier import generate_content_id
        >>> generate_content_id({"source": "cats.txt", "page": 1})  # doctest: +ELLIPSIS
        '...'
        >>> generate_content_id({"page": 1, "source": "cats.txt"}) == generate_content_id(
        ...     {"source": "cats.txt", "page": 1}
        ... )
        True

        ```
    """
    return hash_object(data, registry=registry, length=length, ignore_unhashable=ignore_unhashable)
