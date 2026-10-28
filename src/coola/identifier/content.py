r"""Provide the content-derived identifiers of nested data.

Both identifiers are a function of the data, through
``coola.hashing.hash_object``: equal data (whatever the insertion order
of the mappings) give the same identifier.
``generate_stable_content_id`` returns the digest directly, so it keeps
the full strength and the configurable length of the underlying hash.
``generate_stable_uuid5`` wraps a digest in a valid UUID string (for
example for a UUID-typed database column), whose 122 random bits are the
limit of its collision resistance (birthday bound 2^61), whatever the
hash behind it.
"""

from __future__ import annotations

__all__ = ["generate_stable_content_id", "generate_stable_uuid5"]

import uuid
from typing import TYPE_CHECKING

from coola.hashing import get_stable_registry, hash_object

if TYPE_CHECKING:
    from coola.hashing import HasherRegistry

# Project-specific namespace for deterministic UUIDs. Generated once with
# uuid.uuid4() and fixed here so identifiers are stable across runs and
# reproducible across environments.
_NAMESPACE = uuid.UUID("2b6f6a52-6e83-4b1b-9e3a-0a2e9b2d9c6a")


def generate_stable_content_id(
    data: object,
    *,
    registry: HasherRegistry | None = None,
    length: int = 64,
    ignore_unhashable: bool = False,
) -> str:
    r"""Compute a content-addressed identifier for a nested data
    structure.

    Unlike ``generate_stable_uuid5``, the returned identifier is the raw
    ``hash_object`` digest: it is not reshaped into a UUID, so its
    collision resistance and length are exactly those of the
    underlying hash. Use it when more than the 122 bits of a UUID are
    needed.

    Args:
        data: The data to derive an identifier from. Can be a nested
            structure such as a ``list``, ``dict``, or ``tuple``.
        registry: The registry used to resolve hashers for each data
            type, forwarded to ``hash_object``. If ``None``, the
            stable registry (``get_stable_registry``) is used: it has
            only the built-in hashers, so the identifier does not
            depend on the installed plugins or on ``register_hashers``.
            Passing another registry opts out of this stability.
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
        >>> from coola.identifier import generate_stable_content_id
        >>> generate_stable_content_id({"source": "cats.txt", "page": 1})  # doctest: +ELLIPSIS
        '...'
        >>> generate_stable_content_id(
        ...     {"page": 1, "source": "cats.txt"}
        ... ) == generate_stable_content_id({"source": "cats.txt", "page": 1})
        True

        ```
    """
    if registry is None:
        registry = get_stable_registry()
    return hash_object(data, registry=registry, length=length, ignore_unhashable=ignore_unhashable)


def generate_stable_uuid5(
    data: object,
    *,
    registry: HasherRegistry | None = None,
    namespace: uuid.UUID = _NAMESPACE,
    ignore_unhashable: bool = False,
) -> str:
    r"""Compute a stable, reproducible UUID for a nested data structure.

    Hashes ``data`` via ``generate_stable_content_id`` (at its maximum,
    128-hex-digit length) to guarantee a consistent digest regardless of
    e.g. mapping insertion order, then derives a deterministic UUID from
    that digest using ``uuid.uuid5`` under a fixed namespace.

    Note:
        ``generate_stable_uuid5(a) == generate_stable_uuid5(b)`` if and
        only if ``hash_object(a, length=128) == hash_object(b,
        length=128)`` (modulo the astronomically unlikely case of a
        ``uuid.uuid5`` collision on two different digests). A UUID has
        122 random bits, so its collision resistance is bounded by its
        width (birthday bound 2^61), not by SHA-1 or by the strength of
        the digest: use ``generate_stable_content_id`` when more bits
        are needed.

    Warning:
        The value returned by ``generate_stable_uuid5`` for a given
        ``data`` is stable only as long as ``hash_object`` (and the
        hashers resolved by ``registry`` for the types in ``data``) keep
        producing the same digest for that ``data``. A change to the
        hashing algorithms of ``coola`` in a future release would
        silently change the UUIDs produced here. Do not rely on
        cross-version stability for UUIDs persisted long-term (e.g. as
        database primary keys) unless you pin ``coola`` and pass an
        explicit, version-controlled ``registry``.

    Args:
        data: The data to derive a UUID from. Can be a nested structure
            such as a ``list``, ``dict``, or ``tuple``.
        registry: The registry used to resolve hashers for each data
            type, forwarded to ``hash_object``. If ``None``, the
            stable registry (``get_stable_registry``) is used: it has
            only the built-in hashers, so the identifier does not
            depend on the installed plugins or on ``register_hashers``.
            Passing another registry opts out of this stability.
        namespace: The UUID namespace passed to ``uuid.uuid5``. Defaults
            to a namespace fixed for this module; pass a different one
            to derive UUIDs in a separate identifier space (e.g. to
            avoid collisions with UUIDs minted by another system for
            unrelated data).
        ignore_unhashable: Forwarded to ``hash_object``. If ``True``,
            objects for which no hasher is registered are replaced by a
            deterministic placeholder hash instead of raising an error.
            Defaults to ``False``, which raises a ``KeyError`` when an
            unhashable object is encountered.

    Returns:
        A lowercase UUID string of the form
        ``'xxxxxxxx-xxxx-5xxx-xxxx-xxxxxxxxxxxx'``.

    Raises:
        KeyError: If ``data`` (or a nested object within it) has a type
            for which no hasher is registered and ``ignore_unhashable``
            is ``False``.

    Example:
        ```pycon
        >>> from coola.identifier import generate_stable_uuid5
        >>> generate_stable_uuid5({"source": "cats.txt", "page": 1})  # doctest: +ELLIPSIS
        '...'
        >>> generate_stable_uuid5({"page": 1, "source": "cats.txt"}) == generate_stable_uuid5(
        ...     {"source": "cats.txt", "page": 1}
        ... )
        True

        ```
    """
    digest = generate_stable_content_id(
        data, registry=registry, length=128, ignore_unhashable=ignore_unhashable
    )
    return str(uuid.uuid5(namespace, digest))
