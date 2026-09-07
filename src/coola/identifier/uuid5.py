r"""Provide a deterministic UUID identifier for nested data.

This is deliberately kept separate from ``coola.hashing``:
``hash_object`` and friends produce a hex digest of a chosen ``length``
for content hashing (dedup, caching, comparisons), whereas
``generate_stable_uuid5`` produces a valid UUID string for use as a
stable identifier (e.g. a record ID or a database primary key). The two
solve different problems and are not meant to be interchangeable, hence
the separate module.
"""

from __future__ import annotations

__all__ = ["generate_stable_uuid5"]

import uuid
from typing import TYPE_CHECKING

from coola.hashing import hash_object

if TYPE_CHECKING:
    from coola.hashing import HasherRegistry

# Project-specific namespace for deterministic UUIDs. Generated once with
# uuid.uuid4() and fixed here so identifiers are stable across runs and
# reproducible across environments.
_NAMESPACE = uuid.UUID("2b6f6a52-6e83-4b1b-9e3a-0a2e9b2d9c6a")


def generate_stable_uuid5(
    data: object,
    registry: HasherRegistry | None = None,
    namespace: uuid.UUID = _NAMESPACE,
    ignore_unhashable: bool = False,
) -> str:
    r"""Compute a stable, reproducible UUID for a nested data structure.

    Hashes ``data`` via ``hash_object`` (at its maximum, 128-hex-digit
    length, so the identifier gets the full benefit of the underlying
    hash's collision resistance) to guarantee a consistent digest
    regardless of e.g. mapping insertion order, then derives a
    deterministic UUID from that digest using ``uuid.uuid5`` under a
    fixed namespace.

    Note:
        ``uuid.uuid5`` always returns a 128-bit value regardless of the
        strength of the digest fed into it, so
        ``generate_stable_uuid5(a) == generate_stable_uuid5(b)`` if and
        only if ``hash_object(a) == hash_object(b)`` (modulo the
        astronomically unlikely case of a ``uuid.uuid5`` collision on
        two different digests). Note also that ``uuid.uuid5`` hashes
        its input with SHA-1 internally, so the final UUID's collision
        resistance is bounded by SHA-1 regardless of how strong
        ``hash_object``'s own digest is.

    Warning:
        The value returned by ``generate_stable_uuid5`` for a given
        ``data`` is stable only as long as ``hash_object`` (and the
        hashers
        resolved by ``registry`` for the types in ``data``) keep
        producing the same digest for that ``data``. A change to the
        default registry's hashing algorithms in a future ``coola``
        release would silently change the UUIDs produced here. Do not
        rely on cross-version stability for UUIDs persisted long-term
        (e.g. as database primary keys) unless you pin ``coola`` and
        pass an explicit, version-controlled ``registry``.

    Args:
        data: The data to derive a UUID from. Can be a nested structure
            such as a ``list``, ``dict``, or ``tuple``.
        registry: The registry used to resolve hashers for each data
            type, forwarded to ``hash_object``. If ``None``, the
            default registry is used.
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
    return str(
        uuid.uuid5(
            namespace,
            hash_object(data, registry=registry, length=128, ignore_unhashable=ignore_unhashable),
        )
    )
