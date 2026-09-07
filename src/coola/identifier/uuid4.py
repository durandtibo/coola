r"""Provide a plain random UUID identifier.

This is the simplest generator in the package: a thin wrapper around
``uuid.uuid4`` (RFC 9562), provided for API symmetry with
``generate_uuid7`` and ``generate_stable_uuid5``. Unlike
``generate_uuid7``, the result carries no timestamp component and does
not sort by creation time; unlike ``generate_stable_uuid5``, it is not
derived from any data and is different on every call. Prefer
``generate_uuid7`` when creation-time ordering is useful, and prefer
``generate_ulid`` when the result does not need to be a valid UUID
string.
"""

from __future__ import annotations

__all__ = ["generate_uuid4"]

import uuid


def generate_uuid4() -> str:
    r"""Generate a random UUIDv4 (RFC 9562) identifier.

    Returns:
        A lowercase UUID string of the form
        ``'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'``.

    Example:
        ```pycon
        >>> from coola.identifier import generate_uuid4
        >>> uuid4 = generate_uuid4()
        >>> len(uuid4)
        36

        ```
    """
    return str(uuid.uuid4())
