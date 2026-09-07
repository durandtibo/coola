r"""Provide a helper to prefix identifiers with a type tag.

This is not a new identifier-generation algorithm: it wraps any of the
other generators in this package (or a custom callable) to produce
Stripe-style prefixed identifiers such as ``"cus_01J...` or
``"evt_018f..."``, which make an identifier's type recognizable at a
glance (e.g. in logs, URLs, or support tickets) without needing a
lookup.
"""

from __future__ import annotations

__all__ = ["generate_prefixed_id"]

from typing import TYPE_CHECKING

from coola.identifier.ulid import generate_ulid

if TYPE_CHECKING:
    from collections.abc import Callable


def generate_prefixed_id(prefix: str, generator: Callable[[], str] = generate_ulid) -> str:
    r"""Generate an identifier prefixed with a type tag.

    Args:
        prefix: The prefix identifying the type of object the
            identifier belongs to, e.g. ``"cus"`` for a customer or
            ``"evt"`` for an event. Must not contain the ``"_"``
            separator.
        generator: A zero-argument callable that returns the
            identifier to prefix. Defaults to ``generate_ulid``. Pass
            e.g. ``generate_stable_uuid5`` partially applied to a fixed
            ``data`` argument (via ``functools.partial``) to prefix a
            content-derived identifier instead.

    Returns:
        ``f"{prefix}_{generator()}"``.

    Raises:
        ValueError: If ``prefix`` is empty or contains ``"_"``.

    Example:
        ```pycon
        >>> from coola.identifier import generate_prefixed_id
        >>> generate_prefixed_id("cus")  # doctest: +ELLIPSIS
        'cus_...'
        >>> generate_prefixed_id("evt", generator=lambda: "123")
        'evt_123'

        ```
    """
    if not prefix:
        msg = "prefix must not be empty"
        raise ValueError(msg)
    if "_" in prefix:
        msg = f"prefix must not contain '_', got {prefix!r}"
        raise ValueError(msg)
    return f"{prefix}_{generator()}"
