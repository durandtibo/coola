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

import re
from typing import TYPE_CHECKING

from coola.identifier.ulid import generate_ulid
from coola.validation import validate_not_empty

if TYPE_CHECKING:
    from collections.abc import Callable

_PREFIX_PATTERN = re.compile(r"[A-Za-z0-9-]+")


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
        ValueError: If ``prefix`` is empty, contains ``"_"`` or a
            character other than an ASCII letter, a digit and ``"-"``,
            or if ``generator()`` returns an empty string.
        TypeError: If ``generator()`` does not return a ``str``.

    Example:
        ```pycon
        >>> from coola.identifier import generate_prefixed_id
        >>> generate_prefixed_id("cus")  # doctest: +ELLIPSIS
        'cus_...'
        >>> generate_prefixed_id("evt", generator=lambda: "123")
        'evt_123'

        ```
    """
    validate_not_empty(prefix, name="prefix")
    if "_" in prefix:
        msg = f"prefix must not contain '_', got {prefix!r}"
        raise ValueError(msg)
    if not _PREFIX_PATTERN.fullmatch(prefix):
        msg = f"prefix must only contain ASCII letters, digits and '-', got {prefix!r}"
        raise ValueError(msg)
    identifier = generator()
    if not isinstance(identifier, str):
        msg = f"generator() must return a str, got {type(identifier)!r}"
        raise TypeError(msg)
    if not identifier:
        msg = f"generator() must return a non-empty string, got {identifier!r}"
        raise ValueError(msg)
    return f"{prefix}_{identifier}"
