r"""Provide a Nano ID style random identifier.

Like ``generate_ulid`` and ``generate_uuid7``, a Nano ID is not derived
from data content and carries no timestamp: two calls always produce
different values, and results do not sort by creation time. Unlike those
two, it has no fixed shape: both the alphabet and the length are
configurable, which makes it a better fit than a 26-character ULID or a
36-character UUID for contexts where a short, URL-safe identifier is
wanted (e.g. a slug embedded in a URL).
"""

from __future__ import annotations

__all__ = ["generate_nano_id"]

import os

from coola.identifier.validation import validate_positive

# Default alphabet used by the reference Nano ID implementation: 64
# URL-safe characters (unreserved by RFC 3986), giving each character
# 6 bits of entropy.
_DEFAULT_ALPHABET = "_-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

_DEFAULT_LENGTH = 21

# Upper bound on `length`, and on the number of bytes requested from
# `os.urandom` in a single loop iteration below. Guards against a very
# large `length` (accidental or malicious, e.g. from unvalidated user
# input) driving a correspondingly large `os.urandom` allocation per
# iteration; far larger than any realistic identifier still fits.
_MAX_LENGTH = 1024


def generate_nano_id(length: int = _DEFAULT_LENGTH, alphabet: str = _DEFAULT_ALPHABET) -> str:
    r"""Generate a Nano ID style random identifier.

    Draws ``length`` characters from ``alphabet`` uniformly at random,
    using rejection sampling on ``os.urandom`` bytes so every character
    of ``alphabet`` has exactly equal probability (a plain
    ``byte % len(alphabet)`` would bias the result unless
    ``len(alphabet)`` is a power of two).

    Args:
        length: The number of characters to generate. Must be positive.
        alphabet: The set of characters to draw from. Must contain
            between 1 and 256 distinct characters. Defaults to a
            64-character URL-safe alphabet (digits, upper- and
            lowercase ASCII letters, ``-``, and ``_``), matching the
            reference Nano ID implementation's default.

    Returns:
        A random string of length ``length`` drawn from ``alphabet``.

    Raises:
        ValueError: If ``length`` is not positive or exceeds 1024, or
            ``alphabet`` is empty, has duplicate characters, or has
            more than 256 distinct characters.

    Example:
        ```pycon
        >>> from coola.identifier import generate_nano_id
        >>> nano_id = generate_nano_id()
        >>> len(nano_id)
        21
        >>> short_id = generate_nano_id(length=8, alphabet="0123456789abcdef")
        >>> len(short_id)
        8

        ```
    """
    validate_positive(length, name="length")
    if length > _MAX_LENGTH:
        msg = f"length must be at most {_MAX_LENGTH}, got {length}"
        raise ValueError(msg)
    n = len(alphabet)
    if n == 0:
        msg = "alphabet must not be empty"
        raise ValueError(msg)
    if len(set(alphabet)) != n:
        msg = f"alphabet must not contain duplicate characters, got {alphabet!r}"
        raise ValueError(msg)
    if n > 256:
        msg = f"alphabet must have at most 256 distinct characters, got {n}"
        raise ValueError(msg)

    # Smallest bitmask covering the alphabet's index range, used to
    # reject out-of-range random bytes so every character keeps an
    # equal selection probability regardless of len(alphabet).
    mask = (1 << (n - 1).bit_length()) - 1

    chars: list[str] = []
    while len(chars) < length:
        # Oversample: with the mask applied, roughly n / (mask + 1) of
        # the bytes are accepted, so pull extra bytes up front to
        # usually finish in a single os.urandom call.
        needed = length - len(chars)
        buffer = os.urandom(needed + needed // 4 + 16)
        for byte in buffer:
            index = byte & mask
            if index < n:
                chars.append(alphabet[index])
                if len(chars) == length:
                    break
    return "".join(chars)
