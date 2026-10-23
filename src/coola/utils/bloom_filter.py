r"""Contain the implementation of a Bloom filter."""

from __future__ import annotations

__all__ = ["BloomFilter"]

import hashlib
import math
from typing import TYPE_CHECKING

from coola.validation import validate_gt

if TYPE_CHECKING:
    from collections.abc import Generator


class BloomFilter:
    r"""Fixed-memory approximate set-membership structure.

    A Bloom filter answers "have I probably seen this item before?"
    using a fixed amount of memory. It guarantees no false negatives
    (an item that was added is always reported as present), at the
    cost of a tunable false-positive rate (an item that was never
    added may occasionally be reported as present). Memory usage is
    fixed up front based on the expected number of items and the
    desired false-positive rate, regardless of how many items are
    actually added. It is a standalone utility, typically used for
    approximate duplicate detection over large streams of items.

    Warning:
        The filter is **not thread-safe**: concurrent ``add`` or
        ``add_and_check`` calls can lose bits. Protect it with a lock if
        it is shared between threads.

    Example:
        ```pycon
        >>> from coola.utils.bloom_filter import BloomFilter
        >>> bf = BloomFilter(expected_items=1000, fp_rate=0.01)
        >>> bf.add_and_check(b"hello")
        False
        >>> bf.add_and_check(b"hello")
        True
        >>> b"hello" in bf
        True
        >>> b"world" in bf
        False

        ```
    """

    def __init__(self, expected_items: int = 1_000_000, fp_rate: float = 0.01) -> None:
        r"""Initialize the filter's bit array and hash count.

        Args:
            expected_items: Approximate number of unique items the
                filter is expected to hold. Used to size the bit array
                for the requested false-positive rate; overestimating
                is safe and simply uses more memory, while
                underestimating raises the effective false-positive
                rate as more items are added than planned for.
            fp_rate: Target false-positive probability once
                approximately ``expected_items`` unique items have been
                added, expressed as a value in ``(0, 1)``.

        Raises:
            ValueError: if ``expected_items`` is not positive or
                ``fp_rate`` is not in ``(0, 1)``.
        """
        validate_gt(expected_items, 0, name="expected_items")
        if not 0 < fp_rate < 1:
            msg = f"fp_rate must be in (0, 1), but received {fp_rate}"
            raise ValueError(msg)
        self.size: int = self._optimal_size(expected_items, fp_rate)
        self.hash_count: int = self._optimal_hash_count(self.size, expected_items)
        self.bits: bytearray = bytearray(self.size // 8 + 1)

    @staticmethod
    def _optimal_size(n: int, p: float) -> int:
        r"""Compute the bit-array size minimizing memory for the given
        expected item count ``n`` and target false-positive rate
        ``p``."""
        return max(8, int(-(n * math.log(p)) / (math.log(2) ** 2)))

    # Upper bound on the number of hash rounds per item, regardless of what
    # the ``expected_items``/``fp_rate`` combination would otherwise compute.
    # Without this cap, a very small ``expected_items`` combined with a
    # ``fp_rate`` close to 0 drives the theoretically-optimal hash count
    # arbitrarily high (it grows as ``-log(fp_rate)``), making
    # ``add_and_check`` unboundedly slow for a configuration that is
    # already well past the point of diminishing returns.
    _MAX_HASH_COUNT = 32

    @classmethod
    def _optimal_hash_count(cls, m: int, n: int) -> int:
        r"""Compute the number of hash functions minimizing the false-
        positive rate for a bit array of size ``m`` and expected item
        count ``n``, capped at ``_MAX_HASH_COUNT``."""
        return max(1, min(cls._MAX_HASH_COUNT, int((m / max(n, 1)) * math.log(2))))

    def _hashes(self, item: bytes) -> Generator[int]:
        """Yield ``self.hash_count`` bit indices for ``item``.

        Uses double hashing (two independent hash values combined
        linearly) to cheaply derive many hash functions from two, per
        the standard Kirsch-Mitzenmacher technique. Both values are
        derived from a single BLAKE2b digest, split into two halves,
        which avoids a second hash computation while keeping the two
        values statistically independent. BLAKE2b is used instead of
        SHA-512 because it is faster in CPython's stdlib while being
        just as suitable here (this is not a cryptographic use case);
        the digest is also sized to exactly the 32 bytes this method
        needs, rather than truncating a 64-byte SHA-512 digest.

        Args:
            item: The raw bytes to hash.

        Yields:
            Bit-array indices, in ``range(self.size)``.
        """
        digest = hashlib.blake2b(item, digest_size=32).digest()
        h1 = int.from_bytes(digest[:16], "big")
        h2 = int.from_bytes(digest[16:], "big")
        # Ensure h2 is never 0, or every index collapses to h1 for that item.
        h2 |= 1

        for i in range(self.hash_count):
            yield (h1 + i * h2) % self.size

    def add_and_check(self, item: bytes) -> bool:
        r"""Add an item to the filter and report whether it was probably
        already present.

        Args:
            item: The raw bytes to add/check.

        Returns:
            ``True`` if ``item`` was probably already in the filter
            before this call (i.e. it is probably a duplicate, subject
            to the filter's false-positive rate). ``False`` if it was
            definitely not present before this call. Either way, the
            item's bits are set, so subsequent calls with the same
            item will return ``True``.
        """
        already_present = True
        for idx in self._hashes(item):
            byte_idx, bit_idx = idx // 8, idx % 8
            if not (self.bits[byte_idx] & (1 << bit_idx)):
                already_present = False
                self.bits[byte_idx] |= 1 << bit_idx
        return already_present

    def add(self, item: bytes) -> None:
        r"""Add an item to the filter.

        Args:
            item: The raw bytes to add.

        Example:
            ```pycon
            >>> from coola.utils.bloom_filter import BloomFilter
            >>> bf = BloomFilter(expected_items=1000)
            >>> bf.add(b"hello")
            >>> b"hello" in bf
            True

            ```
        """
        self.add_and_check(item)

    def __contains__(self, item: object) -> bool:
        r"""Indicate if an item is probably in the filter, without adding
        it.

        Args:
            item: The raw bytes to check.

        Returns:
            ``False`` if ``item`` is definitely not in the filter,
            ``True`` if it probably is (subject to the false-positive
            rate).

        Example:
            ```pycon
            >>> from coola.utils.bloom_filter import BloomFilter
            >>> bf = BloomFilter(expected_items=1000)
            >>> b"hello" in bf
            False

            ```
        """
        if not isinstance(item, bytes):
            return False
        return all(self.bits[idx // 8] & (1 << (idx % 8)) for idx in self._hashes(item))

    def fill_ratio(self) -> float:
        r"""Return the fraction of the bits that are set.

        A ratio close to 1 means that the false-positive rate is much
        higher than the requested one.

        Returns:
            The fraction of set bits, in ``[0, 1]``.

        Example:
            ```pycon
            >>> from coola.utils.bloom_filter import BloomFilter
            >>> bf = BloomFilter(expected_items=1000)
            >>> bf.fill_ratio()
            0.0

            ```
        """
        return sum(byte.bit_count() for byte in self.bits) / self.size
