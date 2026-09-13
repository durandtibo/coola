r"""Contain the implementation of a Bloom filter."""

from __future__ import annotations

__all__ = ["BloomFilter"]

import hashlib
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator


class BloomFilter:
    r"""Fixed-memory approximate set-membership structure.

    Used here for approximate exact-duplicate detection over
    document content: guarantees no false negatives (a document that
    truly has appeared before will always be flagged as a duplicate),
    at the cost of a tunable false-positive rate (a document may
    occasionally be flagged as a duplicate when it is not). Memory
    usage is fixed up front based on the expected number of items and
    desired false-positive rate, regardless of how many items are
    actually added.
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
        if expected_items < 1:
            msg = f"expected_items must be greater than 0, but received {expected_items}"
            raise ValueError(msg)
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

    def _hashes(self, item: bytes) -> Generator[int, None, None]:
        """Yield ``self.hash_count`` bit indices for ``item``.

        Uses double hashing (two independent hash values combined
        linearly) to cheaply derive many hash functions from two, per
        the standard Kirsch-Mitzenmacher technique. Both values are
        derived from a single SHA-512 digest, split into two halves,
        which avoids a second hash computation while keeping the two
        values statistically independent.

        Args:
            item: The raw bytes to hash.

        Yields:
            Bit-array indices, in ``range(self.size)``.
        """
        digest = hashlib.sha512(item).digest()
        h1 = int.from_bytes(digest[:32], "big")
        h2 = int.from_bytes(digest[32:], "big")
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
