r"""Implement the validation of random seeds."""

from __future__ import annotations

__all__ = ["validate_seed"]

MAX_SEED = 2**63


def validate_seed(seed: int) -> None:
    r"""Validate a random seed.

    The seed must be an ``int`` in the range ``[0, 2**63)``, which is
    supported by all the backends (``random``, NumPy and PyTorch).

    Args:
        seed: The seed to validate.

    Raises:
        TypeError: If the seed is not an ``int`` (``bool`` is
            rejected too).
        ValueError: If the seed is out of range.

    Example:
        ```pycon
        >>> from coola.random import validate_seed
        >>> validate_seed(42)

        ```
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        msg = f"seed must be an int, got {type(seed)}"
        raise TypeError(msg)
    if not 0 <= seed < MAX_SEED:
        msg = f"seed must be in the range [0, 2**63), got {seed}"
        raise ValueError(msg)
