r"""Contain functions to easily manage randomness."""

from __future__ import annotations

__all__ = ["get_rng_state", "manual_seed", "set_rng_state"]

from typing import Any

from coola.random.default import RandomManager

_rng_manager = RandomManager()


def get_rng_state() -> dict[str, Any]:
    r"""Get the current RNG state.

    Returns:
        The current RNG state.

    Example usage:

    ```pycon

    >>> from coola.random import get_rng_state
    >>> state = get_rng_state()
    >>> state
    {'random': ...}

    ```
    """
    return _rng_manager.get_rng_state()


def manual_seed(seed: int) -> None:
    r"""Set the seed for generating random numbers.

    Args:
        seed: The desired random seed.

    Example usage:

    ```pycon

    >>> import torch
    >>> from coola.random import manual_seed
    >>> manual_seed(42)
    >>> torch.randn(3)
    tensor([...])
    >>> torch.randn(3)
    tensor([...])
    >>> manual_seed(42)
    >>> torch.randn(3)
    tensor([...])

    ```
    """
    _rng_manager.manual_seed(seed)


def set_rng_state(state: dict[str, Any]) -> None:
    r"""Set the RNG state.

    Args:
        state: The new RNG state.

    Example usage:

    ```pycon

    >>> import torch
    >>> from coola.random import get_rng_state, set_rng_state
    >>> st = get_rng_state()
    >>> set_rng_state(st)

    ```
    """
    _rng_manager.set_rng_state(state)
