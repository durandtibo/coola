r"""Implement a random manager for PyTorch."""

from __future__ import annotations

__all__ = ["TorchRandomManager", "torch_seed"]

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

from coola.display import InlineDisplayMixin, NoArgsDisplayMixin
from coola.random.base import BaseRandomManager
from coola.utils.imports import check_torch, is_torch_available
from coola.utils.introspection import is_same_type

if TYPE_CHECKING:
    from collections.abc import Generator

if is_torch_available():
    import torch
else:  # pragma: no cover
    from coola.utils.fallback.torch import torch


class TorchRandomManager(InlineDisplayMixin, NoArgsDisplayMixin, BaseRandomManager):
    r"""Implement a random manager for the library ``torch``.

    Only the default generators of PyTorch (the CPU one and the CUDA
    ones) are managed. The generators created with ``torch.Generator()``
    are independent objects: they are not seeded, and their state is not
    saved nor restored.

    Example:
        ```pycon
        >>> from coola.random import TorchRandomManager
        >>> manager = TorchRandomManager()
        >>> manager.manual_seed(42)

        ```
    """

    def __init__(self) -> None:
        check_torch()

    def __eq__(self, other: object) -> bool:
        return is_same_type(other, self)

    def __hash__(self) -> int:
        return hash(type(self))

    def get_rng_state(self) -> dict[str, Any]:
        return {
            "torch": torch.get_rng_state(),
            "torch.cuda": torch.cuda.get_rng_state_all(),
        }

    def manual_seed(self, seed: int) -> None:
        cast("Any", torch).manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def set_rng_state(self, state: dict[str, Any]) -> None:
        torch.set_rng_state(state["torch"])
        torch.cuda.set_rng_state_all(state["torch.cuda"])


@contextmanager
def torch_seed(seed: int) -> Generator[None]:
    r"""Implement a context manager to manage the PyTorch random seed and
    random number generator (RNG) state.

    The context manager sets the specified random seed and
    restores the original RNG state afterward.

    Args:
        seed: The random number generator seed to use while using
            this context manager.

    Example:
        ```pycon
        >>> import torch
        >>> from coola.random import torch_seed
        >>> with torch_seed(42):
        ...     print(torch.randn(2, 4))
        ...
        tensor([[...]])
        >>> with torch_seed(42):
        ...     print(torch.randn(2, 4))
        ...
        tensor([[...]])

        ```
    """
    with TorchRandomManager().seed(seed):
        yield
