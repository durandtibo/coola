r"""Implement a random manager for PyTorch."""

from __future__ import annotations

__all__ = ["TorchRandomManager", "torch_seed"]

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

from coola.display import InlineDisplayMixin
from coola.random.base import BaseRandomManager
from coola.utils.imports import check_torch, is_torch_available
from coola.utils.introspection import is_same_type

if TYPE_CHECKING:
    from collections.abc import Generator

if is_torch_available():
    import torch
else:  # pragma: no cover
    from coola.utils.fallback.torch import torch


class TorchRandomManager(InlineDisplayMixin, BaseRandomManager):
    r"""Implement a random manager for the library ``torch``.

    Only the default generators of PyTorch (the CPU one, the CUDA
    ones and the MPS one if it is available) are managed. The generators created with ``torch.Generator()``
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
        state = {
            "torch": torch.get_rng_state(),
            "torch.cuda": torch.cuda.get_rng_state_all(),
        }
        if _is_mps_available():
            state["torch.mps"] = torch.mps.get_rng_state()
        return state

    def manual_seed(self, seed: int) -> None:
        cast("Any", torch).manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if _is_mps_available():
            torch.mps.manual_seed(seed)

    def set_rng_state(self, state: dict[str, Any]) -> None:
        # Only the generators in ``state`` are restored.
        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if "torch.cuda" in state:
            num_devices = torch.cuda.device_count()
            if len(state["torch.cuda"]) != num_devices:
                msg = (
                    f"The state has {len(state['torch.cuda'])} CUDA generators, "
                    f"but {num_devices} CUDA devices are available"
                )
                raise ValueError(msg)
            torch.cuda.set_rng_state_all(state["torch.cuda"])
        if "torch.mps" in state:
            if not _is_mps_available():
                msg = "The state has a MPS generator, but MPS is not available"
                raise ValueError(msg)
            torch.mps.set_rng_state(state["torch.mps"])


def _is_mps_available() -> bool:
    r"""Indicate if the MPS backend is available."""
    return torch.backends.mps.is_available()


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
