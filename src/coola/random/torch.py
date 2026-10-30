r"""Implement a random manager for PyTorch."""

from __future__ import annotations

__all__ = ["TorchRandomManager", "get_random_seed", "get_torch_generator"]

from typing import Any

from coola.display import InlineDisplayMixin
from coola.random.base import BaseRandomManager
from coola.random.seed import validate_seed
from coola.utils.imports import check_torch, is_torch_available
from coola.utils.introspection import is_same_type

if is_torch_available():
    import torch
else:  # pragma: no cover
    from coola.utils.fallback.torch import torch


class TorchRandomManager(InlineDisplayMixin, BaseRandomManager):
    r"""Implement a random manager for the library ``torch``.

    Only the default generators of PyTorch (the CPU one, the CUDA
    ones and the MPS one if it is available) are managed. The
    generators created with ``torch.Generator()`` are independent
    objects: they are not seeded, and their state is not saved nor
    restored. Use ``get_torch_generator`` to create a seeded generator.
    The CUDA state is saved only if CUDA is already initialized:
    reading it would initialize CUDA, which fails in a forked worker,
    and an uninitialized CUDA has no state to save.

    Raises:
        MissingPackageError: If ``torch`` is not installed.

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
        state = {"torch": torch.get_rng_state()}
        # ``get_rng_state_all`` initializes CUDA, which fails in a forked worker. A CUDA
        # that is not initialized has no generator state to save.
        if torch.cuda.is_initialized():
            state["torch.cuda"] = torch.cuda.get_rng_state_all()
        if torch.backends.mps.is_available():
            state["torch.mps"] = torch.mps.get_rng_state()
        return state

    def manual_seed(self, seed: int) -> None:
        r"""Set the seed of the PyTorch random number generator.

        Args:
            seed: The seed.

        Raises:
            TypeError: If the seed is not an ``int``.
            ValueError: If the seed is not in the range ``[0, 2**63)``.
        """
        validate_seed(seed)
        torch.manual_seed(seed)  # seeds the CPU, CUDA (lazily) and MPS generators

    def set_rng_state(self, state: dict[str, Any]) -> None:
        r"""Restore the state of the PyTorch random number generators.

        Args:
            state: The state returned by ``get_rng_state``. Only the
                generators in the state are restored.

        Raises:
            ValueError: If the number of CUDA generators of the state
                differs from the number of CUDA devices, or if the state
                has an MPS generator but MPS is not available.
        """
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
            if not torch.backends.mps.is_available():
                msg = "The state has a MPS generator, but MPS is not available"
                raise ValueError(msg)
            torch.mps.set_rng_state(state["torch.mps"])


def get_random_seed(seed: int) -> int:
    r"""Get a random seed derived from a given seed.

    The value is drawn from a ``torch.Generator`` seeded with ``seed``,
    so it is deterministic and does not change the state of the
    default generators.

    Args:
        seed: A random seed to make the process reproducible.

    Returns:
        A random seed in ``[0, 2**63 - 1)``. It is a valid seed for
            ``manual_seed``.

    Raises:
        TypeError: If the seed is not an ``int``.
        ValueError: If the seed is not in the range ``[0, 2**63)``.

    Example:
        ```pycon
        >>> from coola.random import get_random_seed
        >>> get_random_seed(44)
        6176747449835261348

        ```
    """
    return torch.randint(0, 2**63 - 1, size=(1,), generator=get_torch_generator(seed)).item()


def get_torch_generator(seed: int, *, device: torch.device | str | None = "cpu") -> torch.Generator:
    r"""Create a ``torch.Generator`` initialized with a given seed.

    The generator is independent from the default generators of
    PyTorch: it is the way to get reproducible random numbers in a
    thread, where ``random_seed`` must not be used.

    Args:
        seed: A random seed.
        device: The desired device for the generator.

    Returns:
        A ``torch.Generator`` object.

    Raises:
        TypeError: If the seed is not an ``int``.
        ValueError: If the seed is not in the range ``[0, 2**63)``.

    Example:
        ```pycon
        >>> import torch
        >>> from coola.random import get_torch_generator
        >>> generator = get_torch_generator(42)
        >>> torch.rand(2, 4, generator=generator)
        tensor([[0.8823, 0.9150, 0.3829, 0.9593],
                [0.3904, 0.6009, 0.2566, 0.7936]])
        >>> generator = get_torch_generator(42)
        >>> torch.rand(2, 4, generator=generator)
        tensor([[0.8823, 0.9150, 0.3829, 0.9593],
                [0.3904, 0.6009, 0.2566, 0.7936]])

        ```
    """
    validate_seed(seed)
    generator = torch.Generator(device)
    generator.manual_seed(seed)
    return generator
