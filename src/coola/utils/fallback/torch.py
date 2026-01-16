r"""Contain fallback implementations used when ``torch`` dependency is
not available."""

from __future__ import annotations

__all__ = ["cuda", "nn", "torch"]

from types import ModuleType
from typing import Any, NoReturn

from coola.utils.imports import raise_error_torch_missing


class FakeClass:
    r"""Fake class that raises an error because torch is not installed.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Raises:
        RuntimeError: torch is required for this functionality.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        raise_error_torch_missing()


def fake_function(*args: Any, **kwargs: Any) -> NoReturn:  # noqa: ARG001
    r"""Fake function that raises an error because torch is not
    installed.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Raises:
        RuntimeError: torch is required for this functionality.
    """
    raise_error_torch_missing()


cuda: ModuleType = ModuleType("torch.cuda")
cuda.is_available = fake_function
cuda.synchronize = fake_function

nn: ModuleType = ModuleType("torch.nn")
nn.utils = ModuleType("torch.nn.utils")
nn.utils.rnn = ModuleType("torch.nn.utils.rnn")
nn.utils.rnn.PackedSequence = FakeClass

# Create a fake torch package
torch: ModuleType = ModuleType("torch")
torch.cuda = cuda
torch.nn = nn

torch.Tensor = FakeClass
torch.tensor = fake_function
