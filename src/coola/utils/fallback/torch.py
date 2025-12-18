r"""Contain fallback implementations used when ``torch`` dependency is
not available."""

from __future__ import annotations

__all__ = ["torch"]

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


# Create a fake torch package
torch: ModuleType = ModuleType("torch")
torch.nn = ModuleType("torch.nn")
torch.nn.utils = ModuleType("torch.nn.utils")
torch.nn.utils.rnn = ModuleType("torch.nn.utils.rnn")
torch.nn.utils.rnn.PackedSequence = FakeClass
torch.Tensor = FakeClass
torch.tensor = fake_function
