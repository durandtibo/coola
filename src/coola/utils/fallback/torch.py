r"""Contain fallback implementations used when ``torch`` dependency is
not available."""

from __future__ import annotations

__all__ = ["cuda", "nn", "torch"]

from typing import TYPE_CHECKING, NoReturn

from coola.utils.fallback.factory import make_fake_class, make_fake_function, make_fake_module
from coola.utils.imports import raise_torch_missing_error

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

FakeClass: type = make_fake_class(raise_torch_missing_error)
fake_function: Callable[..., NoReturn] = make_fake_function(raise_torch_missing_error)

cuda: ModuleType = make_fake_module(
    "torch.cuda", is_available=fake_function, synchronize=fake_function
)

nn: ModuleType = make_fake_module(
    "torch.nn",
    utils=make_fake_module(
        "torch.nn.utils", rnn=make_fake_module("torch.nn.utils.rnn", PackedSequence=FakeClass)
    ),
)

# Create a fake torch package
torch: ModuleType = make_fake_module(
    "torch", cuda=cuda, nn=nn, Tensor=FakeClass, tensor=fake_function
)
