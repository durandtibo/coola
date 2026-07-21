r"""Contain fallback implementations used when ``torch`` dependency is
not available."""

from __future__ import annotations

__all__ = ["cuda", "nn", "torch"]

from types import ModuleType

from coola.utils.fallback._factory import make_fake_class, make_fake_function
from coola.utils.imports import raise_torch_missing_error

FakeClass = make_fake_class(raise_torch_missing_error)
fake_function = make_fake_function(raise_torch_missing_error)

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
