__all__ = ["numpy_available", "torch_available"]

from pytest import mark

from coola import is_numpy_available, is_torch_available

numpy_available = mark.skipif(not is_numpy_available(), reason="Requires NumPy")
torch_available = mark.skipif(not is_torch_available(), reason="Requires PyTorch")
