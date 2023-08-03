from __future__ import annotations

__all__ = ["Tensor", "ndarray"]


from coola.utils.imports import is_numpy_available, is_torch_available

if is_numpy_available():
    from numpy import ndarray
else:  # pragma: no cover

    class ndarray:  # noqa: N801
        r"""Defines a fake class so the code still works if numpy is not
        installed."""


if is_torch_available():
    from torch import Tensor
else:  # pragma: no cover

    class Tensor:
        r"""Defines a fake class so the code still works if torch is not
        installed."""
