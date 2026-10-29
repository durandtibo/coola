r"""Implement the rounding of the PyTorch tensors."""

from __future__ import annotations

__all__ = ["round_torch"]

from typing import Any

from coola.rounding.common import MAX_EXACT, RoundMode, get_scale
from coola.utils.imports import is_torch_available

if is_torch_available():
    import torch
else:  # pragma: no cover
    from coola.utils.fallback.torch import torch


def round_torch(data: Any, mode: RoundMode, decimals: int) -> Any:
    r"""Round a PyTorch tensor.

    Integer and boolean values are not changed. The computation is made
    in ``float64`` (``float32`` on the ``mps`` device) and the result
    has the dtype and the device of ``data``.

    Args:
        data: The ``torch.Tensor`` to round.
        mode: The rounding mode.
        decimals: The number of decimals.

    Returns:
        The rounded tensor. It never shares memory with ``data``.

    Raises:
        TypeError: If ``data`` is not a tensor or if its dtype is
            complex.

    Example:
        ```pycon
        >>> import torch
        >>> from coola.rounding.torch import round_torch
        >>> round_torch(torch.tensor([1.234, 5.678]), "floor", 1)
        tensor([1.2000, 5.6000])

        ```
    """
    if not isinstance(data, torch.Tensor):
        msg = f"Unsupported type: {type(data).__qualname__}"
        raise TypeError(msg)
    if data.dtype.is_complex:
        msg = f"Unsupported dtype: {data.dtype}"
        raise TypeError(msg)
    scale = get_scale(decimals)
    if not data.dtype.is_floating_point or scale is None:
        return data.detach().clone()
    return _round_float_tensor(data, mode, scale)


def _round_float_tensor(data: torch.Tensor, mode: RoundMode, scale: float) -> torch.Tensor:
    # The computation is made in float64 (float32 on mps). The grid points are compared with the
    # value after a cast to the dtype of the tensor, as the value is.
    x = data.detach().to(torch.float32 if data.device.type == "mps" else torch.float64)
    low = data.dtype != x.dtype

    def grid(index: torch.Tensor) -> torch.Tensor:
        point = index / scale
        return point.to(data.dtype).to(x.dtype) if low else point

    scaled = x * scale
    index = torch.floor(scaled)
    # ``scaled`` is rounded, so the index is wrong by at most one in either direction.
    index = index + (grid(index + 1) <= x).to(x.dtype)
    index = index - (grid(index) > x).to(x.dtype)
    if mode == "ceil":
        index = index + (grid(index) < x).to(x.dtype)
    elif mode == "round":
        middle = (2 * index + 1) / (2 * scale)
        if low:
            middle = middle.to(data.dtype).to(x.dtype)
        # ``torch.round`` sends a halfway case to the even index.
        index = torch.where(
            x > middle, index + 1, torch.where(x == middle, torch.round(index + 0.5), index)
        )
    # ``NaN``, the infinite values and the values past the grid are not changed.
    result = torch.where(scaled.abs() < MAX_EXACT, index / scale, x)
    return torch.copysign(result, x).to(data.dtype)
