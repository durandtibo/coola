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
    values = data.detach().to(torch.float32 if data.device.type == "mps" else torch.float64)
    scaled = values * scale
    if mode == "round":
        result = torch.round(scaled) / scale
    elif mode == "floor":
        index = torch.floor(scaled)
        # ``0.29 * 100`` is below 29: take the grid point if it is not above the value. The grid
        # point is compared in the dtype of the tensor, as the value is.
        index = torch.where(_grid(index + 1, scale, data.dtype) <= values, index + 1, index)
        result = index / scale
    else:
        index = torch.ceil(scaled)
        index = torch.where(_grid(index - 1, scale, data.dtype) >= values, index - 1, index)
        result = index / scale
    result = torch.where(scaled.abs() < MAX_EXACT, result, values)
    return result.to(data.dtype)


def _grid(index: torch.Tensor, scale: float, dtype: torch.dtype) -> torch.Tensor:
    return (index / scale).to(dtype).to(index.dtype)
