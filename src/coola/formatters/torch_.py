from __future__ import annotations

from typing import Any
from unittest.mock import Mock

from coola.formatters.base import BaseFormatter
from coola.summarizers.base import BaseSummarizer
from coola.utils import check_torch, is_torch_available

if is_torch_available():
    import torch
else:
    torch = Mock()  # pragma: no cover


class TensorFormatter(BaseFormatter[torch.Tensor]):
    r"""Implement a formatter for ``torch.Tensor``.

    Args:
    ----
        show_data (bool, optional): If ``True``, the returned string
            is the default string representation (``repr``).
            If ``False``, the returned string only contains the tensor
            metadata. Default: ``False``
    """

    def __init__(self, show_data: bool = False) -> None:
        check_torch()
        self._show_data = bool(show_data)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(show_data={self._show_data})"

    def clone(self) -> TensorFormatter:
        return self.__class__(show_data=self._show_data)

    def equal(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._show_data == other._show_data

    def format(
        self, summarizer: BaseSummarizer, value: torch.Tensor, depth: int = 0, max_depth: int = 1
    ) -> str:
        if self._show_data:
            return repr(value)
        return " | ".join(
            [
                f"{type(value)}",
                f"shape={value.shape}",
                f"dtype={value.dtype}",
                f"device={value.device}",
            ]
        )

    def load_state_dict(self, state: dict) -> None:
        self._show_data = state["show_data"]

    def state_dict(self) -> dict:
        return {"show_data": self._show_data}

    def get_show_data(self) -> bool:
        r"""Indicates if the tensor data or metadata are shown.

        Returns:
        -------
            bool: ``True`` if the tensor data are shown,
                ``False`` if the tensor metadata are shown.

        Example usage:

        .. code-block:: pycon

            >>> from coola.formatters import TensorFormatter
            >>> formatter = TensorFormatter()
            >>> formatter.get_show_data()
            False
        """
        return self._show_data

    def set_show_data(self, show_data: bool) -> None:
        r"""Set if the tensor data or metadata are shown.

        Args:
        ----
            show_data (bool): ``True`` if the tensor data are shown,
                ``False`` if the tensor metadata are shown.

        Raises:
        ------
            TypeError if ``show_data`` is not an boolean.

        Example usage:

        .. code-block:: pycon

            >>> from coola.formatters import TensorFormatter
            >>> formatter = TensorFormatter()
            >>> formatter.set_show_data(True)
            >>> formatter.get_show_data()
            True
        """
        if not isinstance(show_data, bool):
            raise TypeError(
                "Incorrect type for show_data. Expected bool value but received {show_data}"
            )
        self._show_data = show_data
