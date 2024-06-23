r"""Implement a formatter for ``numpy.ndarray``.

The formatter is registered only if ``numpy`` is available.
"""

from __future__ import annotations

__all__ = ["NDArrayFormatter"]

from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

from coola.formatters.base import BaseFormatter
from coola.utils import check_numpy, is_numpy_available

if TYPE_CHECKING:
    from coola.summarizers.base import BaseSummarizer

if is_numpy_available():
    import numpy as np
else:
    np = Mock()  # pragma: no cover


class NDArrayFormatter(BaseFormatter[np.ndarray]):
    r"""Implement a formatter for ``numpy.ndarray``.

    Args:
        show_data: If ``True``, the returned string is the default
            string representation (``repr``). If ``False``,
            the returned string only contains the tensor metadata.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from coola import Summarizer
    >>> from coola.formatters import NDArrayFormatter
    >>> formatter = NDArrayFormatter()
    >>> formatter.format(Summarizer(), np.arange(21))
    <class 'numpy.ndarray'> | shape=(21,) | dtype=int64

    ```
    """

    def __init__(self, show_data: bool = False) -> None:
        check_numpy()
        self._show_data = bool(show_data)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(show_data={self._show_data})"

    def clone(self) -> NDArrayFormatter:
        return self.__class__(show_data=self._show_data)

    def equal(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._show_data == other._show_data

    def format(
        self,
        summarizer: BaseSummarizer,  # noqa: ARG002
        value: np.ndarray,
        depth: int = 0,  # noqa: ARG002
        max_depth: int = 1,  # noqa: ARG002
    ) -> str:
        if self._show_data:
            return repr(value)
        return " | ".join(
            [
                f"{type(value)}",
                f"shape={value.shape}",
                f"dtype={value.dtype}",
            ]
        )

    def load_state_dict(self, state: dict) -> None:
        self._show_data = state["show_data"]

    def state_dict(self) -> dict:
        return {"show_data": self._show_data}

    def get_show_data(self) -> bool:
        r"""Indicate if the array data or metadata are shown.

        Returns:
            ``True`` if the array data are shown,
                ``False`` if the array metadata are shown.

        Example usage:

        ```pycon
        >>> from coola.formatters import NDArrayFormatter
        >>> formatter = NDArrayFormatter()
        >>> formatter.get_show_data()
        False

        ```
        """
        return self._show_data

    def set_show_data(self, show_data: bool) -> None:
        r"""Set if the array data or metadata are shown.

        Args:
            show_data: ``True`` if the array data are shown,
                ``False`` if the array metadata are shown.

        Raises:
            TypeError: if ``show_data`` is not a boolean.

        Example usage:

        ```pycon
        >>> from coola.formatters import NDArrayFormatter
        >>> formatter = NDArrayFormatter()
        >>> formatter.set_show_data(True)
        >>> formatter.get_show_data()
        True

        ```
        """
        if not isinstance(show_data, bool):
            msg = f"Incorrect type for show_data. Expected bool value but received {show_data}"
            raise TypeError(msg)
        self._show_data = show_data
