r"""Implement the default summarizer."""

from __future__ import annotations

__all__ = ["BaseSummarizer", "Summarizer", "summarizer_options"]

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Any, ClassVar

from coola.formatters import (
    BaseFormatter,
    DefaultFormatter,
    MappingFormatter,
    NDArrayFormatter,
    SequenceFormatter,
    SetFormatter,
    TensorFormatter,
)
from coola.summarizers.base import BaseSummarizer
from coola.types import Tensor, ndarray
from coola.utils import is_numpy_available, is_torch_available
from coola.utils.format import str_indent, str_mapping


class Summarizer(BaseSummarizer):
    """Implement the default summarizer.

    The registry is a class variable, so it is shared with all the
    instances of this class.

    Example usage:

    ```pycon

    >>> from coola import Summarizer
    >>> summarizer = Summarizer()
    >>> summarizer
    Summarizer(
      (<class 'collections.abc.Mapping'>): MappingFormatter(max_items=5, num_spaces=2)
      (<class 'collections.abc.Sequence'>): SequenceFormatter(max_items=5, num_spaces=2)
      (<class 'dict'>): MappingFormatter(max_items=5, num_spaces=2)
      (<class 'list'>): SequenceFormatter(max_items=5, num_spaces=2)
      (<class 'object'>): DefaultFormatter(max_characters=-1)
      (<class 'set'>): SetFormatter(max_items=5, num_spaces=2)
      (<class 'tuple'>): SequenceFormatter(max_items=5, num_spaces=2)
      (<class 'numpy.ndarray'>): NDArrayFormatter(show_data=False)
      (<class 'torch.Tensor'>): TensorFormatter(show_data=False)
    )
    >>> print(summarizer.summary(1))
    <class 'int'> 1
    >>> print(summarizer.summary(["abc", "def"]))
    <class 'list'> (length=2)
      (0): abc
      (1): def
    >>> print(summarizer.summary([[0, 1, 2], {"key1": "abc", "key2": "def"}]))
    <class 'list'> (length=2)
      (0): [0, 1, 2]
      (1): {'key1': 'abc', 'key2': 'def'}
    >>> print(summarizer.summary([[0, 1, 2], {"key1": "abc", "key2": "def"}], max_depth=2))
    <class 'list'> (length=2)
      (0): <class 'list'> (length=3)
          (0): 0
          (1): 1
          (2): 2
      (1): <class 'dict'> (length=2)
          (key1): abc
          (key2): def

    ```
    """

    registry: ClassVar[dict[type[object], BaseFormatter]] = {
        Mapping: MappingFormatter(),
        Sequence: SequenceFormatter(),
        dict: MappingFormatter(),
        list: SequenceFormatter(),
        object: DefaultFormatter(),
        set: SetFormatter(),
        tuple: SequenceFormatter(),
    }

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_mapping(self.registry))}\n)"

    def summary(
        self,
        value: Any,
        depth: int = 0,
        max_depth: int = 1,
    ) -> str:
        return self.find_formatter(type(value)).format(
            summarizer=self,
            value=value,
            depth=depth,
            max_depth=max_depth,
        )

    @classmethod
    def add_formatter(
        cls, data_type: type[object], formatter: BaseFormatter, exist_ok: bool = False
    ) -> None:
        r"""Add a formatter for a given data type.

        Args:
            data_type: The data type for this test.
            formatter: The formatter to use for the
                specified type.
            exist_ok: If ``False``, ``RuntimeError`` is raised if the
                data type already exists. This parameter should be set
                to ``True`` to overwrite the formatter for a type.

        Raises:
            RuntimeError: if a formatter is already registered for the
                data type and ``exist_ok=False``.

        Example usage:

        ```pycon

        >>> from coola import Summarizer
        >>> from coola.formatters import MappingFormatter
        >>> Summarizer.add_formatter(dict, MappingFormatter(), exist_ok=True)

        ```
        """
        if data_type in cls.registry and not exist_ok:
            msg = (
                f"A formatter ({cls.registry[data_type]}) is already registered for the data "
                f"type {data_type}. Please use `exist_ok=True` if you want to overwrite the "
                "formatter for this type"
            )
            raise RuntimeError(msg)
        cls.registry[data_type] = formatter

    @classmethod
    def has_formatter(cls, data_type: type[object]) -> bool:
        r"""Indicate if a formatter is registered for the given data
        type.

        Args:
            data_type: The data type to check.

        Returns:
            ``True`` if a formatter is registered,
                otherwise ``False``.

        Example usage:

        ```pycon

        >>> from coola import Summarizer
        >>> Summarizer.has_formatter(list)
        True
        >>> Summarizer.has_formatter(str)
        False

        ```
        """
        return data_type in cls.registry

    @classmethod
    def find_formatter(cls, data_type: Any) -> BaseFormatter:
        r"""Find the formatter associated to an object.

        Args:
            data_type: The data type to get.

        Returns:
            The formatter associated to the data type.

        Raises:
            TypeError: if a formatter cannot be found for this data
                type.

        Example usage:

        ```pycon

        >>> from coola import Summarizer
        >>> Summarizer.find_formatter(list)
        SequenceFormatter(max_items=5, num_spaces=2)
        >>> Summarizer.find_formatter(str)
        DefaultFormatter(max_characters=-1)

        ```
        """
        for object_type in data_type.__mro__:
            formatter = cls.registry.get(object_type, None)
            if formatter is not None:
                return formatter
        msg = f"Incorrect data type: {data_type}"
        raise TypeError(msg)

    @classmethod
    def load_state_dict(cls, state: dict) -> None:
        r"""Load the state values from a dict.

        Args:
            state: A dictionary with state values.

        Example usage:

        ```pycon

        >>> from coola import Summarizer
        >>> Summarizer.load_state_dict({object: {"max_characters": 10}})
        >>> summarizer = Summarizer()
        >>> summarizer.registry[object]
        DefaultFormatter(max_characters=10)
        >>> Summarizer.load_state_dict({object: {"max_characters": -1}})
        >>> summarizer.registry[object]
        DefaultFormatter(max_characters=-1)

        ```
        """
        for data_type, formatter in cls.registry.items():
            if (s := state.get(data_type)) is not None:
                formatter.load_state_dict(s)

    @classmethod
    def state_dict(cls) -> dict:
        r"""Return a dictionary containing state values.

        Returns:
            The state values in a dict.

        Example usage:

        ```pycon

        >>> from coola import Summarizer
        >>> Summarizer.state_dict()
        {<class 'collections.abc.Mapping'>: {'max_items': 5, 'num_spaces': 2},...}

        ```
        """
        return {data_type: formatter.state_dict() for data_type, formatter in cls.registry.items()}

    @classmethod
    def set_max_characters(cls, max_characters: int) -> None:
        r"""Set the maximum of characters for the compatible formatter to
        the specified value.

        To be updated, the formatters need to implement the method
        ``set_max_characters``.

        Args:
            max_characters: The maximum of characters.

        Example usage:

        ```pycon

        >>> from coola import Summarizer
        >>> Summarizer.set_max_characters(10)
        >>> summarizer = Summarizer()
        >>> summarizer.registry[object]
        DefaultFormatter(max_characters=10)
        >>> Summarizer.set_max_characters(-1)
        >>> summarizer.registry[object]
        DefaultFormatter(max_characters=-1)

        ```
        """
        for formatter in cls.registry.values():
            if hasattr(formatter, "set_max_characters"):
                formatter.set_max_characters(max_characters)

    @classmethod
    def set_max_items(cls, max_items: int) -> None:
        r"""Set the maximum number of items for the compatible formatter
        to the specified value.

        To be updated, the formatters need to implement the method
        ``set_max_items``.

        Args:
            max_items: The maximum number of items to show.

        Example usage:

        ```pycon

        >>> from coola import Summarizer
        >>> Summarizer.set_max_items(10)
        >>> summarizer = Summarizer()
        >>> summarizer.registry[dict]
        MappingFormatter(max_items=10, num_spaces=2)
        >>> Summarizer.set_max_items(5)
        >>> summarizer.registry[dict]
        MappingFormatter(max_items=5, num_spaces=2)

        ```
        """
        for formatter in cls.registry.values():
            if hasattr(formatter, "set_max_items"):
                formatter.set_max_items(max_items)

    @classmethod
    def set_num_spaces(cls, num_spaces: int) -> None:
        r"""Set the maximum of items for the compatible formatter to the
        specified value.

        To be updated, the formatters need to implement the method
        ``set_num_spaces``.

        Args:
            num_spaces: The number of spaces for indentation.

        Example usage:

        ```pycon

        >>> from coola import Summarizer
        >>> Summarizer.set_num_spaces(4)
        >>> summarizer = Summarizer()
        >>> summarizer.registry[dict]
        MappingFormatter(max_items=5, num_spaces=4)
        >>> Summarizer.set_num_spaces(2)
        >>> summarizer.registry[dict]
        MappingFormatter(max_items=5, num_spaces=2)

        ```
        """
        for formatter in cls.registry.values():
            if hasattr(formatter, "set_num_spaces"):
                formatter.set_num_spaces(num_spaces)


def set_summarizer_options(
    max_characters: int | None = None, max_items: int | None = None, num_spaces: int | None = None
) -> None:
    r"""Set the ``Summarizer`` options.

    Note: It is recommended to use ``summarizer_options`` rather than
    this function.

    Args:
        max_characters: The maximum number of characters
            to show. If ``None``, the maximum number of characters
            is unchanged.
        max_items: The maximum number of items to show.
            If ``None``, the maximum number of items is unchanged.
        num_spaces: The number of spaces for indentation.
            If ``None``, the number of  spaces for indentation is
            unchanged.

    Example usage:

    ```pycon

    >>> from coola import set_summarizer_options, summary
    >>> print(summary("abcdefghijklmnopqrstuvwxyz"))
    <class 'str'> abcdefghijklmnopqrstuvwxyz
    >>> set_summarizer_options(max_characters=10)
    >>> print(summary("abcdefghijklmnopqrstuvwxyz"))
    <class 'str'> abcdefghij...
    >>> set_summarizer_options(max_characters=-1)
    >>> print(summary("abcdefghijklmnopqrstuvwxyz"))
    <class 'str'> abcdefghijklmnopqrstuvwxyz

    ```
    """
    if max_characters is not None:
        Summarizer.set_max_characters(max_characters)
    if max_items is not None:
        Summarizer.set_max_items(max_items)
    if num_spaces is not None:
        Summarizer.set_num_spaces(num_spaces)


@contextmanager
def summarizer_options(**kwargs: Any) -> None:
    r"""Context manager that temporarily changes the summarizer options.

    Accepted arguments are same as ``set_summarizer_options``.
    The context manager temporary change the configuration of
    ``Summarizer``. This context manager has no effect if
    ``Summarizer`` is not used.

    Args:
        **kwargs: Accepted arguments are same as
            ``set_summarizer_options``.

    Example usage:

    ```pycon

    >>> from coola import summarizer_options, summary
    >>> print(summary("abcdefghijklmnopqrstuvwxyz"))
    <class 'str'> abcdefghijklmnopqrstuvwxyz
    >>> with summarizer_options(max_characters=10):
    ...     print(summary("abcdefghijklmnopqrstuvwxyz"))
    ...
    <class 'str'> abcdefghij...
    >>> print(summary("abcdefghijklmnopqrstuvwxyz"))
    <class 'str'> abcdefghijklmnopqrstuvwxyz

    ```
    """
    state = Summarizer.state_dict()
    set_summarizer_options(**kwargs)
    try:
        yield
    finally:
        Summarizer.load_state_dict(state)


if is_numpy_available() and not Summarizer.has_formatter(ndarray):  # pragma: no cover
    Summarizer.add_formatter(ndarray, NDArrayFormatter())

if is_torch_available() and not Summarizer.has_formatter(Tensor):  # pragma: no cover
    Summarizer.add_formatter(Tensor, TensorFormatter())
