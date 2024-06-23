r"""Define the summarizer base class."""

from __future__ import annotations

__all__ = ["BaseSummarizer"]


from abc import ABC, abstractmethod
from typing import Any


class BaseSummarizer(ABC):
    r"""Define the base class to implement a summarizer.

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

    @abstractmethod
    def summary(
        self,
        value: Any,
        depth: int = 0,
        max_depth: int = 1,
    ) -> str:
        r"""Summarize the input value in a string.

        Args:
            value: The value to summarize.
            depth: The current depth.
            max_depth: The maximum depth to summarize if the
                input is nested.

        Returns:
            The summary as a string.

        Example usage:

        ```pycon
        >>> from coola import Summarizer
        >>> summarizer = Summarizer()
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
