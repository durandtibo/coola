from __future__ import annotations

__all__ = ["BaseSummarizer"]


from abc import ABC, abstractmethod
from typing import Any


class BaseSummarizer(ABC):
    r"""Define the base class to implement a summarizer."""

    @abstractmethod
    def summary(
        self,
        value: Any,
        depth: int = 0,
        max_depth: int = 1,
    ) -> str:
        r"""Summarize the input value in a string.

        Args:
        ----
            value: Specifies the value to summarize.
            max_depth (int, optional): Specifies the maximum depth to
                summarize if the input is nested. Default: ``1``
            summarizer (``BaseSummarizer`` or ``None``): Specifies the
                summarization strategy. If ``None``, the default
                ``Summarizer`` is used. Default: ``None``

        Returns:
        -------
            str: The summary as a string.

        Example usage:

        .. code-block:: pycon

            >>> from coola import Summarizer
            >>> print(Summarizer().summary(1))
            <class 'int'> 1
            >>> print(Summarizer().summary(["abc", "def"]))
            <class 'list'> (length=2)
              (0): abc
              (1): def
            >>> print(Summarizer().summary([[0, 1, 2], {"key1": "abc", "key2": "def"}]))
            <class 'list'> (length=2)
              (0): [0, 1, 2]
              (1): {'key1': 'abc', 'key2': 'def'}
            >>> print(Summarizer().summary([[0, 1, 2], {"key1": "abc", "key2": "def"}], max_depth=2))
            <class 'list'> (length=2)
              (0): <class 'list'> (length=3)
                  (0): 0
                  (1): 1
                  (2): 2
              (1): <class 'dict'> (length=2)
                  (key1): abc
                  (key2): def
        """
