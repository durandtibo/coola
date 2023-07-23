from __future__ import annotations

__all__ = ["Reduction"]


from coola.reducers import auto_reducer
from coola.reducers.registry import ReducerRegistry


class Reduction:
    r"""Implement the class that defines the reduction strategy."""

    reducer = auto_reducer()

    @classmethod
    def available_reducers(cls) -> tuple[str, ...]:
        """Gets the available reducers.

        Returns:
            tuple of strings: The available reducers.

        Example usage:

        .. code-block:: pycon

            >>> from coola import Reduction
            >>> Reduction.available_reducers()  # doctest: +ELLIPSIS
            (...)
        """
        return ReducerRegistry.available_reducers()

    @classmethod
    def check_reducer(cls, reducer: str) -> None:
        r"""Checks if the reducer is available.

        Args:
        ----
            reducer (str): Specifies the reducer name.

        Raises:
        ------
            RuntimeError if the reducer is not available.

        Example usage:

        .. code-block:: pycon

            >>> from coola import Reduction
            >>> Reduction.check_reducer("torch")
        """
        if reducer not in (reducers := cls.available_reducers()):
            raise RuntimeError(f"Incorrect reducer {reducer}. Reducer should be one of {reducers}")

    @classmethod
    def initialize(cls, reducer: str) -> None:
        r"""Initializes the reduction strategy.

        Args:
        ----
            reducer (str): Specifies the name of the reducer to use.

        Example usage:

        .. code-block:: pycon

            >>> from coola import Reduction
            >>> Reduction.initialize("torch")
        """
        cls.check_reducer(reducer)
        cls.reducer = ReducerRegistry.registry[reducer]
