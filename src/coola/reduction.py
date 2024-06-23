r"""Implement the main reduction features."""

from __future__ import annotations

__all__ = ["Reduction"]


from coola.reducers import auto_reducer
from coola.reducers.registry import ReducerRegistry


class Reduction:
    r"""Implement the class that defines the reduction strategy."""

    reducer = auto_reducer()

    @classmethod
    def available_reducers(cls) -> tuple[str, ...]:
        """Get the available reducers.

        Returns:
            The available reducers.

        Example usage:

        ```pycon
        >>> from coola import Reduction
        >>> Reduction.available_reducers()
        (...)

        ```
        """
        return ReducerRegistry.available_reducers()

    @classmethod
    def check_reducer(cls, reducer: str) -> None:
        r"""Check if the reducer is available.

        Args:
            reducer: The reducer name.

        Raises:
            RuntimeError: if the reducer is not available.

        Example usage:

        ```pycon
        >>> from coola import Reduction
        >>> Reduction.check_reducer("torch")

        ```
        """
        if reducer not in (reducers := cls.available_reducers()):
            msg = f"Incorrect reducer {reducer}. Reducer should be one of {reducers}"
            raise RuntimeError(msg)

    @classmethod
    def initialize(cls, reducer: str) -> None:
        r"""Initialize the reduction strategy.

        Args:
            reducer: The name of the reducer to use.

        Example usage:

        ```pycon
        >>> from coola import Reduction
        >>> Reduction.initialize("torch")

        ```
        """
        cls.check_reducer(reducer)
        cls.reducer = ReducerRegistry.registry[reducer]
