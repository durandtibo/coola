r"""Implement a registry of reducers."""

__all__ = ["ReducerRegistry"]

from typing import ClassVar

from coola.reducers.base import BaseReducer
from coola.reducers.native import NativeReducer
from coola.utils.format import str_indent, str_mapping


class ReducerRegistry:
    """Implement the reducer registry.

    The registry is a class variable, so it is shared with all the
    instances of this class.
    """

    registry: ClassVar[dict[str, BaseReducer]] = {"native": NativeReducer()}

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_mapping(self.registry))}\n)"

    @classmethod
    def add_reducer(cls, name: str, reducer: BaseReducer, exist_ok: bool = False) -> None:
        r"""Add a reducer to the registry.

        Args:
            name: The name of the reducer.
            reducer: The reducer.
            exist_ok: If ``False``, ``RuntimeError`` is raised if the
                name already exists. This parameter should be set to
                ``True`` to overwrite the existing reducer.

        Raises:
            RuntimeError: if a reducer is already registered for the
                name and ``exist_ok=False``.

        Example usage:

        ```pycon
        >>> from coola.reducers import ReducerRegistry, NativeReducer
        >>> ReducerRegistry.add_reducer("native", NativeReducer(), exist_ok=True)

        ```
        """
        if name in cls.registry and not exist_ok:
            msg = (
                f"A reducer ({cls.registry[name]}) is already registered for the name "
                f"{name}. Please use `exist_ok=True` if you want to overwrite the "
                "reducer for this type"
            )
            raise RuntimeError(msg)
        cls.registry[name] = reducer

    @classmethod
    def available_reducers(cls) -> tuple[str, ...]:
        """Get the available reducers.

        Returns:
            The available reducers.

        Example usage:

        ```pycon
        >>> from coola.reducers import ReducerRegistry
        >>> ReducerRegistry.available_reducers()
        ('native', 'numpy', 'torch')

        ```
        """
        return tuple(cls.registry.keys())

    @classmethod
    def has_reducer(cls, name: str) -> bool:
        r"""Indicate if a reducer is registered for the given name.

        Args:
            name: The name to check.

        Returns:
            ``True`` if a reducer is registered,
                otherwise ``False``.

        Example usage:

        ```pycon
        >>> from coola.reducers import ReducerRegistry
        >>> ReducerRegistry.has_reducer("native")
        True
        >>> ReducerRegistry.has_reducer("missing")
        False

        ```
        """
        return name in cls.registry
