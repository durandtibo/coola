__all__ = ["ReducerRegistry"]

from coola.reducers.base import BaseReducer
from coola.reducers.basic import BasicReducer
from coola.utils.format import str_indent, str_mapping


class ReducerRegistry:
    """Implements the reducer registry.

    The registry is a class variable, so it is shared with all the
    instances of this class.
    """

    registry: dict[str, BaseReducer] = {"basic": BasicReducer()}

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_mapping(self.registry))}\n)"

    @classmethod
    def add_reducer(cls, name: str, reducer: BaseReducer, exist_ok: bool = False) -> None:
        r"""Adds a reducer to the registry.

        Args:
        ----
            name (str): Specifies the name of the reducer.
            reducer (``BaseReducer``): Specifies the reducer.
            exist_ok (bool, optional): If ``False``, ``RuntimeError``
                is raised if the name already exists. This
                parameter should be set to ``True`` to overwrite the
                existing reducer. Default: ``False``.

        Raises:
        ------
            RuntimeError if a reducer is already registered for the
                name and ``exist_ok=False``.

        Example usage:

        .. code-block:: pycon

            >>> from coola.reducers import ReducerRegistry, BasicReducer
            >>> ReducerRegistry.add_reducer("basic", BasicReducer(), exist_ok=True)
        """
        if name in cls.registry and not exist_ok:
            raise RuntimeError(
                f"A reducer ({cls.registry[name]}) is already registered for the name "
                f"{name}. Please use `exist_ok=True` if you want to overwrite the "
                "reducer for this type"
            )
        cls.registry[name] = reducer

    @classmethod
    def available_reducers(cls) -> tuple[str, ...]:
        """Gets the available reducers.

        Returns:
            tuple of strings: The available reducers.

        Example usage:

        .. code-block:: pycon

            >>> from coola.reducers import ReducerRegistry
            >>> ReducerRegistry.available_reducers()  # doctest: +ELLIPSIS
            (...)
        """
        return tuple(cls.registry.keys())

    @classmethod
    def has_reducer(cls, name: str) -> bool:
        r"""Indicates if a reducer is registered for the given name.

        Args:
        ----
            name (str): Specifies the name to check.

        Returns:
        -------
            bool: ``True`` if a reducer is registered,
                otherwise ``False``.

        Example usage:

        .. code-block:: pycon

            >>> from coola.reducers import ReducerRegistry
            >>> ReducerRegistry.has_reducer("basic")
            True
            >>> ReducerRegistry.has_reducer("missing")
            False
        """
        return name in cls.registry
