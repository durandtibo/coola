r"""Define the child finder of the pydantic models."""

from __future__ import annotations

__all__ = ["PydanticChildFinder", "get_default_child_finders"]

from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin
from coola.iterator.base import BaseChildFinder
from coola.utils.imports import is_pydantic_available

if TYPE_CHECKING or is_pydantic_available():
    import pydantic
else:  # pragma: no cover
    from coola.utils.fallback.pydantic import pydantic

if TYPE_CHECKING:
    from collections.abc import Iterator


class PydanticChildFinder(InlineDisplayMixin, BaseChildFinder["pydantic.BaseModel"]):
    r"""Child finder for pydantic models.

    The children of a model are the values of its fields: the declared
    fields first, in the order of the declaration, then the extra fields
    of a model with ``extra="allow"``. The label of a child is the name
    of its field. A computed field is not a child.

    Example:
        ```pycon
        >>> from pydantic import BaseModel
        >>> from coola.iterator.pydantic import PydanticChildFinder
        >>> class Config(BaseModel):
        ...     lr: float
        ...     layers: list
        ...
        >>> list(PydanticChildFinder().find_children(Config(lr=0.1, layers=[1, 2])))
        [0.1, [1, 2]]

        ```
    """

    def find_children(self, data: pydantic.BaseModel) -> Iterator[Any]:
        return (value for _, value in self.find_items(data))

    def find_items(self, data: pydantic.BaseModel) -> Iterator[tuple[str, Any]]:
        names = [*type(data).model_fields, *(data.model_extra or {})]
        return ((name, getattr(data, name)) for name in names)


def get_default_child_finders() -> dict[type, BaseChildFinder[Any]]:
    r"""Get the default child finders of the pydantic models.

    Returns:
        The child finders of the pydantic models and their associated
            types.

    Notes:
        This function is called by the default registry the first time it
        needs the child finders of the pydantic models.

    Example:
        ```pycon
        >>> from coola.iterator.pydantic import get_default_child_finders
        >>> [t.__name__ for t in get_default_child_finders()]
        ['BaseModel']

        ```
    """
    if not is_pydantic_available():
        return {}
    return {pydantic.BaseModel: PydanticChildFinder()}
