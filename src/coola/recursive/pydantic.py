r"""Define the transformer of the pydantic models."""

from __future__ import annotations

__all__ = ["PydanticModelTransformer", "get_default_transformers"]

from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin
from coola.recursive.base import BaseTransformer
from coola.utils.imports import is_pydantic_available

if TYPE_CHECKING or is_pydantic_available():
    import pydantic
else:  # pragma: no cover
    from coola.utils.fallback.pydantic import pydantic

if TYPE_CHECKING:
    from coola.recursive.base import Expansion, Transformer


class PydanticModelTransformer(InlineDisplayMixin, BaseTransformer["pydantic.BaseModel"]):
    r"""Transformer for pydantic models that recursively transforms the
    fields.

    The values of the fields are the children: the declared fields
    first, in the order of the declaration, then the extra fields of a
    model with ``extra="allow"``. The model is rebuilt with
    ``model_copy(update=...)``: the input is not modified, a frozen
    model is supported, and no validator runs again.

    Notes:
        - The new values are not validated, so a function that changes
          the type of a value gives a model that does not respect its
          annotations.
        - A computed field is not a child, because it is derived from
          the other fields.

    Example:
        ```pycon
        >>> from pydantic import BaseModel
        >>> from coola.recursive import (
        ...     DefaultTransformer,
        ...     SequenceTransformer,
        ...     TransformerRegistry,
        ... )
        >>> from coola.recursive.pydantic import PydanticModelTransformer
        >>> class Config(BaseModel):
        ...     lr: float
        ...     layers: list
        ...
        >>> registry = TransformerRegistry(
        ...     {
        ...         object: DefaultTransformer(),
        ...         list: SequenceTransformer(),
        ...         Config: PydanticModelTransformer(),
        ...     }
        ... )
        >>> registry.transform(Config(lr=0.1, layers=[1, 2]), str)
        Config(lr='0.1', layers=['1', '2'])

        ```
    """

    def expand(self, data: pydantic.BaseModel) -> Expansion | None:
        names = [*type(data).model_fields, *(data.model_extra or {})]

        def rebuild(items: list[Any]) -> Any:
            r"""Rebuild the model.

            Args:
                items: The transformed values of the fields, in the
                    order of the fields.

            Returns:
                A copy of the model with the new values.
            """
            return data.model_copy(update=dict(zip(names, items, strict=True)))

        return [getattr(data, name) for name in names], rebuild


def get_default_transformers() -> dict[type, Transformer[Any]]:
    r"""Get the default transformers of the pydantic models.

    Returns:
        The transformers of the pydantic models and their associated
            types.

    Notes:
        This function is called by the default registry the first time it
        needs the transformers of the pydantic models.

    Example:
        ```pycon
        >>> from coola.recursive.pydantic import get_default_transformers
        >>> [t.__name__ for t in get_default_transformers()]
        ['BaseModel']

        ```
    """
    if not is_pydantic_available():
        return {}
    return {pydantic.BaseModel: PydanticModelTransformer()}
