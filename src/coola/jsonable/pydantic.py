r"""Define the transformers of the pydantic objects."""

from __future__ import annotations

__all__ = ["get_default_transformers"]

from typing import TYPE_CHECKING, Any

from coola.jsonable.transformers import ConvertTransformer
from coola.utils.imports import is_pydantic_available

if TYPE_CHECKING or is_pydantic_available():
    import pydantic
else:  # pragma: no cover
    from coola.utils.fallback.pydantic import pydantic

if TYPE_CHECKING:
    from coola.recursive.base import Transformer

_MASK = "**********"


def get_default_transformers() -> dict[type, Transformer[Any]]:
    r"""Get the default transformers of the pydantic objects.

    A model gives the ``dict`` of ``model_dump()`` (Python mode), which
    is converted in turn by the registry: the sets inside are sorted
    and the arrays are converted. A serializer declared with
    ``when_used="json"`` is therefore not applied: a model that needs its
    JSON mode registers
    ``ConvertTransformer(lambda m: m.model_dump(mode="json"))`` for its
    class. A ``SecretStr`` and a ``SecretBytes`` give ``"**********"``.

    Returns:
        The transformers of the pydantic objects and their associated
            types.

    Notes:
        This function is called by the default registry the first time it
        needs the transformers of the pydantic objects.

    Example:
        ```pycon
        >>> from coola.jsonable.pydantic import get_default_transformers
        >>> [t.__name__ for t in get_default_transformers()]
        ['BaseModel', 'SecretStr', 'SecretBytes']

        ```
    """
    if not is_pydantic_available():
        return {}
    mask = ConvertTransformer(lambda _: _MASK)
    return {
        pydantic.BaseModel: ConvertTransformer(lambda model: model.model_dump()),
        pydantic.SecretStr: mask,
        pydantic.SecretBytes: mask,
    }
