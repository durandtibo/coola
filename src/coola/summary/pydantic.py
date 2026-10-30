r"""Implement a summarizer for pydantic models."""

from __future__ import annotations

__all__ = ["PydanticModelSummarizer", "get_default_summarizers"]

from typing import TYPE_CHECKING, Any

from coola.display.pydantic import secret_field_names
from coola.summary.object_fields import BaseObjectFieldsSummarizer
from coola.utils.imports import is_pydantic_available

if TYPE_CHECKING or is_pydantic_available():
    import pydantic
else:  # pragma: no cover
    from coola.utils.fallback.pydantic import pydantic

if TYPE_CHECKING:
    from coola.summary.base import Summarizer


class PydanticModelSummarizer(BaseObjectFieldsSummarizer["pydantic.BaseModel"]):
    r"""Summarizer for ``pydantic.BaseModel`` instances.

    The children are the fields and the extra values of the model,
    labeled by their name. The fields annotated as a secret type
    (``SecretStr``, ``SecretBytes``) are not shown, as in
    ``str_pydantic_model``.

    Example:
        ```pycon
        >>> from pydantic import BaseModel, SecretStr
        >>> from coola.summary import SummaryConfig
        >>> from coola.summary.pydantic import PydanticModelSummarizer
        >>> class User(BaseModel):
        ...     name: str
        ...     password: SecretStr
        ...
        >>> node = PydanticModelSummarizer().summarize(
        ...     User(name="a", password="b"), SummaryConfig(), compact=False
        ... )
        >>> list(node.children)
        [('name', 'a')]

        ```
    """

    def _get_names(self, data: pydantic.BaseModel) -> list[str]:
        r"""Get the names of the fields to show.

        Args:
            data: The pydantic model.

        Returns:
            The names of the fields, followed by the extra fields. The
                secret fields are hidden.
        """
        secrets = secret_field_names(data)
        names = [name for name in type(data).model_fields if name not in secrets]
        names.extend(data.model_extra or {})
        return names


def get_default_summarizers() -> dict[type, Summarizer[Any]]:
    r"""Get the default summarizers of the pydantic objects.

    Returns:
        The summarizers of the pydantic objects and their associated types.

    Notes:
        This function is called by the default registry the first time it
        needs the summarizers of the pydantic objects.
    """
    if not is_pydantic_available():
        return {}
    return {pydantic.BaseModel: PydanticModelSummarizer()}
