r"""Contain utilities to generate formatted strings of pydantic
objects."""

from __future__ import annotations

__all__ = ["repr_pydantic_model", "secret_field_names", "str_pydantic_model"]

from typing import TYPE_CHECKING, Any, get_args

from coola.utils.format import repr_mapping_line, str_mapping_line
from coola.utils.imports import is_pydantic_available
from coola.utils.mapping import sort_by_keys

if TYPE_CHECKING:
    from collections.abc import Callable

if is_pydantic_available():
    from pydantic import BaseModel, SecretStr


def secret_field_names(model: BaseModel) -> set[str]:
    """Return the names of fields annotated as ``SecretStr`` (including
    ``Optional[SecretStr]``).

    Args:
        model: The pydantic model to inspect.

    Returns:
        The set of field names whose annotation is (or includes)
            ``SecretStr``.
    """
    names = set()
    for name, field in type(model).model_fields.items():
        annotation = field.annotation
        candidates = get_args(annotation) or (annotation,)
        if SecretStr in candidates:
            names.add(name)
    return names


def _format_pydantic_model(
    model: BaseModel,
    *,
    mapping_line_fn: Callable[[dict], str],
    sort: bool = True,
    exclude_none: bool = False,
    exclude_secret: bool = True,
    exclude_fields: list[str] | None = None,
) -> str:
    """Shared implementation for ``str_pydantic_model`` and
    ``repr_pydantic_model``."""
    config: dict[str, Any] = model.model_dump()
    if exclude_secret:
        secret_fields = secret_field_names(model)
        config = {k: v for k, v in config.items() if k not in secret_fields}
    if exclude_none:
        config = {k: v for k, v in config.items() if v is not None}
    if exclude_fields:
        excluded = set(exclude_fields)
        config = {k: v for k, v in config.items() if k not in excluded}
    if sort:
        config = sort_by_keys(config)
    cls = type(model).__qualname__
    return f"{cls}({mapping_line_fn(config)})"


def str_pydantic_model(
    model: BaseModel,
    *,
    sort: bool = True,
    exclude_none: bool = False,
    exclude_secret: bool = True,
    exclude_fields: list[str] | None = None,
) -> str:
    """Return a formatted, single-line string representation of a
    pydantic model.

    Args:
        model: The pydantic model to format.
        sort: If ``True``, sort fields by name.
        exclude_none: If ``True``, omit fields whose value is ``None``.
        exclude_secret: If ``True``, omit fields typed as ``SecretStr``
            entirely, rather than showing the masked value.
        exclude_fields: Optional list of field names to omit. Names
            that do not exist on the model are silently ignored.

    Returns:
        A string like ``"ClassName(field1=value1, field2=value2)"``.
    """
    return _format_pydantic_model(
        model,
        mapping_line_fn=str_mapping_line,
        sort=sort,
        exclude_none=exclude_none,
        exclude_secret=exclude_secret,
        exclude_fields=exclude_fields,
    )


def repr_pydantic_model(
    model: BaseModel,
    *,
    sort: bool = True,
    exclude_none: bool = False,
    exclude_secret: bool = True,
    exclude_fields: list[str] | None = None,
) -> str:
    """Return a formatted, single-line ``repr``-style representation of
    a pydantic model.

    Args:
        model: The pydantic model to format.
        sort: If ``True``, sort fields by name.
        exclude_none: If ``True``, omit fields whose value is ``None``.
        exclude_secret: If ``True``, omit fields typed as ``SecretStr``
            entirely, rather than showing the masked value.
        exclude_fields: Optional list of field names to omit. Names
            that do not exist on the model are silently ignored.

    Returns:
        A string like ``"ClassName(field1=value1, field2=value2)"``
            using ``repr`` for each value.
    """
    return _format_pydantic_model(
        model,
        mapping_line_fn=repr_mapping_line,
        sort=sort,
        exclude_none=exclude_none,
        exclude_secret=exclude_secret,
        exclude_fields=exclude_fields,
    )
