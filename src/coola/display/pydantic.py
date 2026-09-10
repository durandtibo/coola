r"""Contain utilities to generate formatted strings of pydantic
objects."""

from __future__ import annotations

__all__ = ["repr_pydantic_model", "secret_field_names", "str_pydantic_model"]

import warnings
from typing import TYPE_CHECKING, Any, get_args

from coola.utils.format import repr_mapping_line, str_mapping_line
from coola.utils.imports import is_pydantic_available
from coola.utils.mapping import sort_by_keys

if TYPE_CHECKING:
    from collections.abc import Callable

if is_pydantic_available():  # pragma: no cover
    from pydantic import BaseModel, SecretStr


def secret_field_names(model: BaseModel) -> set[str]:
    """Return the names of fields annotated as ``SecretStr`` (including
    ``Optional[SecretStr]``).

    Args:
        model: The pydantic model to inspect.

    Returns:
        The set of field names whose annotation is (or includes)
            ``SecretStr``.

    Example:
        ```pycon
        >>> from pydantic import BaseModel, SecretStr
        >>> from coola.display.pydantic import secret_field_names
        >>> class Config(BaseModel):
        ...     name: str
        ...     token: SecretStr
        ...
        >>> secret_field_names(Config(name="my-app", token="s3cr3t"))
        {'token'}

        ```
    """
    names = set()
    for name, field in type(model).model_fields.items():
        annotation = field.annotation
        candidates = get_args(annotation) or (annotation,)
        if SecretStr in candidates:
            names.add(name)
    return names


def _mask_secret_fields(model: BaseModel, dumped: dict[str, Any]) -> dict[str, Any]:
    """Recursively remove ``SecretStr`` fields from a ``model_dump()``
    result, including fields nested inside child ``BaseModel``
    values."""
    secret_fields = secret_field_names(model)
    config = {k: v for k, v in dumped.items() if k not in secret_fields}
    for name in type(model).model_fields:
        if name in config:
            config[name] = _mask_nested_secret_fields(getattr(model, name), config[name])
    return config


def _mask_nested_secret_fields(value: Any, dumped: Any) -> Any:
    """Apply :func:`_mask_secret_fields` to ``value``/``dumped`` pairs
    nested in ``BaseModel``, list/tuple, or dict containers."""
    if isinstance(value, BaseModel) and isinstance(dumped, dict):
        return _mask_secret_fields(value, dumped)
    if isinstance(value, (list, tuple)) and isinstance(dumped, (list, tuple)):
        return type(dumped)(
            _mask_nested_secret_fields(v, d) for v, d in zip(value, dumped, strict=False)
        )
    if isinstance(value, dict) and isinstance(dumped, dict):
        return {
            k: _mask_nested_secret_fields(v, dumped[k]) for k, v in value.items() if k in dumped
        }
    return dumped


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
        config = _mask_secret_fields(model, config)
    if exclude_none:
        config = {k: v for k, v in config.items() if v is not None}
    if exclude_fields:
        excluded = set(exclude_fields)
        unknown = excluded - config.keys()
        if unknown:
            msg = (
                f"'exclude_fields' contains field names that do not exist on "
                f"{type(model).__qualname__}: {sorted(unknown)}"
            )
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
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

    Example:
        ```pycon
        >>> from pydantic import BaseModel
        >>> from coola.display.pydantic import str_pydantic_model
        >>> class Config(BaseModel):
        ...     name: str
        ...     count: int
        ...
        >>> str_pydantic_model(Config(name="my-app", count=3))
        'Config(count=3, name=my-app)'

        ```
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

    Example:
        ```pycon
        >>> from pydantic import BaseModel
        >>> from coola.display.pydantic import repr_pydantic_model
        >>> class Config(BaseModel):
        ...     name: str
        ...     count: int
        ...
        >>> repr_pydantic_model(Config(name="my-app", count=3))
        "Config(count=3, name='my-app')"

        ```
    """
    return _format_pydantic_model(
        model,
        mapping_line_fn=repr_mapping_line,
        sort=sort,
        exclude_none=exclude_none,
        exclude_secret=exclude_secret,
        exclude_fields=exclude_fields,
    )
