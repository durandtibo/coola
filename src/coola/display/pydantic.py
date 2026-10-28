r"""Contain utilities to generate formatted strings of pydantic
objects."""

from __future__ import annotations

__all__ = [
    "is_secret_annotation",
    "repr_pydantic_model",
    "secret_field_names",
    "str_pydantic_model",
]

import warnings
from typing import TYPE_CHECKING, Any, cast, get_args, get_origin

from coola.formatting.structure import repr_mapping_line, str_mapping_line
from coola.utils.imports import is_pydantic_available
from coola.utils.mapping import sort_by_keys

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

if TYPE_CHECKING or is_pydantic_available():  # pragma: no cover
    import pydantic
    from pydantic import BaseModel, SecretBytes, SecretStr
else:  # pragma: no cover
    from coola.utils.fallback.pydantic import BaseModel, SecretBytes, SecretStr, pydantic


def is_secret_annotation(annotation: object) -> bool:
    """Indicate if an annotation is, or contains, a secret type.

    The secret types are ``SecretStr``, ``SecretBytes`` and ``Secret``.
    The arguments of generic aliases (``Optional``, ``tuple``,
    ``Annotated``, ...) are inspected recursively.

    Args:
        annotation: The type annotation to inspect.

    Returns:
        ``True`` if the annotation is or contains a secret type.

    Example:
        ```pycon
        >>> from typing import Optional
        >>> from pydantic import SecretStr
        >>> from coola.display.pydantic import is_secret_annotation
        >>> is_secret_annotation(Optional[SecretStr])
        True
        >>> is_secret_annotation(int)
        False

        ```
    """
    # ``Secret`` is only available in pydantic 2.7 and later.
    secret_types = (SecretStr, SecretBytes, getattr(pydantic, "Secret", SecretBytes))
    origin = get_origin(annotation) or annotation
    if isinstance(origin, type) and issubclass(origin, secret_types):
        return True
    return any(is_secret_annotation(arg) for arg in get_args(annotation))


def secret_field_names(model: BaseModel) -> set[str]:
    """Return the names of fields annotated as a secret type.

    A field is secret if its annotation is ``SecretStr``,
    ``SecretBytes`` or ``Secret``, or contains one of them (for example
    ``Optional[SecretStr]`` or ``tuple[SecretStr, int]``). The whole
    field is then considered secret.

    Args:
        model: The pydantic model to inspect.

    Returns:
        The set of field names whose annotation is (or includes) a
            secret type.

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
    return {
        name
        for name, field in type(model).model_fields.items()
        if is_secret_annotation(field.annotation)
    }


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


def _mask_nested_secret_fields(value: object, dumped: object) -> Any:
    """Apply ``_mask_secret_fields`` to ``value``/``dumped`` pairs
    nested in ``BaseModel``, list/tuple, or dict containers."""
    if isinstance(value, BaseModel) and isinstance(dumped, dict):
        return _mask_secret_fields(value, cast("dict[str, Any]", dumped))
    if isinstance(value, (list, tuple)) and isinstance(dumped, (list, tuple)):
        values = cast("Sequence[object]", value)
        dumped_values = cast("Sequence[object]", dumped)
        items = [
            _mask_nested_secret_fields(v, d) for v, d in zip(values, dumped_values, strict=False)
        ]
        return tuple(items) if isinstance(dumped, tuple) else items
    if isinstance(value, dict) and isinstance(dumped, dict):
        value_map = cast("dict[Any, object]", value)
        dumped_map = cast("dict[Any, object]", dumped)
        return {
            k: _mask_nested_secret_fields(v, dumped_map[k])
            for k, v in value_map.items()
            if k in dumped_map
        }
    return dumped


def _format_pydantic_model(
    model: BaseModel,
    *,
    mapping_line_fn: Callable[[dict[str, Any]], str],
    sort: bool = True,
    exclude_none: bool = False,
    exclude_secret: bool = True,
    exclude_fields: list[str] | None = None,
) -> str:
    """Shared implementation for ``str_pydantic_model`` and
    ``repr_pydantic_model``."""
    config: dict[str, Any] = model.model_dump()
    known_fields = set(type(model).model_fields)
    if exclude_secret:
        config = _mask_secret_fields(model, config)
    if exclude_none:
        config = {k: v for k, v in config.items() if v is not None}
    if exclude_fields:
        excluded = set(exclude_fields)
        unknown = excluded - known_fields
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
        exclude_secret: If ``True``, omit fields typed as a secret type
            (``SecretStr``, ``SecretBytes``, ``Secret``)
            entirely, rather than showing the masked value.
        exclude_fields: Optional list of field names to omit. A
            ``RuntimeWarning`` is emitted for names that are not fields
            of the model; a field already omitted by ``exclude_none`` or
            ``exclude_secret`` is not reported.

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
        exclude_secret: If ``True``, omit fields typed as a secret type
            (``SecretStr``, ``SecretBytes``, ``Secret``)
            entirely, rather than showing the masked value.
        exclude_fields: Optional list of field names to omit. A
            ``RuntimeWarning`` is emitted for names that are not fields
            of the model; a field already omitted by ``exclude_none`` or
            ``exclude_secret`` is not reported.

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
