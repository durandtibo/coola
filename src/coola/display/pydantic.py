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
from typing import TYPE_CHECKING, Any, TypedDict, get_args, get_origin

from coola.formatting.structure import repr_mapping_line, str_mapping_line
from coola.utils.imports import is_pydantic_available
from coola.utils.mapping import sort_by_keys

if TYPE_CHECKING:
    from collections.abc import Callable

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


class _Options(TypedDict):
    """The options of the root model that apply to the nested models."""

    sort: bool
    exclude_none: bool
    exclude_secret: bool


class _NestedModel:
    """Wrap a nested pydantic model so that its ``str`` and ``repr`` are
    the formatted strings of :func:`str_pydantic_model` and
    :func:`repr_pydantic_model`, with the options of the root model.

    The generic ``str_mapping_line`` and ``repr_mapping_line`` can then
    format the values of a model without knowing about pydantic.

    Args:
        model: The nested pydantic model.
        options: The options ``sort``, ``exclude_none`` and
            ``exclude_secret`` of the root model.
    """

    def __init__(self, model: BaseModel, options: _Options) -> None:
        self.model = model
        self.options = options

    def __repr__(self) -> str:
        r"""Return the ``repr`` of the model, with the options of the
        root model.

        Returns:
            The representation of the model.
        """
        return _format_model(self.model, line_fn=repr_mapping_line, **self.options)

    def __str__(self) -> str:
        r"""Return the ``str`` of the model, with the options of the root
        model.

        Returns:
            The string of the model.
        """
        return _format_model(self.model, line_fn=str_mapping_line, **self.options)

    def __eq__(self, other: object) -> bool:
        r"""Indicate if two wrappers hold equal models.

        Args:
            other: The other object.

        Returns:
            ``True`` if ``other`` is a wrapper of an equal model,
                otherwise ``False``.
        """
        return isinstance(other, _NestedModel) and self.model == other.model

    def __hash__(self) -> int:
        r"""Return the hash of the model.

        Returns:
            The hash of the wrapped model.
        """
        return hash(self.model)


def _shown(value: object, options: _Options) -> object:
    """Prepare a field value to be formatted.

    A pydantic model is replaced by a :class:`_NestedModel`, so it
    keeps its name and the options of the root model. The lists,
    tuples, sets, frozensets and dicts (exactly these types) are rebuilt
    with their values prepared. Any other value is returned unchanged.

    Args:
        value: The value of a field.
        options: The options ``sort``, ``exclude_none`` and
            ``exclude_secret`` of the root model.

    Returns:
        The value to format.

    Example:
        ```pycon
        >>> from coola.display.pydantic import _shown
        >>> _shown([1, 2], {})
        [1, 2]

        ```
    """
    if isinstance(value, BaseModel):
        return _NestedModel(value, options)
    if isinstance(value, (list, tuple, set, frozenset)) and type(value) in {
        list,
        tuple,
        set,
        frozenset,
    }:
        return type(value)(_shown(v, options) for v in value)
    if type(value) is dict:
        return {k: _shown(v, options) for k, v in value.items()}
    return value


def _format_model(
    model: BaseModel,
    *,
    line_fn: Callable[[dict[str, Any]], str],
    sort: bool,
    exclude_none: bool,
    exclude_secret: bool,
    exclude_fields: list[str] | None = None,
) -> str:
    r"""Shared implementation for ``str_pydantic_model`` and
    ``repr_pydantic_model``.

    The fields are read from the model (no serializer runs), the fields
    declared with ``Field(repr=False)`` are hidden, and the values of
    ``extra="allow"`` are shown.

    Args:
        model: The model to format.
        line_fn: The function that formats the fields on one line.
        sort: If ``True``, the fields are sorted by name.
        exclude_none: If ``True``, the fields whose value is ``None``
            are hidden.
        exclude_secret: If ``True``, the secret fields are hidden.
        exclude_fields: The names of the fields to hide. A name that is
            not a field of the model raises a warning.

    Returns:
        The formatted model.
    """
    fields: dict[str, Any] = {
        name: getattr(model, name) for name, info in type(model).model_fields.items() if info.repr
    }
    fields |= model.model_extra or {}
    if exclude_secret:
        for name in secret_field_names(model):
            fields.pop(name, None)
    if exclude_none:
        fields = {k: v for k, v in fields.items() if v is not None}
    if exclude_fields:
        excluded = set(exclude_fields)
        unknown = excluded - set(type(model).model_fields) - set(model.model_extra or {})
        if unknown:
            msg = (
                f"'exclude_fields' contains field names that do not exist on "
                f"{type(model).__qualname__}: {sorted(unknown)}"
            )
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
        fields = {k: v for k, v in fields.items() if k not in excluded}
    if sort:
        fields = sort_by_keys(fields)
    options = _Options(sort=sort, exclude_none=exclude_none, exclude_secret=exclude_secret)
    shown = {k: _shown(v, options) for k, v in fields.items()}
    return f"{type(model).__qualname__}({line_fn(shown)})"


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

    The fields are read from the model: no serializer runs, a field
    declared with ``Field(repr=False)`` is hidden (as in the ``repr`` of
    pydantic), the values of ``extra="allow"`` are shown, and a
    ``computed_field`` is not shown because it is not a field. A nested
    model keeps its name and is formatted with ``sort``,
    ``exclude_none`` and ``exclude_secret`` (also inside lists, tuples,
    sets, frozensets and dicts). The nested secrets are never shown:
    they are omitted, or masked by pydantic if ``exclude_secret`` is
    ``False``.

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
            ``exclude_secret`` is not reported. The names apply to the
            root model only.

    Returns:
        A string like ``"ClassName(field1=value1, field2=value2)"``.

    Example:
        ```pycon
        >>> from pydantic import BaseModel
        >>> from coola.display.pydantic import str_pydantic_model
        >>> class Child(BaseModel):
        ...     name: str
        ...
        >>> class Config(BaseModel):
        ...     name: str
        ...     count: int
        ...     child: Child
        ...
        >>> str_pydantic_model(Config(name="my-app", count=3, child=Child(name="a")))
        'Config(child=Child(name=a), count=3, name=my-app)'

        ```
    """
    return _format_model(
        model,
        line_fn=str_mapping_line,
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

    It has the same options and behaviour as
    :func:`str_pydantic_model`, but it uses ``repr`` for each value.

    Args:
        model: The pydantic model to format.
        sort: See :func:`str_pydantic_model`.
        exclude_none: See :func:`str_pydantic_model`.
        exclude_secret: See :func:`str_pydantic_model`.
        exclude_fields: See :func:`str_pydantic_model`.

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
    return _format_model(
        model,
        line_fn=repr_mapping_line,
        sort=sort,
        exclude_none=exclude_none,
        exclude_secret=exclude_secret,
        exclude_fields=exclude_fields,
    )
