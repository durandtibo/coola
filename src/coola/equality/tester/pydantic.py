r"""Implement an equality tester for pydantic models.

This module provides an equality tester for ``pydantic.BaseModel``
instances. The fields are compared through the registry, so the
tolerance and ``equal_nan`` apply to the field values, and a model that
holds a NumPy array can be compared.
"""

from __future__ import annotations

__all__ = ["PydanticModelEqualityTester"]

from typing import TYPE_CHECKING, Any

from coola.equality._helpers.utils import AttributeName
from coola.equality.tester.base import BaseEqualityTester
from coola.utils.imports import check_pydantic, is_pydantic_available

if TYPE_CHECKING or is_pydantic_available():
    import pydantic
else:  # pragma: no cover
    from coola.utils.fallback.pydantic import pydantic

if TYPE_CHECKING:
    from collections.abc import Iterator

    from coola.equality.engine import Engine


class PydanticModelEqualityTester(BaseEqualityTester[pydantic.BaseModel]):
    r"""Implement an equality tester for ``pydantic.BaseModel``.

    The two models must have the same type, the same field values and
    the same extra values. The values are compared with the registry, so
    they support the tolerance and ``equal_nan``. The private attributes
    are not compared, as the fields are what the model validates and
    serializes. The other entries of the instance ``__dict__`` (for
    example the value of a ``functools.cached_property`` that was read
    on one model only) are ignored too. The location of a difference is
    the name of the field, or ``model_extra``.

    Raises:
        MissingPackageError: If ``pydantic`` is not installed.

    Example:
        ```pycon
        >>> from pydantic import BaseModel
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.engine import Engine
        >>> from coola.equality.tester import PydanticModelEqualityTester, get_default_registry
        >>> class Point(BaseModel):
        ...     x: float
        ...     y: float
        ...
        >>> engine = Engine(EqualityConfig())
        >>> engine.run(Point(x=1.0, y=2.0), Point(x=1.0, y=2.0))
        True
        >>> engine.run(Point(x=1.0, y=2.0), Point(x=1.0, y=3.0))
        False

        ```
    """

    def __init__(self) -> None:
        check_pydantic()

    def objects_are_equal(
        self, actual: pydantic.BaseModel, expected: pydantic.BaseModel, _ctx: Engine
    ) -> bool | Iterator[tuple[object, object, object]]:
        # ``type(actual).model_fields``: the access from an instance is deprecated.
        yield from (
            (getattr(actual, name), getattr(expected, name), AttributeName(name))
            for name in type(actual).model_fields
        )
        yield actual.model_extra, expected.model_extra, AttributeName("model_extra")


def get_default_testers() -> dict[type, BaseEqualityTester[Any]]:
    r"""Get the default equality testers of the pydantic objects.

    Returns:
        The equality testers of the pydantic objects and their associated types.

    Notes:
        This function is called by the default registry the first time it
        needs the equality testers of the pydantic objects.
    """
    if not is_pydantic_available():
        return {}

    return {pydantic.BaseModel: PydanticModelEqualityTester()}
