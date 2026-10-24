r"""Implement an equality tester for pydantic models.

This module provides an equality tester for ``pydantic.BaseModel``
instances. The fields are compared through the registry, so the
tolerance and ``equal_nan`` apply to the field values, and a model that
holds a NumPy array can be compared.
"""

from __future__ import annotations

__all__ = ["PydanticModelEqualityTester"]

from typing import TYPE_CHECKING, Any

from coola.equality.handler import (
    PydanticModelSameFieldsHandler,
    SameAttributeHandler,
    SameObjectHandler,
    SameTypeHandler,
    TrueHandler,
    create_chain,
)
from coola.equality.tester.chain import BaseChainEqualityTester
from coola.utils.imports import check_pydantic, is_pydantic_available

if TYPE_CHECKING or is_pydantic_available():
    import pydantic
else:  # pragma: no cover
    from coola.utils.fallback.pydantic import pydantic

if TYPE_CHECKING:
    from coola.equality.handler import BaseEqualityHandler
    from coola.equality.tester.base import BaseEqualityTester


class PydanticModelEqualityTester(BaseChainEqualityTester[pydantic.BaseModel]):
    r"""Implement an equality tester for ``pydantic.BaseModel``.

    The two models must have the same type, the same field values and
    the same extra values. The values are compared with the registry,
    so they support the tolerance and ``equal_nan``. The private
    attributes are not compared, as the fields are what the model
    validates and serializes. The handler chain:
    1. SameObjectHandler: Check for object identity
    2. SameTypeHandler: Verify both have the same type
    3. PydanticModelSameFieldsHandler: Compare the values of the declared
       fields (the cached properties are ignored)
    4. SameAttributeHandler("model_extra"): Compare the extra values
    5. TrueHandler: Return True if all checks pass

    Example:
        ```pycon
        >>> from pydantic import BaseModel
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import PydanticModelEqualityTester
        >>> class Point(BaseModel):
        ...     x: float
        ...     y: float
        ...
        >>> config = EqualityConfig()
        >>> tester = PydanticModelEqualityTester()
        >>> tester.objects_are_equal(Point(x=1.0, y=2.0), Point(x=1.0, y=2.0), config)
        True
        >>> tester.objects_are_equal(Point(x=1.0, y=2.0), Point(x=1.0, y=3.0), config)
        False

        ```
    """

    def _check_dependencies(self) -> None:
        check_pydantic()

    def _create_handler(self) -> BaseEqualityHandler:
        return create_chain(
            SameObjectHandler(),
            SameTypeHandler(),
            PydanticModelSameFieldsHandler(),
            SameAttributeHandler(name="model_extra"),
            TrueHandler(),
        )


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
