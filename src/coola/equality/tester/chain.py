r"""Implement a base class for equality testers that run a handler
chain."""

from __future__ import annotations

__all__ = ["BaseChainEqualityTester"]

from abc import abstractmethod
from typing import TYPE_CHECKING, TypeVar

from coola.display import InlineDisplayMixin
from coola.equality.tester.base import BaseEqualityTester
from coola.utils.introspection import is_same_type

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig
    from coola.equality.handler import BaseEqualityHandler

T = TypeVar("T")


class BaseChainEqualityTester(InlineDisplayMixin, BaseEqualityTester[T]):
    r"""Define the base class for the equality testers that compare two
    objects with a chain of handlers.

    A child class only needs to implement ``_create_handler``, which
    returns the first handler of the chain. If the tester requires an
    optional package, it can also override ``_check_dependencies`` to
    raise an error when the package is not installed. The chain is
    created when the tester is initialized.

    Two testers of the same class are equal. A child class with
    parameters must override ``_display_fields`` and ``equal``.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import ObjectEqualHandler, SameObjectHandler, create_chain
        >>> from coola.equality.tester.chain import BaseChainEqualityTester
        >>> class MyTester(BaseChainEqualityTester[object]):
        ...     def _create_handler(self):
        ...         return create_chain(SameObjectHandler(), ObjectEqualHandler())
        ...
        >>> tester = MyTester()
        >>> tester
        MyTester()
        >>> tester.objects_are_equal(1, 1, EqualityConfig())
        True
        >>> tester.objects_are_equal(1, 2, EqualityConfig())
        False

        ```
    """

    def __init__(self) -> None:
        self._check_dependencies()
        self._handler = self._create_handler()

    def _check_dependencies(self) -> None:
        r"""Check that the packages required by the tester are installed.

        By default, the tester has no requirement.

        Raises:
            MissingPackageError: if a required package is not installed.
        """

    @abstractmethod
    def _create_handler(self) -> BaseEqualityHandler:
        r"""Create the chain of handlers used to compare two objects.

        Returns:
            The first handler of the chain.
        """

    def equal(self, other: object) -> bool:
        return is_same_type(other, self)

    def objects_are_equal(self, actual: T, expected: object, config: EqualityConfig) -> bool:
        return self._handler.handle(actual, expected, config=config)
