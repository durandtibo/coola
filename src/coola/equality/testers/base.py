r"""Define the tester base class."""

from __future__ import annotations

__all__ = ["BaseEqualityTester"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig

logger = logging.getLogger(__name__)


class BaseEqualityTester(ABC):
    r"""Define the base class to implement an equality tester."""

    @abstractmethod
    def equal(self, actual: Any, expected: Any, config: EqualityConfig) -> bool:
        r"""Indicate if two objects are equal or not.

        Args:
            actual: The actual input.
            expected: The expected input.
            config: The equality configuration.

        Returns:
            ``True`` if the two objects are equal, otherwise ``False``.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from coola.equality import EqualityConfig
        >>> from coola.equality.testers import EqualityTester
        >>> tester = EqualityTester()
        >>> config = EqualityConfig(tester=tester)
        >>> tester.equal([np.ones((2, 3)), np.zeros(2)], [np.ones((2, 3)), np.zeros(2)], config)
        True
        >>> tester.equal([np.ones((2, 3)), np.ones(2)], [np.ones((2, 3)), np.zeros(2)], config)
        False

        ```
        """
