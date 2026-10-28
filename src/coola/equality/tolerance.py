r"""Define the tolerance of a comparison."""

from __future__ import annotations

__all__ = ["Tolerance"]

from dataclasses import dataclass

from coola.validation import validate_finite, validate_ge, validate_number


@dataclass(frozen=True, slots=True)
class Tolerance:
    r"""Define the tolerance of a comparison.

    Two numbers ``actual`` and ``expected`` are close if
    ``|actual - expected| <= atol + rtol * max(|actual|, |expected|)``
    (the ``numpy.isclose`` rule made symmetric). The default is no tolerance (exact comparison).

    Args:
        rtol: The relative tolerance parameter. Must be non-negative
            and finite. Defaults to 0.0.
        atol: The absolute tolerance parameter. Must be non-negative
            and finite. Defaults to 0.0.

    Raises:
        TypeError: If ``rtol`` or ``atol`` is not a real number (a
            ``bool`` is rejected).
        ValueError: If ``rtol`` or ``atol`` is negative or not finite.

    Example:
        ```pycon
        >>> from coola.equality import Tolerance
        >>> Tolerance()
        Tolerance(rtol=0.0, atol=0.0)
        >>> Tolerance.numpy_default()
        Tolerance(rtol=1e-05, atol=1e-08)

        ```
    """

    rtol: float = 0.0
    atol: float = 0.0

    def __post_init__(self) -> None:
        validate_number(self.rtol, name="rtol")
        validate_number(self.atol, name="atol")
        validate_ge(self.rtol, 0, name="rtol")
        validate_ge(self.atol, 0, name="atol")
        validate_finite(self.rtol, name="rtol")
        validate_finite(self.atol, name="atol")

    @classmethod
    def numpy_default(cls) -> Tolerance:
        r"""Create the tolerance used by ``objects_are_allclose``.

        The values (``rtol=1e-5``, ``atol=1e-8``) are the defaults of
        ``numpy.allclose``.

        Returns:
            The tolerance ``Tolerance(rtol=1e-5, atol=1e-8)``.

        Example:
            ```pycon
            >>> from coola.equality import Tolerance
            >>> Tolerance.numpy_default()
            Tolerance(rtol=1e-05, atol=1e-08)

            ```
        """
        return cls(rtol=1e-5, atol=1e-8)
