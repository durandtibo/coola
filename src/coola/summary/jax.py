r"""Implement the summarizer for JAX arrays.

The summarizer shows metadata (shape, dtype) instead of the values by
default.
"""

from __future__ import annotations

__all__ = ["JaxArraySummarizer"]

from typing import TYPE_CHECKING

from coola.summary.metadata import BaseMetadataSummarizer
from coola.utils.imports import check_jax

if TYPE_CHECKING:
    import jax.numpy as jnp


class JaxArraySummarizer(BaseMetadataSummarizer["jnp.ndarray"]):
    r"""Implement a summarizer for ``jax.numpy.ndarray`` objects.

    By default, it displays the metadata (type, shape, dtype) rather
    than the values.

    Args:
        show_data: If ``True``, returns ``repr(array)``, displaying
            the values. If ``False`` (default), returns only the
            metadata: ``<class> | shape=<shape> | dtype=<dtype>``.

    Raises:
        MissingPackageError: If JAX is not installed.

    Example:
        ```pycon
        >>> import jax.numpy as jnp
        >>> from coola.summary import SummarizerRegistry, JaxArraySummarizer
        >>> registry = SummarizerRegistry()
        >>> summarizer = JaxArraySummarizer()
        >>> print(summarizer.summarize(jnp.ones((2, 3)), registry))  # doctest: +ELLIPSIS
        <class '...'> | shape=(2, 3) | dtype=float32

        ```
    """

    def _check_dependencies(self) -> None:
        check_jax()

    def _describe(self, data: jnp.ndarray) -> list[str]:
        return [f"shape={data.shape}", f"dtype={data.dtype}"]
