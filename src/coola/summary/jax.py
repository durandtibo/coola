r"""Implement the summarizer for JAX arrays.

The summarizer shows metadata (shape, dtype) instead of the values by
default.
"""

from __future__ import annotations

__all__ = ["JaxArraySummarizer"]

from typing import TYPE_CHECKING, Any

from coola.summary.metadata import BaseMetadataSummarizer, Field
from coola.utils.imports import check_jax, is_jax_available

if TYPE_CHECKING or is_jax_available():
    import jax.numpy as jnp
else:  # pragma: no cover
    from coola.utils.fallback.jax import jnp


if TYPE_CHECKING:
    from coola.summary.base import Summarizer


class JaxArraySummarizer(BaseMetadataSummarizer[jnp.ndarray]):
    r"""Implement a summarizer for ``jax.numpy.ndarray`` objects.

    By default, it displays the metadata (type, shape, dtype) rather
    than the values.

    Raises:
        MissingPackageError: If JAX is not installed.

    Example:
        ```pycon
        >>> import jax.numpy as jnp
        >>> from coola.summary import JaxArraySummarizer, SummaryConfig
        >>> summarizer = JaxArraySummarizer()
        >>> print(summarizer.summarize(jnp.ones((2, 3)), SummaryConfig(), compact=False))  # doctest: +ELLIPSIS
        <class '...'> | shape=(2, 3) | dtype=float32

        ```
    """

    def _check_dependencies(self) -> None:
        check_jax()

    def _describe(self, data: jnp.ndarray) -> list[Field]:
        return [f"shape={data.shape}", f"dtype={data.dtype}"]


def get_default_summarizers() -> dict[type, Summarizer[Any]]:
    r"""Get the default summarizers of the JAX arrays.

    Returns:
        The summarizers of the JAX arrays and their associated types.

    Notes:
        This function is called by the default registry the first time it
        needs the summarizers of the JAX arrays.
    """
    if not is_jax_available():
        return {}

    # ``jnp.ndarray`` is ``jax.Array``, in the MRO of the concrete array class. Creating an
    # array to get this class would initialize the JAX backend as a side effect.
    return {jnp.ndarray: JaxArraySummarizer()}
