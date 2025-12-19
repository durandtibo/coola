r"""Contain fallback implementations used when ``jax`` dependency is not
available."""

from __future__ import annotations

__all__ = ["jax", "jnp"]

from types import ModuleType
from typing import Any

from coola.utils.imports import raise_error_jax_missing


class FakeClass:
    r"""Fake class that raises an error because jax is not installed.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Raises:
        RuntimeError: jax is required for this functionality.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        raise_error_jax_missing()


# Create a fake jax package
jax: ModuleType = ModuleType("jax")
jax.numpy = ModuleType("jax.numpy")
jax.numpy.ndarray = FakeClass

# Export jnp as an alias for convenience
jnp: ModuleType = jax.numpy
