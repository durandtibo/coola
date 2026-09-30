r"""Contain fallback implementations used when ``jax`` dependency is not
available."""

from __future__ import annotations

__all__ = ["jax", "jnp", "numpy"]

from typing import TYPE_CHECKING

from coola.utils.fallback.factory import make_fake_class, make_fake_module
from coola.utils.imports import raise_jax_missing_error

if TYPE_CHECKING:
    from types import ModuleType

FakeClass: type = make_fake_class(raise_jax_missing_error)

numpy: ModuleType = make_fake_module("jax.numpy", ndarray=FakeClass)

# Create a fake jax package
jax: ModuleType = make_fake_module("jax", numpy=numpy)

# Export jnp as an alias for convenience
jnp: ModuleType = numpy
