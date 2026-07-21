r"""Contain fallback implementations used when ``jax`` dependency is not
available."""

from __future__ import annotations

__all__ = ["jax", "jnp", "numpy"]

from types import ModuleType

from coola.utils.fallback._factory import make_fake_class
from coola.utils.imports import raise_jax_missing_error

FakeClass = make_fake_class(raise_jax_missing_error)

numpy: ModuleType = ModuleType("jax.numpy")
numpy.ndarray = FakeClass

# Create a fake jax package
jax: ModuleType = ModuleType("jax")
jax.numpy = numpy

# Export jnp as an alias for convenience
jnp: ModuleType = jax.numpy
