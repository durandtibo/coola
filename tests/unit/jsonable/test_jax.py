from __future__ import annotations

from unittest.mock import patch

from coola.jsonable import to_jsonable
from coola.jsonable.jax import get_default_transformers
from coola.testing.marks import jax_available
from coola.utils.imports import is_jax_available

if is_jax_available():
    import jax.numpy as jnp


@jax_available
def test_get_default_transformers_keys() -> None:
    assert list(get_default_transformers()) == [jnp.ndarray]


def test_get_default_transformers_without_jax() -> None:
    with patch("coola.jsonable.jax.is_jax_available", lambda: False):
        assert get_default_transformers() == {}


@jax_available
def test_to_jsonable_array() -> None:
    assert to_jsonable(jnp.arange(3)) == [0, 1, 2]


@jax_available
def test_to_jsonable_array_0d() -> None:
    assert to_jsonable(jnp.array(1.5)) == 1.5


@jax_available
def test_to_jsonable_array_2d() -> None:
    assert to_jsonable(jnp.ones((2, 2))) == [[1.0, 1.0], [1.0, 1.0]]


@jax_available
def test_to_jsonable_array_empty() -> None:
    assert to_jsonable(jnp.array([])) == []


@jax_available
def test_to_jsonable_array_in_a_container() -> None:
    assert to_jsonable({"a": (jnp.arange(2),)}) == {"a": [[0, 1]]}
