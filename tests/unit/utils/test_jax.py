from __future__ import annotations

from unittest.mock import Mock, patch

from coola.testing.marks import jax_available
from coola.utils.imports import is_jax_available
from coola.utils.jax import get_jax_array_types

if is_jax_available():
    import jax
else:  # pragma: no cover
    jax = Mock()


########################################
#     Tests for get_jax_array_types    #
########################################


@jax_available
def test_get_jax_array_types_starts_with_jax_array() -> None:
    assert get_jax_array_types()[0] is jax.Array


@jax_available
def test_get_jax_array_types_covers_the_concrete_array_class() -> None:
    assert any(issubclass(type(jax.numpy.ones(1)), t) for t in get_jax_array_types())


@jax_available
def test_get_jax_array_types_adds_the_concrete_class_if_jax_array_is_not_in_its_mro() -> None:
    impl = type("ArrayImpl", (), {})
    with patch("coola.utils.jax.import_module", return_value=Mock(ArrayImpl=impl)):
        assert get_jax_array_types() == (jax.Array, impl)


@jax_available
def test_get_jax_array_types_skips_the_concrete_class_if_jax_array_is_in_its_mro() -> None:
    impl = type("ArrayImpl", (jax.Array,), {})
    with patch("coola.utils.jax.import_module", return_value=Mock(ArrayImpl=impl)):
        assert get_jax_array_types() == (jax.Array,)


@jax_available
def test_get_jax_array_types_import_error() -> None:
    with patch("coola.utils.jax.import_module", side_effect=ImportError("no module")):
        assert get_jax_array_types() == (jax.Array,)


@jax_available
def test_get_jax_array_types_missing_attribute() -> None:
    with patch("coola.utils.jax.import_module", return_value=object()):
        assert get_jax_array_types() == (jax.Array,)


def test_get_jax_array_types_jax_not_available() -> None:
    with patch("coola.utils.jax.is_jax_available", return_value=False):
        assert get_jax_array_types() == ()
