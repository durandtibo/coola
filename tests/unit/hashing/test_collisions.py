from __future__ import annotations

import pytest

from coola.hashing import hash_object
from coola.utils.imports import is_numpy_available

if is_numpy_available():
    import numpy as np


@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param(1, "1", id="int-str"),
        pytest.param(1.5, "1.5", id="float-str"),
        pytest.param(True, "True", id="bool-str"),
        pytest.param(None, "None", id="none-str"),
        pytest.param([], "", id="empty-list-str"),
        pytest.param({}, "", id="empty-dict-str"),
        pytest.param([], {}, id="empty-list-dict"),
        pytest.param(b"", "", id="empty-bytes-str"),
        pytest.param(b"abc", "abc", id="bytes-str"),
        pytest.param(["a"], "a", id="list-str"),
    ],
)
def test_hash_object_different_types_do_not_collide(first: object, second: object) -> None:
    assert hash_object(first) != hash_object(second)


def test_hash_object_list_and_tuple_hash_the_same() -> None:
    assert hash_object([1, 2]) == hash_object((1, 2))


@pytest.mark.skipif(not is_numpy_available(), reason="Requires NumPy")
def test_hash_object_numpy_scalar_hashes_like_builtin() -> None:
    assert hash_object(np.float64(1.5)) == hash_object(1.5)
