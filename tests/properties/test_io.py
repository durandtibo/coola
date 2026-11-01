r"""Property-based tests of ``coola.io``: the round trips of the
formats, and a failed save leaves the directory as it was."""

from __future__ import annotations

import math
import pickle
import tempfile
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings, strategies as st

from coola.io import (
    PickleLoader,
    PickleSaver,
    TextLoader,
    TextSaver,
    load_json,
    load_pickle,
    load_text,
    save_json,
    save_pickle,
    save_text,
)

# Lone surrogates cannot be encoded in UTF-8.
texts = st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=50)
keys = st.text(max_size=5)

json_values = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | texts,
    lambda children: st.lists(children, max_size=4) | st.dictionaries(keys, children, max_size=4),
    max_leaves=10,
)
pickle_values = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False)
    | st.binary(max_size=10)
    | texts,
    lambda children: (
        st.lists(children, max_size=4)
        | st.tuples(children, children)
        | st.sets(st.integers(), max_size=4)
        | st.dictionaries(keys, children, max_size=4)
    ),
    max_leaves=10,
)

# A temporary directory per example: the ``tmp_path`` fixture is shared by all the examples.
_settings = settings(max_examples=30, deadline=None)


@_settings
@given(text=texts)
def test_text_round_trip(text: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir).joinpath("data.txt")
        save_text(text, path)
        assert load_text(path) == text


@_settings
@given(text=st.text(alphabet=st.sampled_from("ab\r\n"), max_size=20))
def test_text_round_trip_preserves_line_endings(text: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir).joinpath("data.txt")
        TextSaver().save(text, path)
        assert TextLoader().load(path) == text


@_settings
@given(value=json_values)
def test_json_round_trip(value: Any) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir).joinpath("data.json")
        save_json(value, path)
        assert load_json(path) == value


@_settings
@given(value=pickle_values)
def test_pickle_round_trip(value: Any) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir).joinpath("data.pkl")
        save_pickle(value, path)
        assert load_pickle(path) == value


@_settings
@given(value=pickle_values)
def test_pickle_classes_round_trip(value: Any) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir).joinpath("data.pkl")
        PickleSaver().save(value, path)
        assert PickleLoader().load(path) == value


@_settings
@given(value=st.sampled_from([math.nan, math.inf, -math.inf]))
def test_json_allow_nan_false_rejects_nan_and_leaves_the_directory_empty(value: float) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="Out of range float values"):
            save_json(value, Path(tmpdir).joinpath("data.json"), allow_nan=False)
        assert list(Path(tmpdir).iterdir()) == []


@_settings
@given(value=st.sampled_from([object(), lambda: 1, {1, 2}, b"abc"]))
def test_json_save_of_a_rejected_object_leaves_the_directory_empty(value: Any) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(TypeError, match="not JSON serializable"):
            save_json(value, Path(tmpdir).joinpath("data.json"))
        assert list(Path(tmpdir).iterdir()) == []


@_settings
@given(value=st.sampled_from([lambda: 1, (x for x in [1])]))
def test_pickle_save_of_a_rejected_object_leaves_the_directory_empty(value: Any) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises((pickle.PicklingError, AttributeError, TypeError), match="pickle"):
            save_pickle(value, Path(tmpdir).joinpath("data.pkl"))
        assert list(Path(tmpdir).iterdir()) == []
