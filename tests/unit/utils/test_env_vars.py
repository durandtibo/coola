"""Unit tests for environment variable utilities."""

from __future__ import annotations

import os
from unittest import mock

import pytest

from coola.utils.env_vars import temp_env_vars

###################################
#     Tests for temp_env_vars     #
###################################


@mock.patch.dict(os.environ, {}, clear=True)
def test_temp_env_vars_set_new_variable() -> None:
    """Test setting a new environment variable that doesn't exist."""
    assert "TEST_VAR" not in os.environ

    with temp_env_vars({"TEST_VAR": "test_value"}):
        assert os.environ["TEST_VAR"] == "test_value"

    assert "TEST_VAR" not in os.environ


@mock.patch.dict(os.environ, {}, clear=True)
def test_temp_env_vars_set_multiple_new_variables() -> None:
    """Test setting multiple new environment variables."""
    assert "VAR1" not in os.environ
    assert "VAR2" not in os.environ
    assert "VAR3" not in os.environ

    with temp_env_vars({"VAR1": "value1", "VAR2": "value2", "VAR3": "value3"}):
        assert os.environ["VAR1"] == "value1"
        assert os.environ["VAR2"] == "value2"
        assert os.environ["VAR3"] == "value3"

    assert "VAR1" not in os.environ
    assert "VAR2" not in os.environ
    assert "VAR3" not in os.environ


@mock.patch.dict(os.environ, {"EXISTING_VAR": "original_value"}, clear=True)
def test_temp_env_vars_override_existing_variable() -> None:
    """Test overriding an existing environment variable."""
    assert os.environ["EXISTING_VAR"] == "original_value"

    with temp_env_vars({"EXISTING_VAR": "temporary_value"}):
        assert os.environ["EXISTING_VAR"] == "temporary_value"

    assert os.environ["EXISTING_VAR"] == "original_value"


@mock.patch.dict(os.environ, {"VAR1": "original1", "VAR2": "original2"}, clear=True)
def test_temp_env_vars_mix_new_and_existing() -> None:
    """Test mixing new variables and overriding existing ones."""
    assert os.environ["VAR1"] == "original1"
    assert os.environ["VAR2"] == "original2"
    assert "VAR3" not in os.environ

    with temp_env_vars({"VAR1": "temp1", "VAR3": "temp3"}):
        assert os.environ["VAR1"] == "temp1"
        assert os.environ["VAR2"] == "original2"  # Unchanged
        assert os.environ["VAR3"] == "temp3"

    assert os.environ["VAR1"] == "original1"
    assert os.environ["VAR2"] == "original2"
    assert "VAR3" not in os.environ


@mock.patch.dict(os.environ, {}, clear=True)
def test_temp_env_vars_empty_dict() -> None:
    """Test with an empty dictionary (no changes)."""
    with temp_env_vars({}):
        pass

    assert len(os.environ) == 0


@mock.patch.dict(os.environ, {}, clear=True)
def test_temp_env_vars_converts_to_string() -> None:
    """Test that non-string values are converted to strings."""
    with temp_env_vars({"INT_VAR": 42, "FLOAT_VAR": 3.14, "BOOL_VAR": True}):
        assert os.environ["INT_VAR"] == "42"
        assert os.environ["FLOAT_VAR"] == "3.14"
        assert os.environ["BOOL_VAR"] == "True"

    assert "INT_VAR" not in os.environ
    assert "FLOAT_VAR" not in os.environ
    assert "BOOL_VAR" not in os.environ


@mock.patch.dict(os.environ, {"EXISTING_VAR": "original"}, clear=True)
def test_temp_env_vars_restores_on_exception() -> None:
    """Test that original values are restored even when an exception
    occurs."""
    with (  # noqa: PT012
        pytest.raises(ValueError, match="Test exception"),
        temp_env_vars({"EXISTING_VAR": "temporary", "NEW_VAR": "new"}),
    ):
        assert os.environ["EXISTING_VAR"] == "temporary"
        assert os.environ["NEW_VAR"] == "new"
        msg = "Test exception"
        raise ValueError(msg)

    # Verify cleanup happened despite the exception
    assert os.environ["EXISTING_VAR"] == "original"
    assert "NEW_VAR" not in os.environ


@mock.patch.dict(os.environ, {}, clear=True)
def test_temp_env_vars_nested_contexts() -> None:
    """Test nested context managers."""
    with temp_env_vars({"VAR": "outer"}):
        assert os.environ["VAR"] == "outer"

        with temp_env_vars({"VAR": "inner"}):
            assert os.environ["VAR"] == "inner"

        # Should restore to outer value, not remove it
        assert os.environ["VAR"] == "outer"

    assert "VAR" not in os.environ


@mock.patch.dict(os.environ, {"VAR1": "original1"}, clear=True)
def test_temp_env_vars_nested_with_different_vars() -> None:
    """Test nested contexts with different variables."""
    with temp_env_vars({"VAR1": "temp1", "VAR2": "temp2"}):
        assert os.environ["VAR1"] == "temp1"
        assert os.environ["VAR2"] == "temp2"

        with temp_env_vars({"VAR3": "temp3"}):
            assert os.environ["VAR1"] == "temp1"  # Unchanged
            assert os.environ["VAR2"] == "temp2"  # Unchanged
            assert os.environ["VAR3"] == "temp3"

        assert os.environ["VAR1"] == "temp1"
        assert os.environ["VAR2"] == "temp2"
        assert "VAR3" not in os.environ

    assert os.environ["VAR1"] == "original1"
    assert "VAR2" not in os.environ


@mock.patch.dict(os.environ, {}, clear=True)
def test_temp_env_vars_special_characters() -> None:
    """Test environment variables with special characters in values."""
    special_value = "value with spaces and special chars: !@#$%^&*()"

    with temp_env_vars({"SPECIAL_VAR": special_value}):
        assert os.environ["SPECIAL_VAR"] == special_value

    assert "SPECIAL_VAR" not in os.environ


@mock.patch.dict(os.environ, {"VAR": "original"}, clear=True)
def test_temp_env_vars_multiple_sequential_uses() -> None:
    """Test using the context manager multiple times sequentially."""
    assert os.environ["VAR"] == "original"

    with temp_env_vars({"VAR": "first"}):
        assert os.environ["VAR"] == "first"

    assert os.environ["VAR"] == "original"

    with temp_env_vars({"VAR": "second"}):
        assert os.environ["VAR"] == "second"

    assert os.environ["VAR"] == "original"
