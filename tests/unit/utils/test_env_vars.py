"""Unit tests for environment variable utilities."""

from __future__ import annotations

import contextlib
import logging
import os
from unittest import mock
from unittest.mock import patch

import pytest

from coola.utils.env_vars import check_env_vars, get_required_env_var, temp_env_vars

####################################
#     Tests for check_env_vars     #
####################################


@patch.dict(os.environ, {"TEST_VAR": "test_value"}, clear=True)
def test_check_env_vars_returns_true_for_defined_var() -> None:
    """Test that a defined environment variable is reported as True."""
    assert check_env_vars(["TEST_VAR"]) == {"TEST_VAR": True}


@patch.dict(os.environ, {}, clear=True)
def test_check_env_vars_returns_false_for_missing_var() -> None:
    """Test that a missing environment variable is reported as False."""
    assert check_env_vars(["MISSING_VAR"]) == {"MISSING_VAR": False}


@patch.dict(os.environ, {"TEST_VAR": ""}, clear=True)
def test_check_env_vars_returns_false_for_empty_string_value() -> None:
    """Test that a variable set to an empty string is reported as not
    defined."""
    assert check_env_vars(["TEST_VAR"]) == {"TEST_VAR": False}


@patch.dict(os.environ, {"TEST_VAR": "  "}, clear=True)
def test_check_env_vars_returns_false_for_blank_value() -> None:
    """Test that a variable set to blank characters is reported as not
    defined."""
    assert check_env_vars(["TEST_VAR"]) == {"TEST_VAR": False}


@patch.dict(os.environ, {"VAR_ONE": "value_one"}, clear=True)
def test_check_env_vars_returns_mapping_for_multiple_vars() -> None:
    """Test that a mix of defined and missing variables is reported
    correctly."""
    assert check_env_vars(["VAR_ONE", "VAR_TWO"]) == {
        "VAR_ONE": True,
        "VAR_TWO": False,
    }


@patch.dict(os.environ, {}, clear=True)
def test_check_env_vars_returns_empty_dict_for_empty_list() -> None:
    """Test that an empty list of variable names returns an empty
    dict."""
    assert check_env_vars([]) == {}


@patch.dict(os.environ, {"TEST_VAR": "test_value"}, clear=True)
def test_check_env_vars_does_not_raise_by_default() -> None:
    """Test that no error is raised when a variable is missing and
    raise_on_missing is not set."""
    check_env_vars(["MISSING_VAR"])


@patch.dict(os.environ, {"TEST_VAR": "test_value"}, clear=True)
def test_check_env_vars_does_not_raise_when_all_defined() -> None:
    """Test that no error is raised when raise_on_missing is True but
    all variables are defined."""
    check_env_vars(["TEST_VAR"], raise_on_missing=True)


@patch.dict(os.environ, {}, clear=True)
def test_check_env_vars_raises_on_missing_var() -> None:
    """Test that EnvironmentError is raised when a variable is missing
    and raise_on_missing is True."""
    with pytest.raises(
        EnvironmentError, match=r"Missing required environment variable\(s\): MISSING_VAR"
    ):
        check_env_vars(["MISSING_VAR"], raise_on_missing=True)


@patch.dict(os.environ, {"VAR_ONE": "value_one"}, clear=True)
def test_check_env_vars_raises_and_lists_all_missing_vars() -> None:
    """Test that EnvironmentError lists all missing variables, not just
    the first one."""
    with pytest.raises(
        EnvironmentError,
        match=r"Missing required environment variable\(s\): VAR_TWO, VAR_THREE",
    ):
        check_env_vars(["VAR_ONE", "VAR_TWO", "VAR_THREE"], raise_on_missing=True)


@patch.dict(os.environ, {}, clear=True)
def test_check_env_vars_logs_success_message(caplog: pytest.LogCaptureFixture) -> None:
    """Test that a defined variable logs a plain-text info message."""
    os.environ["TEST_VAR"] = "test_value"
    with caplog.at_level(logging.INFO):
        check_env_vars(["TEST_VAR"])
    assert any(
        record.levelno == logging.INFO and "'TEST_VAR' is defined." in record.message
        for record in caplog.records
    )


@patch.dict(os.environ, {}, clear=True)
def test_check_env_vars_logs_warning_message(caplog: pytest.LogCaptureFixture) -> None:
    """Test that a missing variable logs a plain-text warning
    message."""
    with caplog.at_level(logging.WARNING):
        check_env_vars(["MISSING_VAR"])
    assert any(
        record.levelno == logging.WARNING and "'MISSING_VAR' is NOT defined." in record.message
        for record in caplog.records
    )


##########################################
#     Tests for get_required_env_var     #
##########################################


@patch.dict(os.environ, {"TEST_VAR": "test_value"}, clear=True)
def test_get_required_env_var_returns_value() -> None:
    """Test that a valid environment variable is returned."""
    assert get_required_env_var("TEST_VAR") == "test_value"


@patch.dict(os.environ, {"TEST_VAR": "  test_value  "}, clear=True)
def test_get_required_env_var_strips_whitespace() -> None:
    """Test that leading and trailing whitespace is removed."""
    assert get_required_env_var("TEST_VAR") == "test_value"


@patch.dict(os.environ, {"TEST_VAR": "\t\ntest_value\n\t"}, clear=True)
def test_get_required_env_var_strips_tabs_and_newlines() -> None:
    """Test that tabs and newlines are stripped."""
    assert get_required_env_var("TEST_VAR") == "test_value"


@patch.dict(os.environ, {}, clear=True)
def test_get_required_env_var_raises_on_missing_var() -> None:
    """Test that ValueError is raised when variable doesn't exist."""
    with pytest.raises(ValueError, match=r"Environment variable 'MISSING_VAR' is required"):
        get_required_env_var("MISSING_VAR")


@patch.dict(os.environ, {"EMPTY_VAR": ""}, clear=True)
def test_get_required_env_var_raises_on_empty_string() -> None:
    """Test that ValueError is raised when variable is empty string."""
    with pytest.raises(ValueError, match=r"Environment variable 'EMPTY_VAR' is required"):
        get_required_env_var("EMPTY_VAR")


@patch.dict(os.environ, {"WHITESPACE_VAR": "   "}, clear=True)
def test_get_required_env_var_raises_on_whitespace_only() -> None:
    """Test that ValueError is raised when variable contains only
    whitespace."""
    with pytest.raises(ValueError, match="Environment variable 'WHITESPACE_VAR' is required"):
        get_required_env_var("WHITESPACE_VAR")


@patch.dict(os.environ, {"TAB_VAR": "\t\t\t"}, clear=True)
def test_get_required_env_var_raises_on_tabs_only() -> None:
    """Test that ValueError is raised when variable contains only
    tabs."""
    with pytest.raises(ValueError, match="Environment variable 'TAB_VAR' is required"):
        get_required_env_var("TAB_VAR")


@patch.dict(os.environ, {"TEST_VAR": "hello world"}, clear=True)
def test_get_required_env_var_preserves_internal_whitespace() -> None:
    """Test that whitespace within the value is preserved."""
    assert get_required_env_var("TEST_VAR") == "hello world"


@patch.dict(os.environ, {"TEST_VAR": "user@example.com:p@ssw0rd!"}, clear=True)
def test_get_required_env_var_with_special_characters() -> None:
    """Test that special characters in values are handled correctly."""
    assert get_required_env_var("TEST_VAR") == "user@example.com:p@ssw0rd!"


###################################
#     Tests for temp_env_vars     #
###################################


@mock.patch.dict(os.environ, {}, clear=True)
def test_temp_env_vars_set_new_variable() -> None:
    """Test setting a new environment variable that doesn't exist."""
    with temp_env_vars({"TEST_VAR": "test_value"}):
        assert os.environ["TEST_VAR"] == "test_value"


@mock.patch.dict(os.environ, {}, clear=True)
def test_temp_env_vars_remove_new_variable_after_the_context() -> None:
    """Test that a new environment variable is removed after the
    context."""
    with temp_env_vars({"TEST_VAR": "test_value"}):
        pass
    assert "TEST_VAR" not in os.environ


@pytest.mark.parametrize(
    ("name", "value"), [("VAR1", "value1"), ("VAR2", "value2"), ("VAR3", "value3")]
)
@mock.patch.dict(os.environ, {}, clear=True)
def test_temp_env_vars_set_multiple_new_variables(name: str, value: str) -> None:
    """Test setting multiple new environment variables."""
    with temp_env_vars({"VAR1": "value1", "VAR2": "value2", "VAR3": "value3"}):
        assert os.environ[name] == value


@pytest.mark.parametrize("name", ["VAR1", "VAR2", "VAR3"])
@mock.patch.dict(os.environ, {}, clear=True)
def test_temp_env_vars_remove_multiple_new_variables_after_the_context(name: str) -> None:
    """Test that multiple new environment variables are removed after
    the context."""
    with temp_env_vars({"VAR1": "value1", "VAR2": "value2", "VAR3": "value3"}):
        pass
    assert name not in os.environ


@mock.patch.dict(os.environ, {"EXISTING_VAR": "original_value"}, clear=True)
def test_temp_env_vars_override_existing_variable() -> None:
    """Test overriding an existing environment variable."""
    with temp_env_vars({"EXISTING_VAR": "temporary_value"}):
        assert os.environ["EXISTING_VAR"] == "temporary_value"


@mock.patch.dict(os.environ, {"EXISTING_VAR": "original_value"}, clear=True)
def test_temp_env_vars_restore_existing_variable_after_the_context() -> None:
    """Test restoring an overridden environment variable."""
    with temp_env_vars({"EXISTING_VAR": "temporary_value"}):
        pass
    assert os.environ["EXISTING_VAR"] == "original_value"


@mock.patch.dict(os.environ, {"VAR1": "original1", "VAR2": "original2"}, clear=True)
def test_temp_env_vars_mix_new_and_existing_overrides() -> None:
    """Test overriding an existing variable when mixing new and existing
    ones."""
    with temp_env_vars({"VAR1": "temp1", "VAR3": "temp3"}):
        assert os.environ["VAR1"] == "temp1"


@mock.patch.dict(os.environ, {"VAR1": "original1", "VAR2": "original2"}, clear=True)
def test_temp_env_vars_mix_new_and_existing_keeps_unchanged() -> None:
    """Test that a variable that is not set is unchanged."""
    with temp_env_vars({"VAR1": "temp1", "VAR3": "temp3"}):
        assert os.environ["VAR2"] == "original2"  # Unchanged


@mock.patch.dict(os.environ, {"VAR1": "original1", "VAR2": "original2"}, clear=True)
def test_temp_env_vars_mix_new_and_existing_sets_new() -> None:
    """Test that a new variable is set."""
    with temp_env_vars({"VAR1": "temp1", "VAR3": "temp3"}):
        assert os.environ["VAR3"] == "temp3"


@mock.patch.dict(os.environ, {"VAR1": "original1", "VAR2": "original2"}, clear=True)
def test_temp_env_vars_mix_new_and_existing_restores_existing() -> None:
    """Test restoring the existing variable after the context."""
    with temp_env_vars({"VAR1": "temp1", "VAR3": "temp3"}):
        pass
    assert os.environ["VAR1"] == "original1"


@mock.patch.dict(os.environ, {"VAR1": "original1", "VAR2": "original2"}, clear=True)
def test_temp_env_vars_mix_new_and_existing_keeps_unchanged_after_the_context() -> None:
    """Test that a variable that is not set is unchanged after the
    context."""
    with temp_env_vars({"VAR1": "temp1", "VAR3": "temp3"}):
        pass
    assert os.environ["VAR2"] == "original2"


@mock.patch.dict(os.environ, {"VAR1": "original1", "VAR2": "original2"}, clear=True)
def test_temp_env_vars_mix_new_and_existing_removes_new() -> None:
    """Test removing the new variable after the context."""
    with temp_env_vars({"VAR1": "temp1", "VAR3": "temp3"}):
        pass
    assert "VAR3" not in os.environ


@mock.patch.dict(os.environ, {}, clear=True)
def test_temp_env_vars_empty_dict() -> None:
    """Test with an empty dictionary (no changes)."""
    with temp_env_vars({}):
        pass

    assert len(os.environ) == 0


@pytest.mark.parametrize(
    ("name", "value"), [("INT_VAR", "42"), ("FLOAT_VAR", "3.14"), ("BOOL_VAR", "True")]
)
@mock.patch.dict(os.environ, {}, clear=True)
def test_temp_env_vars_converts_to_string(name: str, value: str) -> None:
    """Test that non-string values are converted to strings."""
    with temp_env_vars({"INT_VAR": 42, "FLOAT_VAR": 3.14, "BOOL_VAR": True}):
        assert os.environ[name] == value


@pytest.mark.parametrize("name", ["INT_VAR", "FLOAT_VAR", "BOOL_VAR"])
@mock.patch.dict(os.environ, {}, clear=True)
def test_temp_env_vars_removes_converted_variables_after_the_context(name: str) -> None:
    """Test that the converted variables are removed after the
    context."""
    with temp_env_vars({"INT_VAR": 42, "FLOAT_VAR": 3.14, "BOOL_VAR": True}):
        pass
    assert name not in os.environ


@mock.patch.dict(os.environ, {"EXISTING_VAR": "original"}, clear=True)
def test_temp_env_vars_restores_existing_variable_on_exception() -> None:
    """Test that original values are restored even when an exception
    occurs."""
    with contextlib.suppress(ValueError), temp_env_vars({"EXISTING_VAR": "temporary"}):
        msg = "Test exception"
        raise ValueError(msg)
    assert os.environ["EXISTING_VAR"] == "original"


@mock.patch.dict(os.environ, {"EXISTING_VAR": "original"}, clear=True)
def test_temp_env_vars_removes_new_variable_on_exception() -> None:
    """Test that new variables are removed even when an exception
    occurs."""
    with contextlib.suppress(ValueError), temp_env_vars({"NEW_VAR": "new"}):
        msg = "Test exception"
        raise ValueError(msg)
    assert "NEW_VAR" not in os.environ


@mock.patch.dict(os.environ, {"EXISTING_VAR": "original"}, clear=True)
def test_temp_env_vars_propagates_the_exception() -> None:
    """Test that the exception raised in the context is propagated."""
    msg = "Test exception"
    with (
        pytest.raises(ValueError, match=r"Test exception"),
        temp_env_vars({"EXISTING_VAR": "temporary"}),
    ):
        raise ValueError(msg)


@mock.patch.dict(os.environ, {}, clear=True)
def test_temp_env_vars_nested_contexts_outer_value() -> None:
    """Test the outer value of nested context managers."""
    with temp_env_vars({"VAR": "outer"}):
        assert os.environ["VAR"] == "outer"


@mock.patch.dict(os.environ, {}, clear=True)
def test_temp_env_vars_nested_contexts_inner_value() -> None:
    """Test the inner value of nested context managers."""
    with temp_env_vars({"VAR": "outer"}), temp_env_vars({"VAR": "inner"}):
        assert os.environ["VAR"] == "inner"


@mock.patch.dict(os.environ, {}, clear=True)
def test_temp_env_vars_nested_contexts_restores_the_outer_value() -> None:
    """Test restoring to outer value, not removing it."""
    with temp_env_vars({"VAR": "outer"}):
        with temp_env_vars({"VAR": "inner"}):
            pass
        assert os.environ["VAR"] == "outer"


@mock.patch.dict(os.environ, {}, clear=True)
def test_temp_env_vars_nested_contexts_removes_the_variable() -> None:
    """Test removing the variable after the outer context."""
    with temp_env_vars({"VAR": "outer"}), temp_env_vars({"VAR": "inner"}):
        pass
    assert "VAR" not in os.environ


@pytest.mark.parametrize(
    ("name", "value"), [("VAR1", "temp1"), ("VAR2", "temp2"), ("VAR3", "temp3")]
)
@mock.patch.dict(os.environ, {"VAR1": "original1"}, clear=True)
def test_temp_env_vars_nested_with_different_vars(name: str, value: str) -> None:
    """Test nested contexts with different variables."""
    with (
        temp_env_vars({"VAR1": "temp1", "VAR2": "temp2"}),
        temp_env_vars({"VAR3": "temp3"}),
    ):
        assert os.environ[name] == value


@pytest.mark.parametrize(("name", "value"), [("VAR1", "temp1"), ("VAR2", "temp2")])
@mock.patch.dict(os.environ, {"VAR1": "original1"}, clear=True)
def test_temp_env_vars_nested_with_different_vars_keeps_the_outer_vars(
    name: str, value: str
) -> None:
    """Test that the outer variables are kept after the inner
    context."""
    with temp_env_vars({"VAR1": "temp1", "VAR2": "temp2"}):
        with temp_env_vars({"VAR3": "temp3"}):
            pass
        assert os.environ[name] == value


@mock.patch.dict(os.environ, {"VAR1": "original1"}, clear=True)
def test_temp_env_vars_nested_with_different_vars_removes_the_inner_var() -> None:
    """Test that the inner variable is removed after the inner
    context."""
    with temp_env_vars({"VAR1": "temp1", "VAR2": "temp2"}):
        with temp_env_vars({"VAR3": "temp3"}):
            pass
        assert "VAR3" not in os.environ


@mock.patch.dict(os.environ, {"VAR1": "original1"}, clear=True)
def test_temp_env_vars_nested_with_different_vars_restores_the_existing_var() -> None:
    """Test restoring the existing variable after the outer context."""
    with temp_env_vars({"VAR1": "temp1", "VAR2": "temp2"}), temp_env_vars({"VAR3": "temp3"}):
        pass
    assert os.environ["VAR1"] == "original1"


@mock.patch.dict(os.environ, {"VAR1": "original1"}, clear=True)
def test_temp_env_vars_nested_with_different_vars_removes_the_new_var() -> None:
    """Test removing the new variable after the outer context."""
    with temp_env_vars({"VAR1": "temp1", "VAR2": "temp2"}), temp_env_vars({"VAR3": "temp3"}):
        pass
    assert "VAR2" not in os.environ


@mock.patch.dict(os.environ, {}, clear=True)
def test_temp_env_vars_special_characters() -> None:
    """Test environment variables with special characters in values."""
    special_value = "value with spaces and special chars: !@#$%^&*()"
    with temp_env_vars({"SPECIAL_VAR": special_value}):
        assert os.environ["SPECIAL_VAR"] == special_value


@mock.patch.dict(os.environ, {}, clear=True)
def test_temp_env_vars_special_characters_removed_after_the_context() -> None:
    """Test that the variable with special characters is removed."""
    with temp_env_vars({"SPECIAL_VAR": "value with spaces and special chars: !@#$%^&*()"}):
        pass
    assert "SPECIAL_VAR" not in os.environ


@mock.patch.dict(os.environ, {"VAR": "original"}, clear=True)
def test_temp_env_vars_multiple_sequential_uses_first() -> None:
    """Test the first use of the context manager."""
    with temp_env_vars({"VAR": "first"}):
        assert os.environ["VAR"] == "first"


@mock.patch.dict(os.environ, {"VAR": "original"}, clear=True)
def test_temp_env_vars_multiple_sequential_uses_restores_after_the_first() -> None:
    """Test restoring after the first use."""
    with temp_env_vars({"VAR": "first"}):
        pass
    assert os.environ["VAR"] == "original"


@mock.patch.dict(os.environ, {"VAR": "original"}, clear=True)
def test_temp_env_vars_multiple_sequential_uses_second() -> None:
    """Test the second use of the context manager."""
    with temp_env_vars({"VAR": "first"}):
        pass
    with temp_env_vars({"VAR": "second"}):
        assert os.environ["VAR"] == "second"


@mock.patch.dict(os.environ, {"VAR": "original"}, clear=True)
def test_temp_env_vars_multiple_sequential_uses_restores_after_the_second() -> None:
    """Test restoring after the second use."""
    with temp_env_vars({"VAR": "first"}):
        pass
    with temp_env_vars({"VAR": "second"}):
        pass
    assert os.environ["VAR"] == "original"


def test_temp_env_vars_removed_inside_block_does_not_raise() -> None:
    name = "COOLA_TEST_TEMP_ENV_REMOVED"
    os.environ.pop(name, None)
    with temp_env_vars({name: "value"}):
        del os.environ[name]
    assert name not in os.environ


def test_temp_env_vars_restores_when_setting_fails() -> None:
    name = "COOLA_TEST_TEMP_ENV_PARTIAL"
    os.environ.pop(name, None)
    # The second value cannot be stored in the environment (embedded null byte).
    with pytest.raises(ValueError, match="null"), temp_env_vars({name: "value", "BAD": "a\0b"}):
        pass  # pragma: no cover


def test_temp_env_vars_removes_the_set_variable_when_setting_fails() -> None:
    name = "COOLA_TEST_TEMP_ENV_PARTIAL"
    os.environ.pop(name, None)
    with contextlib.suppress(ValueError), temp_env_vars({name: "value", "BAD": "a\0b"}):
        pass  # pragma: no cover
    assert name not in os.environ


def test_temp_env_vars_does_not_set_the_bad_variable_when_setting_fails() -> None:
    name = "COOLA_TEST_TEMP_ENV_PARTIAL"
    os.environ.pop(name, None)
    with contextlib.suppress(ValueError), temp_env_vars({name: "value", "BAD": "a\0b"}):
        pass  # pragma: no cover
    assert "BAD" not in os.environ


def test_check_env_vars_missing_level(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        check_env_vars(["MISSING_VAR_FOR_LEVEL"], missing_level=logging.INFO)
    assert [r.levelno for r in caplog.records if "NOT defined" in r.message] == [logging.INFO]
