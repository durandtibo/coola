from __future__ import annotations

import pytest

from coola.hashing import hash_pydantic_model
from coola.hashing.pydantic import unwrap_secrets
from coola.testing.fixtures import pydantic_available
from coola.utils.imports import is_pydantic_available

if is_pydantic_available():
    from pydantic import BaseModel, SecretBytes, SecretStr
else:
    BaseModel, SecretBytes, SecretStr = object, object, object


class Point(BaseModel):
    x: int
    y: int


class Creds(BaseModel):
    username: str
    password: SecretStr


class CredsWithToken(BaseModel):
    username: str
    password: SecretStr
    token: SecretBytes


class Nested(BaseModel):
    name: str
    creds: Creds


###################################
#     Tests for unwrap_secrets    #
###################################

# --- Non-secret values pass through unchanged ---


@pydantic_available
def test_unwrap_secrets_plain_dict() -> None:
    value = {"a": 1, "b": "x"}
    assert unwrap_secrets(value, on_secret="error") == {"a": 1, "b": "x"}


@pydantic_available
def test_unwrap_secrets_plain_list() -> None:
    value = [1, "x", True]
    assert unwrap_secrets(value, on_secret="error") == [1, "x", True]


@pydantic_available
def test_unwrap_secrets_nested_structure_no_secrets() -> None:
    value = {"a": [1, {"b": "x"}], "c": None}
    assert unwrap_secrets(value, on_secret="error") == {"a": [1, {"b": "x"}], "c": None}


@pydantic_available
def test_unwrap_secrets_scalar_passthrough() -> None:
    assert unwrap_secrets(42, on_secret="error") == 42
    assert unwrap_secrets("hello", on_secret="error") == "hello"
    assert unwrap_secrets(None, on_secret="error") is None


# --- on_secret="reveal" ---


@pydantic_available
def test_unwrap_secrets_reveal_secret_str_in_dict() -> None:
    value = {"user": "alice", "pwd": SecretStr("hunter2")}
    assert unwrap_secrets(value, on_secret="reveal") == {"user": "alice", "pwd": "hunter2"}


@pydantic_available
def test_unwrap_secrets_reveal_secret_bytes_in_dict() -> None:
    value = {"token": SecretBytes(b"raw-token")}
    assert unwrap_secrets(value, on_secret="reveal") == {"token": "raw-token"}


@pydantic_available
def test_unwrap_secrets_reveal_secret_in_list() -> None:
    value = [SecretStr("s1"), "x", SecretStr("s2")]
    assert unwrap_secrets(value, on_secret="reveal") == ["s1", "x", "s2"]


@pydantic_available
def test_unwrap_secrets_reveal_top_level_secret_str() -> None:
    assert unwrap_secrets(SecretStr("hunter2"), on_secret="reveal") == "hunter2"


@pydantic_available
def test_unwrap_secrets_reveal_top_level_secret_bytes() -> None:
    assert unwrap_secrets(SecretBytes(b"raw-token"), on_secret="reveal") == "raw-token"


@pydantic_available
def test_unwrap_secrets_reveal_nested_secret() -> None:
    value = {"outer": {"inner": [SecretStr("deep")]}}
    assert unwrap_secrets(value, on_secret="reveal") == {"outer": {"inner": ["deep"]}}


# --- on_secret="exclude" ---


@pydantic_available
def test_unwrap_secrets_exclude_secret_str_in_dict() -> None:
    value = {"user": "alice", "pwd": SecretStr("hunter2")}
    assert unwrap_secrets(value, on_secret="exclude") == {"user": "alice"}


@pydantic_available
def test_unwrap_secrets_exclude_secret_bytes_in_dict() -> None:
    value = {"user": "alice", "token": SecretBytes(b"raw-token")}
    assert unwrap_secrets(value, on_secret="exclude") == {"user": "alice"}


@pydantic_available
def test_unwrap_secrets_exclude_secret_in_list() -> None:
    value = [SecretStr("s1"), "x", SecretStr("s2")]
    assert unwrap_secrets(value, on_secret="exclude") == ["x"]


@pydantic_available
def test_unwrap_secrets_exclude_all_secrets_in_list() -> None:
    value = [SecretStr("s1"), SecretStr("s2")]
    assert unwrap_secrets(value, on_secret="exclude") == []


@pydantic_available
def test_unwrap_secrets_exclude_nested_secret() -> None:
    value = {"outer": {"inner": [SecretStr("deep"), "keep"]}}
    assert unwrap_secrets(value, on_secret="exclude") == {"outer": {"inner": ["keep"]}}


@pydantic_available
def test_unwrap_secrets_exclude_top_level_secret_raises() -> None:
    with pytest.raises(ValueError, match="Cannot exclude a top-level secret"):
        unwrap_secrets(SecretStr("hunter2"), on_secret="exclude")


# --- on_secret="error" ---


@pydantic_available
def test_unwrap_secrets_error_secret_str_in_dict_raises() -> None:
    value = {"pwd": SecretStr("hunter2")}
    with pytest.raises(ValueError, match="SecretStr/SecretBytes"):
        unwrap_secrets(value, on_secret="error")


@pydantic_available
def test_unwrap_secrets_error_secret_bytes_in_list_raises() -> None:
    value = [SecretBytes(b"raw-token")]
    with pytest.raises(ValueError, match="SecretStr/SecretBytes"):
        unwrap_secrets(value, on_secret="error")


@pydantic_available
def test_unwrap_secrets_error_top_level_secret_raises() -> None:
    with pytest.raises(ValueError, match="SecretStr/SecretBytes"):
        unwrap_secrets(SecretStr("hunter2"), on_secret="error")


@pydantic_available
def test_unwrap_secrets_error_no_secret_does_not_raise() -> None:
    assert unwrap_secrets({"user": "alice"}, on_secret="error") == {"user": "alice"}


########################################
#     Tests for hash_pydantic_model    #
########################################

# --- Return type and format ---


@pydantic_available
def test_hash_pydantic_model_returns_str() -> None:
    assert isinstance(hash_pydantic_model(Point(x=1, y=2)), str)


@pydantic_available
def test_hash_pydantic_model_returns_lowercase_hex() -> None:
    result = hash_pydantic_model(Point(x=1, y=2))
    assert all(c in "0123456789abcdef" for c in result)


@pydantic_available
def test_hash_pydantic_model_default_length_is_64() -> None:
    assert len(hash_pydantic_model(Point(x=1, y=2))) == 64


@pydantic_available
@pytest.mark.parametrize("length", [2, 16, 32, 64, 128])
def test_hash_pydantic_model_output_length_matches_requested(length: int) -> None:
    assert len(hash_pydantic_model(Point(x=1, y=2), length=length)) == length


@pydantic_available
def test_hash_pydantic_model_invalid_length_raises() -> None:
    with pytest.raises(ValueError, match="63"):
        hash_pydantic_model(Point(x=1, y=2), length=63)


# --- Determinism and field-order independence ---


@pydantic_available
def test_hash_pydantic_model_is_deterministic() -> None:
    model = Point(x=1, y=2)
    assert hash_pydantic_model(model) == hash_pydantic_model(model)


@pydantic_available
def test_hash_pydantic_model_independent_of_field_declaration_order() -> None:
    assert hash_pydantic_model(Point(x=1, y=2)) == hash_pydantic_model(Point(y=2, x=1))


@pydantic_available
def test_hash_pydantic_model_equal_models_produce_same_hash() -> None:
    assert hash_pydantic_model(Point(x=1, y=2)) == hash_pydantic_model(Point(x=1, y=2))


# --- Sensitivity ---


@pydantic_available
def test_hash_pydantic_model_different_values_produce_different_hashes() -> None:
    assert hash_pydantic_model(Point(x=1, y=2)) != hash_pydantic_model(Point(x=1, y=3))


@pydantic_available
def test_hash_pydantic_model_nested_model_field_change_affects_hash() -> None:
    a = Nested(name="n", creds=Creds(username="alice", password="hunter2"))
    b = Nested(name="n", creds=Creds(username="bob", password="hunter2"))
    assert hash_pydantic_model(a, on_secret="exclude") != hash_pydantic_model(
        b, on_secret="exclude"
    )


# --- on_secret behavior ---


@pydantic_available
def test_hash_pydantic_model_default_on_secret_is_error() -> None:
    with pytest.raises(ValueError, match="SecretStr/SecretBytes"):
        hash_pydantic_model(Creds(username="alice", password="hunter2"))


@pydantic_available
def test_hash_pydantic_model_on_secret_exclude_ignores_secret_value() -> None:
    a = Creds(username="alice", password="hunter2")
    b = Creds(username="alice", password="different")
    assert hash_pydantic_model(a, on_secret="exclude") == hash_pydantic_model(
        b, on_secret="exclude"
    )


@pydantic_available
def test_hash_pydantic_model_on_secret_exclude_still_sensitive_to_non_secret_fields() -> None:
    a = Creds(username="alice", password="hunter2")
    b = Creds(username="bob", password="hunter2")
    assert hash_pydantic_model(a, on_secret="exclude") != hash_pydantic_model(
        b, on_secret="exclude"
    )


@pydantic_available
def test_hash_pydantic_model_on_secret_reveal_sensitive_to_secret_value() -> None:
    a = Creds(username="alice", password="hunter2")
    b = Creds(username="alice", password="different")
    assert hash_pydantic_model(a, on_secret="reveal") != hash_pydantic_model(b, on_secret="reveal")


@pydantic_available
def test_hash_pydantic_model_on_secret_reveal_secret_bytes() -> None:
    a = CredsWithToken(username="alice", password="hunter2", token=b"tok-1")
    b = CredsWithToken(username="alice", password="hunter2", token=b"tok-2")
    assert hash_pydantic_model(a, on_secret="reveal") != hash_pydantic_model(
        b,
        on_secret="reveal",
    )


@pydantic_available
def test_hash_pydantic_model_reveal_and_exclude_produce_different_hashes() -> None:
    model = Creds(username="alice", password="hunter2")
    assert hash_pydantic_model(model, on_secret="reveal") != hash_pydantic_model(
        model,
        on_secret="exclude",
    )


@pydantic_available
def test_hash_pydantic_model_multiple_secret_fields_all_handled() -> None:
    model = CredsWithToken(username="alice", password="hunter2", token=b"raw-token")
    # Should not raise, and should not include either secret's value.
    a = hash_pydantic_model(model, on_secret="exclude")
    b = hash_pydantic_model(
        CredsWithToken(username="alice", password="different", token=b"other-token"),
        on_secret="exclude",
    )
    assert a == b
