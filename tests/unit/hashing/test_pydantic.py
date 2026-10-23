from __future__ import annotations

from collections import OrderedDict
from typing import Any

import pytest

from coola.hashing import HasherRegistry, get_default_registry, hash_pydantic_model
from coola.hashing.pydantic import PydanticModelHasher, unwrap_secrets
from coola.testing.fixtures import pydantic_available
from coola.utils.imports import is_pydantic_available

if is_pydantic_available():
    from pydantic import BaseModel, ConfigDict, SecretBytes, SecretStr
else:
    BaseModel, ConfigDict, SecretBytes, SecretStr = object, dict, object, object


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


class Point3(BaseModel):
    x: int
    y: int


class AnyValue(BaseModel):
    value: Any


class Open(BaseModel):
    model_config = ConfigDict(extra="allow")

    x: int


class Nested(BaseModel):
    name: str
    creds: Creds


@pytest.fixture
def registry() -> HasherRegistry:
    return get_default_registry()


#########################################
#     Tests for PydanticModelHasher     #
#########################################


@pydantic_available
def test_pydantic_model_hasher_repr_default() -> None:
    assert repr(PydanticModelHasher()) == "PydanticModelHasher(on_secret='error')"


@pydantic_available
def test_pydantic_model_hasher_repr_custom_on_secret() -> None:
    assert (
        repr(PydanticModelHasher(on_secret="exclude")) == "PydanticModelHasher(on_secret='exclude')"
    )


@pydantic_available
def test_pydantic_model_hasher_str_default() -> None:
    assert str(PydanticModelHasher()) == "PydanticModelHasher(on_secret=error)"


@pydantic_available
def test_pydantic_model_hasher_default_on_secret_is_error() -> None:
    assert PydanticModelHasher()._on_secret == "error"  # noqa: S105


@pydantic_available
def test_pydantic_model_hasher_hash_matches_function(registry: HasherRegistry) -> None:
    point = Point(x=1, y=2)
    assert PydanticModelHasher().hash(point, registry=registry) == hash_pydantic_model(point)


@pydantic_available
def test_pydantic_model_hasher_hash_known_value(registry: HasherRegistry) -> None:
    point = Point(x=1, y=2)
    assert (
        PydanticModelHasher().hash(point, registry=registry)
        == "91738e8f4e95f91bd94446057cccce6c1c4349a0f4af76f062a49cc8d4191b26"
    )


@pydantic_available
def test_pydantic_model_hasher_hash_returns_str(registry: HasherRegistry) -> None:
    assert isinstance(PydanticModelHasher().hash(Point(x=1, y=2), registry=registry), str)


@pydantic_available
def test_pydantic_model_hasher_hash_default_length_is_64(registry: HasherRegistry) -> None:
    assert len(PydanticModelHasher().hash(Point(x=1, y=2), registry=registry)) == 64


@pydantic_available
@pytest.mark.parametrize(
    ("length", "expected"),
    [
        pytest.param(16, "b679486885776e0f", id="16"),
        pytest.param(
            64,
            "91738e8f4e95f91bd94446057cccce6c1c4349a0f4af76f062a49cc8d4191b26",
            id="64-default",
        ),
    ],
)
def test_pydantic_model_hasher_hash_length(
    length: int, expected: str, registry: HasherRegistry
) -> None:
    result = PydanticModelHasher().hash(Point(x=1, y=2), registry=registry, length=length)
    assert result == expected


@pydantic_available
@pytest.mark.parametrize("length", [16, 32, 64])
def test_pydantic_model_hasher_hash_length_length_is_respected(
    length: int, registry: HasherRegistry
) -> None:
    result = PydanticModelHasher().hash(Point(x=1, y=2), registry=registry, length=length)
    assert len(result) == length


@pydantic_available
def test_pydantic_model_hasher_hash_is_deterministic(registry: HasherRegistry) -> None:
    hasher = PydanticModelHasher()
    point = Point(x=1, y=2)
    assert hasher.hash(point, registry=registry) == hasher.hash(point, registry=registry)


@pydantic_available
def test_pydantic_model_hasher_hash_independent_of_field_declaration_order(
    registry: HasherRegistry,
) -> None:
    hasher = PydanticModelHasher()
    assert hasher.hash(Point(x=1, y=2), registry=registry) == hasher.hash(
        Point(y=2, x=1), registry=registry
    )


@pydantic_available
def test_pydantic_model_hasher_hash_different_values_different_hashes(
    registry: HasherRegistry,
) -> None:
    hasher = PydanticModelHasher()
    assert hasher.hash(Point(x=1, y=2), registry=registry) != hasher.hash(
        Point(x=1, y=3), registry=registry
    )


@pydantic_available
def test_pydantic_model_hasher_hash_ignore_unhashable_has_no_effect(
    registry: HasherRegistry,
) -> None:
    # When every field value is hashable, ignore_unhashable changes nothing.
    hasher = PydanticModelHasher()
    point = Point(x=1, y=2)
    assert hasher.hash(point, registry=registry) == hasher.hash(
        point, registry=registry, ignore_unhashable=True
    )


@pydantic_available
def test_pydantic_model_hasher_hash_unhashable_field_raises(registry: HasherRegistry) -> None:
    model = AnyValue(value=object())
    with pytest.raises(KeyError):
        PydanticModelHasher().hash(model, registry=registry)


@pydantic_available
def test_pydantic_model_hasher_hash_unhashable_field_ignored_warns(
    registry: HasherRegistry,
) -> None:
    model = AnyValue(value=object())
    with pytest.warns(UserWarning, match="placeholder"):
        PydanticModelHasher().hash(model, registry=registry, ignore_unhashable=True)


@pydantic_available
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_pydantic_model_hasher_hash_unhashable_field_ignored_returns_str(
    registry: HasherRegistry,
) -> None:
    model = AnyValue(value=object())
    result = PydanticModelHasher().hash(model, registry=registry, ignore_unhashable=True)
    assert isinstance(result, str)


@pydantic_available
def test_pydantic_model_hasher_hash_uses_registry() -> None:
    # Field values are hashed through the registry, so a registry that
    # cannot hash ints cannot hash a model with int fields.
    with pytest.raises(KeyError):
        PydanticModelHasher().hash(Point(x=1, y=2), registry=HasherRegistry())


@pydantic_available
def test_pydantic_model_hasher_hash_depends_on_model_class(registry: HasherRegistry) -> None:
    hasher = PydanticModelHasher()
    assert hasher.hash(Point(x=1, y=2), registry=registry) != hasher.hash(
        Point3(x=1, y=2), registry=registry
    )


@pydantic_available
def test_pydantic_model_hasher_hash_field_types_are_not_collapsed_list_and_tuple(
    registry: HasherRegistry,
) -> None:
    hasher = PydanticModelHasher()
    assert hasher.hash(AnyValue(value=[1, 2]), registry=registry) != hasher.hash(
        AnyValue(value=(1, 2)), registry=registry
    )


@pydantic_available
def test_pydantic_model_hasher_hash_field_types_are_not_collapsed_int_and_float(
    registry: HasherRegistry,
) -> None:
    hasher = PydanticModelHasher()
    assert hasher.hash(AnyValue(value=1), registry=registry) != hasher.hash(
        AnyValue(value=1.0), registry=registry
    )


@pydantic_available
def test_pydantic_model_hasher_hash_set_field_is_order_independent(
    registry: HasherRegistry,
) -> None:
    hasher = PydanticModelHasher()
    assert hasher.hash(AnyValue(value={1, 2, 3}), registry=registry) == hasher.hash(
        AnyValue(value={3, 2, 1}), registry=registry
    )


@pydantic_available
def test_pydantic_model_hasher_hash_nested_model(registry: HasherRegistry) -> None:
    hasher = PydanticModelHasher()
    assert hasher.hash(AnyValue(value=Point(x=1, y=2)), registry=registry) != hasher.hash(
        AnyValue(value=Point(x=1, y=3)), registry=registry
    )


@pydantic_available
def test_pydantic_model_hasher_hash_extra_fields_are_included(registry: HasherRegistry) -> None:
    hasher = PydanticModelHasher()
    assert hasher.hash(Open(x=1, extra=2), registry=registry) != hasher.hash(
        Open(x=1, extra=3), registry=registry
    )


@pydantic_available
def test_pydantic_model_hasher_hash_secret_error_does_not_leak_value(
    registry: HasherRegistry,
) -> None:
    creds = Creds(username="alice", password="hunter2")
    with pytest.raises(ValueError, match="SecretStr/SecretBytes") as exc_info:
        PydanticModelHasher().hash(creds, registry=registry)
    assert "hunter2" not in str(exc_info.value)


@pydantic_available
def test_pydantic_model_hasher_default_on_secret_raises_for_secret_field(
    registry: HasherRegistry,
) -> None:
    hasher = PydanticModelHasher()
    creds = Creds(username="alice", password="hunter2")
    with pytest.raises(ValueError, match="SecretStr/SecretBytes"):
        hasher.hash(creds, registry=registry)


@pydantic_available
def test_pydantic_model_hasher_on_secret_exclude_ignores_secret_value(
    registry: HasherRegistry,
) -> None:
    hasher = PydanticModelHasher(on_secret="exclude")
    a = Creds(username="alice", password="hunter2")
    b = Creds(username="alice", password="different")
    assert hasher.hash(a, registry=registry) == hasher.hash(b, registry=registry)


@pydantic_available
def test_pydantic_model_hasher_on_secret_reveal_sensitive_to_secret_value(
    registry: HasherRegistry,
) -> None:
    hasher = PydanticModelHasher(on_secret="reveal")
    a = Creds(username="alice", password="hunter2")
    b = Creds(username="alice", password="different")
    assert hasher.hash(a, registry=registry) != hasher.hash(b, registry=registry)


@pydantic_available
def test_pydantic_model_hasher_on_secret_is_configured_once_via_constructor(
    registry: HasherRegistry,
) -> None:
    exclude_hasher = PydanticModelHasher(on_secret="exclude")
    reveal_hasher = PydanticModelHasher(on_secret="reveal")
    creds = Creds(username="alice", password="hunter2")
    assert exclude_hasher.hash(creds, registry=registry) != reveal_hasher.hash(
        creds, registry=registry
    )


####################################
#     Tests for unwrap_secrets     #
####################################

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
def test_unwrap_secrets_scalar_passthrough_int() -> None:
    assert unwrap_secrets(42, on_secret="error") == 42


@pydantic_available
def test_unwrap_secrets_scalar_passthrough_str() -> None:
    assert unwrap_secrets("hello", on_secret="error") == "hello"


@pydantic_available
def test_unwrap_secrets_scalar_passthrough_none() -> None:
    assert unwrap_secrets(None, on_secret="error") is None


@pydantic_available
def test_unwrap_secrets_unhandled_on_secret() -> None:
    with pytest.raises(AssertionError):
        unwrap_secrets(SecretStr("hunter2"), on_secret="invalid")  # type: ignore[arg-type]


# --- on_secret="reveal" ---


@pydantic_available
def test_unwrap_secrets_reveal_secret_str_in_dict() -> None:
    value = {"user": "alice", "pwd": SecretStr("hunter2")}
    assert unwrap_secrets(value, on_secret="reveal") == {"user": "alice", "pwd": "hunter2"}


@pydantic_available
def test_unwrap_secrets_reveal_secret_bytes_in_dict() -> None:
    value = {"token": SecretBytes(b"raw-token")}
    assert unwrap_secrets(value, on_secret="reveal") == {"token": b"raw-token"}


@pydantic_available
def test_unwrap_secrets_reveal_secret_in_list() -> None:
    value = [SecretStr("s1"), "x", SecretStr("s2")]
    assert unwrap_secrets(value, on_secret="reveal") == ["s1", "x", "s2"]


@pydantic_available
def test_unwrap_secrets_reveal_top_level_secret_str() -> None:
    assert unwrap_secrets(SecretStr("hunter2"), on_secret="reveal") == "hunter2"


@pydantic_available
def test_unwrap_secrets_reveal_top_level_secret_bytes() -> None:
    assert unwrap_secrets(SecretBytes(b"raw-token"), on_secret="reveal") == b"raw-token"


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


#########################################
#     Tests for hash_pydantic_model     #
#########################################

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


@pydantic_available
@pytest.mark.parametrize("container", [tuple, list, set, frozenset])
def test_hash_pydantic_model_secret_in_container_raises_by_default(container: type) -> None:
    class Model(BaseModel):
        values: Any

    with pytest.raises(ValueError, match="SecretStr"):
        hash_pydantic_model(Model(values=container([SecretStr("a")])))


@pydantic_available
@pytest.mark.parametrize("container", [tuple, list, set, frozenset])
def test_hash_pydantic_model_secret_in_container_reveal_depends_on_secret(
    container: type,
) -> None:
    class Model(BaseModel):
        values: Any

    first = hash_pydantic_model(Model(values=container([SecretStr("a")])), on_secret="reveal")
    second = hash_pydantic_model(Model(values=container([SecretStr("b")])), on_secret="reveal")
    assert first != second


@pydantic_available
@pytest.mark.parametrize("container", [tuple, list, set, frozenset])
def test_hash_pydantic_model_secret_in_container_exclude_ignores_secret(
    container: type,
) -> None:
    class Model(BaseModel):
        values: Any

    first = hash_pydantic_model(Model(values=container([SecretStr("a"), "x"])), on_secret="exclude")
    second = hash_pydantic_model(
        Model(values=container([SecretStr("b"), "x"])), on_secret="exclude"
    )
    assert first == second


@pydantic_available
def test_hash_pydantic_model_set_is_order_independent() -> None:
    class Model(BaseModel):
        values: Any

    assert hash_pydantic_model(Model(values={"a", "b", "c"})) == hash_pydantic_model(
        Model(values={"c", "b", "a"})
    )


@pydantic_available
def test_unwrap_secrets_preserves_container_types_values() -> None:
    value = {"a": (1, SecretStr("x")), "b": {1, 2}, "c": frozenset({3}), "d": [1]}
    result = unwrap_secrets(value, on_secret="reveal")
    assert result == {"a": (1, "x"), "b": {1, 2}, "c": frozenset({3}), "d": [1]}


@pydantic_available
def test_unwrap_secrets_preserves_container_types_tuple() -> None:
    value = {"a": (1, SecretStr("x")), "b": {1, 2}, "c": frozenset({3}), "d": [1]}
    result = unwrap_secrets(value, on_secret="reveal")
    assert isinstance(result["a"], tuple)


@pydantic_available
def test_unwrap_secrets_preserves_container_types_set() -> None:
    value = {"a": (1, SecretStr("x")), "b": {1, 2}, "c": frozenset({3}), "d": [1]}
    result = unwrap_secrets(value, on_secret="reveal")
    assert isinstance(result["b"], set)


@pydantic_available
def test_unwrap_secrets_preserves_container_types_frozenset() -> None:
    value = {"a": (1, SecretStr("x")), "b": {1, 2}, "c": frozenset({3}), "d": [1]}
    result = unwrap_secrets(value, on_secret="reveal")
    assert isinstance(result["c"], frozenset)


@pydantic_available
def test_unwrap_secrets_leaves_other_values_unchanged() -> None:
    value = OrderedDict(a=1)
    assert unwrap_secrets(value, on_secret="error") is value


@pydantic_available
def test_hash_pydantic_model_uses_given_registry() -> None:
    with pytest.raises(KeyError):
        hash_pydantic_model(Point(x=1, y=2), registry=HasherRegistry())


@pydantic_available
def test_hash_pydantic_model_depends_on_model_class() -> None:
    assert hash_pydantic_model(Point(x=1, y=2)) != hash_pydantic_model(Point3(x=1, y=2))


@pydantic_available
def test_hash_pydantic_model_secret_bytes_reveal_matches_bytes_field_differs_by_secret() -> None:
    class Plain(BaseModel):
        token: bytes

    class Secret(BaseModel):
        token: SecretBytes

    # Different classes, so the hashes differ; the point is that revealing
    # a SecretBytes does not raise and is sensitive to the content.
    assert hash_pydantic_model(Secret(token=b"a"), on_secret="reveal") != hash_pydantic_model(
        Secret(token=b"b"), on_secret="reveal"
    )


@pydantic_available
def test_hash_pydantic_model_secret_bytes_reveal_matches_bytes_field_plain_model_is_a_str() -> None:
    class Plain(BaseModel):
        token: bytes

    class Secret(BaseModel):
        token: SecretBytes

    assert isinstance(hash_pydantic_model(Plain(token=b"a")), str)


@pydantic_available
def test_hash_pydantic_model_self_referencing_model_raises_value_error() -> None:
    model = AnyValue(value=None)
    model.value = [model]
    with pytest.raises(ValueError, match="self-referencing"):
        hash_pydantic_model(model)


@pydantic_available
def test_hash_pydantic_model_same_model_twice_is_not_a_cycle() -> None:
    inner = Point(x=1, y=2)
    assert hash_pydantic_model(AnyValue(value=[inner, inner]))
