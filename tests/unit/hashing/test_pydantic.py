from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pytest

from coola.hashing import (
    HasherRegistry,
    SecretHasher,
    get_default_registry,
    get_secret_hashers,
    hash_object,
)
from coola.hashing.pydantic import PydanticModelHasher
from coola.testing.marks import pydantic_available
from coola.utils.imports import is_pydantic_available
from tests.unit.hashing.helpers import encode_node, hash_with

if is_pydantic_available():
    from pydantic import BaseModel, ConfigDict, SecretBytes, SecretStr

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

    def make_creds(username: str, secret: str) -> Creds:
        # validate a dict so the rule S106 does not see a password passed to a keyword
        return Creds.model_validate({"username": username, "password": secret})


@pytest.fixture
def registry() -> HasherRegistry:
    return get_default_registry()


#########################################
#     Tests for PydanticModelHasher     #
#########################################


@pydantic_available
def test_pydantic_model_hasher_repr() -> None:
    assert repr(PydanticModelHasher()) == "PydanticModelHasher()"


@pydantic_available
def test_pydantic_model_hasher_str() -> None:
    assert str(PydanticModelHasher()) == "PydanticModelHasher()"


@pydantic_available
def test_pydantic_model_hasher_hash_known_value(registry: HasherRegistry) -> None:
    point = Point(x=1, y=2)
    assert (
        hash_with(PydanticModelHasher(), point, registry)
        == "235d706b42b86ee043ee2f39c80e0f22343cdcc7fc013f1163055dff91ab09db"
    )


@pydantic_available
def test_pydantic_model_hasher_hash_returns_str(registry: HasherRegistry) -> None:
    assert isinstance(hash_with(PydanticModelHasher(), Point(x=1, y=2), registry), str)


@pydantic_available
def test_pydantic_model_hasher_hash_default_length_is_64(registry: HasherRegistry) -> None:
    assert len(hash_with(PydanticModelHasher(), Point(x=1, y=2), registry)) == 64


@pydantic_available
@pytest.mark.parametrize(
    ("length", "expected"),
    [
        pytest.param(16, "235d706b42b86ee0", id="16"),
        pytest.param(
            64,
            "235d706b42b86ee043ee2f39c80e0f22343cdcc7fc013f1163055dff91ab09db",
            id="64-default",
        ),
    ],
)
def test_pydantic_model_hasher_hash_length(
    length: int, expected: str, registry: HasherRegistry
) -> None:
    result = hash_with(PydanticModelHasher(), Point(x=1, y=2), registry, length=length)
    assert result == expected


@pydantic_available
@pytest.mark.parametrize("length", [16, 32, 64])
def test_pydantic_model_hasher_hash_length_length_is_respected(
    length: int, registry: HasherRegistry
) -> None:
    result = hash_with(PydanticModelHasher(), Point(x=1, y=2), registry, length=length)
    assert len(result) == length


@pydantic_available
def test_pydantic_model_hasher_hash_is_deterministic(registry: HasherRegistry) -> None:
    hasher = PydanticModelHasher()
    point = Point(x=1, y=2)
    assert hash_with(hasher, point, registry) == hash_with(hasher, point, registry)


@pydantic_available
def test_pydantic_model_hasher_hash_independent_of_field_declaration_order(
    registry: HasherRegistry,
) -> None:
    hasher = PydanticModelHasher()
    assert hash_with(hasher, Point(x=1, y=2), registry) == hash_with(
        hasher, Point(y=2, x=1), registry
    )


@pydantic_available
def test_pydantic_model_hasher_hash_different_values_different_hashes(
    registry: HasherRegistry,
) -> None:
    hasher = PydanticModelHasher()
    assert hash_with(hasher, Point(x=1, y=2), registry) != hash_with(
        hasher, Point(x=1, y=3), registry
    )


@pydantic_available
def test_pydantic_model_hasher_hash_ignore_unhashable_has_no_effect(
    registry: HasherRegistry,
) -> None:
    # When every field value is hashable, ignore_unhashable changes nothing.
    hasher = PydanticModelHasher()
    point = Point(x=1, y=2)
    assert hash_with(hasher, point, registry) == hash_with(
        hasher, point, registry, ignore_unhashable=True
    )


@pydantic_available
def test_pydantic_model_hasher_hash_unhashable_field_raises(registry: HasherRegistry) -> None:
    model = AnyValue(value=object())
    with pytest.raises(KeyError, match="is not registered"):
        hash_with(PydanticModelHasher(), model, registry)


@pydantic_available
def test_pydantic_model_hasher_hash_unhashable_field_ignored_warns(
    registry: HasherRegistry,
) -> None:
    model = AnyValue(value=object())
    with pytest.warns(UserWarning, match="placeholder"):
        hash_with(PydanticModelHasher(), model, registry, ignore_unhashable=True)


@pydantic_available
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_pydantic_model_hasher_hash_unhashable_field_ignored_returns_str(
    registry: HasherRegistry,
) -> None:
    model = AnyValue(value=object())
    result = hash_with(PydanticModelHasher(), model, registry, ignore_unhashable=True)
    assert isinstance(result, str)


@pydantic_available
def test_pydantic_model_hasher_hash_uses_registry() -> None:
    # Field values are hashed through the registry, so a registry that
    # cannot hash ints cannot hash a model with int fields.
    with pytest.raises(KeyError, match="is not registered"):
        hash_with(PydanticModelHasher(), Point(x=1, y=2), HasherRegistry())


@pydantic_available
def test_pydantic_model_hasher_hash_depends_on_model_class(registry: HasherRegistry) -> None:
    hasher = PydanticModelHasher()
    assert hash_with(hasher, Point(x=1, y=2), registry) != hash_with(
        hasher, Point3(x=1, y=2), registry
    )


@pydantic_available
def test_pydantic_model_hasher_hash_field_types_are_not_collapsed_list_and_tuple(
    registry: HasherRegistry,
) -> None:
    hasher = PydanticModelHasher()
    assert hash_with(hasher, AnyValue(value=[1, 2]), registry) != hash_with(
        hasher, AnyValue(value=(1, 2)), registry
    )


@pydantic_available
def test_pydantic_model_hasher_hash_field_types_are_not_collapsed_int_and_float(
    registry: HasherRegistry,
) -> None:
    hasher = PydanticModelHasher()
    assert hash_with(hasher, AnyValue(value=1), registry) != hash_with(
        hasher, AnyValue(value=1.0), registry
    )


@pydantic_available
def test_pydantic_model_hasher_hash_set_field_is_order_independent(
    registry: HasherRegistry,
) -> None:
    hasher = PydanticModelHasher()
    assert hash_with(hasher, AnyValue(value={1, 2, 3}), registry) == hash_with(
        hasher, AnyValue(value={3, 2, 1}), registry
    )


@pydantic_available
def test_pydantic_model_hasher_hash_nested_model(registry: HasherRegistry) -> None:
    hasher = PydanticModelHasher()
    assert hash_with(hasher, AnyValue(value=Point(x=1, y=2)), registry) != hash_with(
        hasher, AnyValue(value=Point(x=1, y=3)), registry
    )


@pydantic_available
def test_pydantic_model_hasher_hash_extra_fields_are_included(registry: HasherRegistry) -> None:
    hasher = PydanticModelHasher()
    assert hash_with(hasher, Open(x=1, extra=2), registry) != hash_with(
        hasher, Open(x=1, extra=3), registry
    )


@pydantic_available
def test_pydantic_model_hasher_encode_node_is_unordered_with_pairs() -> None:
    node = encode_node(PydanticModelHasher(), Point(x=1, y=2))
    assert (node.ordered, node.group) == (False, 2)


@pydantic_available
def test_pydantic_model_hasher_self_referencing_model_raises() -> None:
    model = AnyValue(value=[])
    model.value.append(model)
    with pytest.raises(ValueError, match="self-referencing"):
        hash_object(model)


@pydantic_available
def test_pydantic_model_hasher_same_model_twice_is_not_a_cycle() -> None:
    point = Point(x=1, y=2)
    assert hash_object([point, point]) == hash_object([Point(x=1, y=2), Point(x=1, y=2)])


########################################
#     Tests for SecretHasher     #
########################################


def secret_registry(on_secret: Literal["reveal", "exclude", "error"]) -> HasherRegistry:
    registry = get_default_registry().copy()
    registry.register_many(get_secret_hashers(on_secret), exist_ok=True)
    return registry


@pydantic_available
def test_secret_hasher_repr() -> None:
    assert repr(SecretHasher("exclude")) == "SecretHasher(on_secret='exclude')"


@pydantic_available
def test_secret_hasher_default_on_secret_is_error() -> None:
    assert repr(SecretHasher()) == "SecretHasher(on_secret='error')"


@pydantic_available
def test_secret_hasher_invalid_on_secret_raises() -> None:
    invalid_policy: Any = "hide"
    with pytest.raises(ValueError, match="on_secret must be"):
        SecretHasher(invalid_policy)


@pydantic_available
def test_secret_hasher_encode_reveal_secret_str() -> None:
    assert SecretHasher("reveal").encode(SecretStr("abc")) == b"pydantic.SecretStr\x00abc"


@pydantic_available
def test_secret_hasher_encode_reveal_secret_bytes() -> None:
    assert SecretHasher("reveal").encode(SecretBytes(b"abc")) == b"pydantic.SecretBytes\x00abc"


@pydantic_available
def test_secret_hasher_encode_exclude_secret_str() -> None:
    assert SecretHasher("exclude").encode(SecretStr("abc")) == b"pydantic.SecretStr\x00"


@pydantic_available
def test_secret_hasher_encode_exclude_secret_bytes() -> None:
    assert SecretHasher("exclude").encode(SecretBytes(b"abc")) == b"pydantic.SecretBytes\x00"


@pydantic_available
def test_secret_hasher_encode_error_raises() -> None:
    with pytest.raises(ValueError, match="SecretStr/SecretBytes"):
        SecretHasher().encode(SecretStr("abc"))


@pydantic_available
def test_secret_hasher_encode_error_does_not_leak_the_value() -> None:
    with pytest.raises(ValueError, match="SecretStr/SecretBytes") as exc_info:
        SecretHasher().encode(SecretStr("hunter2"))
    assert "hunter2" not in str(exc_info.value)


@pydantic_available
def test_get_secret_hashers_keys() -> None:
    assert set(get_secret_hashers("reveal")) == {SecretStr, SecretBytes}


@pydantic_available
def test_get_secret_hashers_policy() -> None:
    hasher = get_secret_hashers("reveal")[SecretStr]
    assert isinstance(hasher, SecretHasher)
    assert repr(hasher) == "SecretHasher(on_secret='reveal')"


@pydantic_available
def test_get_secret_hashers_default_policy_is_error() -> None:
    hasher = get_secret_hashers()[SecretBytes]
    assert isinstance(hasher, SecretHasher)
    assert repr(hasher) == "SecretHasher(on_secret='error')"


@pydantic_available
def test_default_registry_secret_in_a_model_raises() -> None:
    with pytest.raises(ValueError, match="SecretStr/SecretBytes"):
        hash_object(make_creds("alice", "hunter2"))


@pydantic_available
def test_default_registry_secret_error_does_not_leak_the_value() -> None:
    with pytest.raises(ValueError, match="SecretStr/SecretBytes") as exc_info:
        hash_object(make_creds("alice", "hunter2"))
    assert "hunter2" not in str(exc_info.value)


@pydantic_available
@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(lambda s: {"k": s}, id="dict"),
        pytest.param(lambda s: [s], id="list"),
        pytest.param(lambda s: (s,), id="tuple"),
        pytest.param(lambda s: {s}, id="set"),
    ],
)
def test_default_registry_secret_in_a_container_raises(factory: Any) -> None:
    with pytest.raises(ValueError, match="SecretStr/SecretBytes"):
        hash_object(factory(SecretStr("hunter2")))


@pydantic_available
def test_default_registry_bare_secret_raises() -> None:
    with pytest.raises(ValueError, match="SecretStr/SecretBytes"):
        hash_object(SecretStr("hunter2"))


@pydantic_available
def test_exclude_secrets_in_models_with_different_secrets_have_the_same_hash() -> None:
    registry = secret_registry("exclude")
    a = make_creds("alice", "hunter2")
    b = make_creds("alice", "different")
    assert registry.hash(a) == registry.hash(b)


@pydantic_available
def test_exclude_secrets_keeps_the_other_fields() -> None:
    registry = secret_registry("exclude")
    a = make_creds("alice", "hunter2")
    b = make_creds("bob", "hunter2")
    assert registry.hash(a) != registry.hash(b)


@pydantic_available
def test_exclude_secrets_keeps_the_position_of_the_secret() -> None:
    registry = secret_registry("exclude")
    assert registry.hash([SecretStr("a"), 1]) != registry.hash([1, SecretStr("a")])


@pydantic_available
def test_exclude_secrets_differs_from_a_list_without_the_secret() -> None:
    registry = secret_registry("exclude")
    assert registry.hash([SecretStr("a"), 1]) != registry.hash([1])


@pydantic_available
def test_exclude_secrets_in_a_dict() -> None:
    registry = secret_registry("exclude")
    assert registry.hash({"k": SecretStr("a")}) == registry.hash({"k": SecretStr("b")})


@pydantic_available
def test_exclude_top_level_secret() -> None:
    registry = secret_registry("exclude")
    assert registry.hash(SecretStr("a")) == registry.hash(SecretStr("b"))


@pydantic_available
def test_exclude_secret_str_and_secret_bytes_differ() -> None:
    registry = secret_registry("exclude")
    assert registry.hash(SecretStr("a")) != registry.hash(SecretBytes(b"a"))


@pydantic_available
def test_exclude_secrets_in_a_nested_model() -> None:
    registry = secret_registry("exclude")
    a = Nested(name="n", creds=make_creds("alice", "one"))
    b = Nested(name="n", creds=make_creds("alice", "two"))
    assert registry.hash(a) == registry.hash(b)


@pydantic_available
def test_exclude_secrets_in_a_dataclass() -> None:
    @dataclass
    class Holder:
        secret: SecretStr

    registry = secret_registry("exclude")
    assert registry.hash(Holder(SecretStr("a"))) == registry.hash(Holder(SecretStr("b")))


@pydantic_available
def test_exclude_multiple_secret_fields() -> None:
    registry = secret_registry("exclude")
    a = CredsWithToken.model_validate({"username": "alice", "password": "a", "token": b"x"})
    b = CredsWithToken.model_validate({"username": "alice", "password": "b", "token": b"y"})
    assert registry.hash(a) == registry.hash(b)


@pydantic_available
def test_reveal_secrets_depends_on_the_value() -> None:
    registry = secret_registry("reveal")
    a = make_creds("alice", "hunter2")
    b = make_creds("alice", "different")
    assert registry.hash(a) != registry.hash(b)


@pydantic_available
def test_reveal_secrets_equal_values_have_the_same_hash() -> None:
    registry = secret_registry("reveal")
    a = make_creds("alice", "hunter2")
    b = make_creds("alice", "hunter2")
    assert registry.hash(a) == registry.hash(b)


@pydantic_available
def test_reveal_secret_bytes_depends_on_the_value() -> None:
    registry = secret_registry("reveal")
    assert registry.hash(SecretBytes(b"a")) != registry.hash(SecretBytes(b"b"))


@pydantic_available
def test_reveal_secret_differs_from_the_plain_string() -> None:
    registry = secret_registry("reveal")
    assert registry.hash(SecretStr("a")) != registry.hash("a")


@pydantic_available
def test_reveal_and_exclude_differ() -> None:
    creds = make_creds("alice", "hunter2")
    assert secret_registry("reveal").hash(creds) != secret_registry("exclude").hash(creds)


@pydantic_available
def test_reveal_secrets_in_a_container() -> None:
    registry = secret_registry("reveal")
    assert registry.hash([SecretStr("a")]) != registry.hash([SecretStr("b")])


@pydantic_available
def test_secret_policy_does_not_change_the_default_registry() -> None:
    secret_registry("exclude")
    with pytest.raises(ValueError, match="SecretStr/SecretBytes"):
        hash_object(SecretStr("a"))
