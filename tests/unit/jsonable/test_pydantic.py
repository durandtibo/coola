from __future__ import annotations

from unittest.mock import patch

import pytest

from coola.jsonable import ConvertTransformer, get_default_registry, to_jsonable
from coola.jsonable.pydantic import get_default_transformers
from coola.testing.marks import pydantic_available
from coola.utils.imports import is_pydantic_available

if is_pydantic_available():
    from pydantic import BaseModel, ConfigDict, SecretBytes, SecretStr

    class Tags(BaseModel):
        tags: set[str]

    class User(BaseModel):
        name: str
        hidden: SecretStr

    class Outer(BaseModel):
        user: User
        tags: Tags

    class Opaque(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        value: object


@pydantic_available
def test_get_default_transformers_keys() -> None:
    assert list(get_default_transformers()) == [BaseModel, SecretStr, SecretBytes]


def test_get_default_transformers_without_pydantic() -> None:
    with patch("coola.jsonable.pydantic.is_pydantic_available", lambda: False):
        assert get_default_transformers() == {}


@pydantic_available
def test_to_jsonable_model() -> None:
    assert to_jsonable(User(name="a", hidden="pw")) == {"name": "a", "hidden": "**********"}


@pydantic_available
def test_to_jsonable_model_sorts_the_sets() -> None:
    assert to_jsonable(Tags(tags={"d", "a", "c", "b"})) == {"tags": ["a", "b", "c", "d"]}


@pydantic_available
def test_to_jsonable_nested_models() -> None:
    assert to_jsonable(Outer(user=User(name="a", hidden="pw"), tags=Tags(tags={"b", "a"}))) == {
        "user": {"name": "a", "hidden": "**********"},
        "tags": {"tags": ["a", "b"]},
    }


@pydantic_available
def test_to_jsonable_model_in_a_container() -> None:
    assert to_jsonable([Tags(tags={"a"})]) == [{"tags": ["a"]}]


@pydantic_available
def test_to_jsonable_model_with_an_unsupported_field_raises() -> None:
    with pytest.raises(TypeError, match=r"Cannot convert an object of type object"):
        to_jsonable(Opaque(value=object()))


@pydantic_available
def test_to_jsonable_secret_str() -> None:
    assert to_jsonable(SecretStr("pw")) == "**********"


@pydantic_available
def test_to_jsonable_secret_bytes() -> None:
    assert to_jsonable(SecretBytes(b"pw")) == "**********"


@pydantic_available
def test_to_jsonable_model_class_can_register_its_json_mode() -> None:
    registry = get_default_registry().copy()
    registry.register(Tags, ConvertTransformer(lambda model: model.model_dump(mode="json")))
    assert to_jsonable(Tags(tags={"a"}), registry=registry) == {"tags": ["a"]}
