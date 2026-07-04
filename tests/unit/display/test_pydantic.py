from __future__ import annotations

import pytest

from coola.display import repr_pydantic_model, str_pydantic_model
from coola.display.pydantic import secret_field_names
from coola.testing.fixtures import pydantic_available
from coola.utils.imports import is_pydantic_available

if is_pydantic_available():
    from pydantic import BaseModel, SecretStr
else:
    BaseModel, SecretStr = None, None


class MyModel(BaseModel):
    name: str
    age: int
    token: SecretStr
    nickname: str | None = None


class NoSecretModel(BaseModel):
    a: int
    b: str


class OptionalSecretModel(BaseModel):
    key: SecretStr | None = None
    value: int = 0


##########################################
#     Tests for secret_field_names      #
##########################################


@pydantic_available
def test_secret_field_names_basic() -> None:
    model = MyModel(name="alice", age=30, token=SecretStr("s3cr3t"))
    assert secret_field_names(model) == {"token"}


@pydantic_available
def test_secret_field_names_no_secret() -> None:
    model = NoSecretModel(a=1, b="x")
    assert secret_field_names(model) == set()


@pydantic_available
def test_secret_field_names_optional_secret() -> None:
    model = OptionalSecretModel(key=SecretStr("s3cr3t"), value=1)
    assert secret_field_names(model) == {"key"}


@pydantic_available
def test_secret_field_names_optional_secret_none() -> None:
    model = OptionalSecretModel(value=1)
    assert secret_field_names(model) == {"key"}


##########################################
#     Tests for str_pydantic_model      #
##########################################


@pydantic_available
def test_str_pydantic_model_default() -> None:
    model = MyModel(name="alice", age=30, token=SecretStr("s3cr3t"))
    assert str_pydantic_model(model) == "MyModel(age=30, name=alice, nickname=None)"


@pydantic_available
def test_str_pydantic_model_exclude_none_false() -> None:
    model = MyModel(name="alice", age=30, token=SecretStr("s3cr3t"))
    assert (
        str_pydantic_model(model, exclude_none=False)
        == "MyModel(age=30, name=alice, nickname=None)"
    )


@pydantic_available
def test_str_pydantic_model_exclude_none_true() -> None:
    model = MyModel(name="alice", age=30, token=SecretStr("s3cr3t"))
    assert str_pydantic_model(model, exclude_none=True) == "MyModel(age=30, name=alice)"


@pydantic_available
def test_str_pydantic_model_exclude_secret_false() -> None:
    model = MyModel(name="alice", age=30, token=SecretStr("s3cr3t"))
    assert (
        str_pydantic_model(model, exclude_secret=False)
        == "MyModel(age=30, name=alice, nickname=None, token=**********)"
    )


@pydantic_available
def test_str_pydantic_model_sort_false() -> None:
    model = MyModel(name="alice", age=30, token=SecretStr("s3cr3t"))
    assert str_pydantic_model(model, sort=False) == "MyModel(name=alice, age=30, nickname=None)"


@pydantic_available
def test_str_pydantic_model_exclude_fields() -> None:
    model = MyModel(name="alice", age=30, token=SecretStr("s3cr3t"))
    assert str_pydantic_model(model, exclude_fields=["nickname"]) == "MyModel(age=30, name=alice)"


@pydantic_available
def test_str_pydantic_model_exclude_fields_missing_field() -> None:
    model = MyModel(name="alice", age=30, token=SecretStr("s3cr3t"))
    assert (
        str_pydantic_model(model, exclude_fields=["does_not_exist"])
        == "MyModel(age=30, name=alice, nickname=None)"
    )


@pydantic_available
def test_str_pydantic_model_exclude_fields_empty_list() -> None:
    model = MyModel(name="alice", age=30, token=SecretStr("s3cr3t"))
    assert (
        str_pydantic_model(model, exclude_fields=[]) == "MyModel(age=30, name=alice, nickname=None)"
    )


@pydantic_available
def test_str_pydantic_model_no_secret_fields() -> None:
    model = NoSecretModel(a=1, b="x")
    assert str_pydantic_model(model) == "NoSecretModel(a=1, b=x)"


###########################################
#     Tests for repr_pydantic_model      #
###########################################


@pydantic_available
def test_repr_pydantic_model_default() -> None:
    model = MyModel(name="alice", age=30, token=SecretStr("s3cr3t"))
    assert repr_pydantic_model(model) == "MyModel(age=30, name='alice', nickname=None)"


@pydantic_available
def test_repr_pydantic_model_exclude_none_true() -> None:
    model = MyModel(name="alice", age=30, token=SecretStr("s3cr3t"))
    assert repr_pydantic_model(model, exclude_none=True) == "MyModel(age=30, name='alice')"


@pydantic_available
def test_repr_pydantic_model_exclude_secret_false() -> None:
    model = MyModel(name="alice", age=30, token=SecretStr("s3cr3t"))
    assert (
        repr_pydantic_model(model, exclude_secret=False)
        == "MyModel(age=30, name='alice', nickname=None, token=SecretStr('**********'))"
    )


@pydantic_available
def test_repr_pydantic_model_exclude_fields() -> None:
    model = MyModel(name="alice", age=30, token=SecretStr("s3cr3t"))
    assert (
        repr_pydantic_model(model, exclude_fields=["nickname"]) == "MyModel(age=30, name='alice')"
    )


@pydantic_available
@pytest.mark.parametrize("sort", [True, False])
def test_str_and_repr_agree_on_field_set(sort: bool) -> None:
    model = MyModel(name="alice", age=30, token=SecretStr("s3cr3t"))
    s = str_pydantic_model(model, sort=sort)
    r = repr_pydantic_model(model, sort=sort)
    # same class name and field order, differing only in value formatting
    assert s.split("(")[0] == r.split("(")[0]
