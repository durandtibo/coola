from __future__ import annotations

import warnings
from typing import Annotated, Optional

import pytest

from coola.display import repr_pydantic_model, str_pydantic_model
from coola.display.pydantic import is_secret_annotation, secret_field_names
from coola.testing.fixtures import pydantic_available
from coola.utils.imports import is_pydantic_available

if is_pydantic_available():
    from pydantic import BaseModel, Secret, SecretBytes, SecretStr
else:
    BaseModel, Secret, SecretBytes, SecretStr = object, object, object, object


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


class OtherSecretsModel(BaseModel):
    data: SecretBytes
    wrapped: Secret[int]
    pair: tuple[SecretStr, int]
    value: int = 0


class ChildModel(BaseModel):
    api_key: SecretStr
    host: str


class ParentModel(BaseModel):
    name: str
    child: ChildModel


class ParentWithChildListModel(BaseModel):
    name: str
    children: list[ChildModel]


class ParentWithChildDictModel(BaseModel):
    name: str
    children: dict[str, ChildModel]


##########################################
#     Tests for is_secret_annotation     #
##########################################


@pydantic_available
@pytest.mark.parametrize(
    "annotation",
    [
        SecretStr,
        SecretBytes,
        Secret[int],
        Optional[SecretStr],  # noqa: UP045
        SecretStr | None,
        tuple[SecretStr, int],
        list[SecretStr],
        dict[str, list[SecretBytes]],
        Annotated[SecretStr, "meta"],
    ],
)
def test_is_secret_annotation_true(annotation: object) -> None:
    assert is_secret_annotation(annotation)


@pydantic_available
@pytest.mark.parametrize(
    "annotation",
    [str, int, None, "SecretStr", list[int], dict[str, str], int | None, Annotated[int, "meta"]],
)
def test_is_secret_annotation_false(annotation: object) -> None:
    assert not is_secret_annotation(annotation)


########################################
#     Tests for secret_field_names     #
########################################


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
def test_secret_field_names_secret_bytes_secret_and_generic() -> None:
    model = OtherSecretsModel(data=b"x", wrapped=1, pair=(SecretStr("s3cr3t"), 1))
    assert secret_field_names(model) == {"data", "wrapped", "pair"}


@pydantic_available
def test_secret_field_names_optional_secret_none() -> None:
    model = OptionalSecretModel(value=1)
    assert secret_field_names(model) == {"key"}


########################################
#     Tests for str_pydantic_model     #
########################################


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
def test_str_pydantic_model_exclude_fields_missing_field_warns() -> None:
    model = MyModel(name="alice", age=30, token=SecretStr("s3cr3t"))
    with pytest.warns(RuntimeWarning, match="does_not_exist"):
        str_pydantic_model(model, exclude_fields=["does_not_exist"])


@pydantic_available
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_str_pydantic_model_exclude_fields_missing_field_output() -> None:
    model = MyModel(name="alice", age=30, token=SecretStr("s3cr3t"))
    out = str_pydantic_model(model, exclude_fields=["does_not_exist"])
    assert out == "MyModel(age=30, name=alice, nickname=None)"


@pydantic_available
def test_str_pydantic_model_secret_bytes_secret_and_generic_excluded() -> None:
    model = OtherSecretsModel(data=b"x", wrapped=1, pair=(SecretStr("s3cr3t"), 1))
    assert str_pydantic_model(model) == "OtherSecretsModel(value=0)"


@pydantic_available
def test_str_pydantic_model_exclude_fields_secret_field_no_warning() -> None:
    model = MyModel(name="alice", age=30, token=SecretStr("s3cr3t"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = str_pydantic_model(model, exclude_fields=["token"])
    assert out == "MyModel(age=30, name=alice, nickname=None)"


@pydantic_available
def test_str_pydantic_model_exclude_fields_none_field_no_warning() -> None:
    model = MyModel(name="alice", age=30, token=SecretStr("s3cr3t"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = str_pydantic_model(model, exclude_none=True, exclude_fields=["nickname"])
    assert out == "MyModel(age=30, name=alice)"


@pydantic_available
def test_str_pydantic_model_exclude_fields_missing_field_and_valid_field_warns() -> None:
    model = MyModel(name="alice", age=30, token=SecretStr("s3cr3t"))
    with pytest.warns(RuntimeWarning, match="does_not_exist"):
        str_pydantic_model(model, exclude_fields=["does_not_exist", "nickname"])


@pydantic_available
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_str_pydantic_model_exclude_fields_missing_field_and_valid_field_output() -> None:
    model = MyModel(name="alice", age=30, token=SecretStr("s3cr3t"))
    out = str_pydantic_model(model, exclude_fields=["does_not_exist", "nickname"])
    assert out == "MyModel(age=30, name=alice)"


@pydantic_available
def test_str_pydantic_model_exclude_fields_no_warning_for_valid_field() -> None:
    model = MyModel(name="alice", age=30, token=SecretStr("s3cr3t"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = str_pydantic_model(model, exclude_fields=["nickname"])
    assert out == "MyModel(age=30, name=alice)"


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


#########################################
#     Tests for repr_pydantic_model     #
#########################################


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
def test_repr_pydantic_model_exclude_fields_missing_field_warns() -> None:
    model = MyModel(name="alice", age=30, token=SecretStr("s3cr3t"))
    with pytest.warns(RuntimeWarning, match="does_not_exist"):
        repr_pydantic_model(model, exclude_fields=["does_not_exist"])


@pydantic_available
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_repr_pydantic_model_exclude_fields_missing_field_output() -> None:
    model = MyModel(name="alice", age=30, token=SecretStr("s3cr3t"))
    out = repr_pydantic_model(model, exclude_fields=["does_not_exist"])
    assert out == "MyModel(age=30, name='alice', nickname=None)"


@pydantic_available
@pytest.mark.parametrize("sort", [True, False])
def test_str_and_repr_agree_on_field_set(sort: bool) -> None:
    model = MyModel(name="alice", age=30, token=SecretStr("s3cr3t"))
    s = str_pydantic_model(model, sort=sort)
    r = repr_pydantic_model(model, sort=sort)
    # same class name and field order, differing only in value formatting
    assert s.split("(")[0] == r.split("(")[0]


#################################################################
#     Tests for exclude_secret with nested BaseModel fields     #
#################################################################


@pydantic_available
def test_str_pydantic_model_nested_secret_excluded_by_default_key_is_excluded() -> None:
    model = ParentModel(
        name="alice", child=ChildModel(api_key=SecretStr("s3cr3t"), host="example.com")
    )
    out = str_pydantic_model(model)
    assert "api_key" not in out


@pydantic_available
def test_str_pydantic_model_nested_secret_excluded_by_default_secret_is_hidden() -> None:
    model = ParentModel(
        name="alice", child=ChildModel(api_key=SecretStr("s3cr3t"), host="example.com")
    )
    out = str_pydantic_model(model)
    assert "s3cr3t" not in out


@pydantic_available
def test_str_pydantic_model_nested_secret_excluded_by_default_output() -> None:
    model = ParentModel(
        name="alice", child=ChildModel(api_key=SecretStr("s3cr3t"), host="example.com")
    )
    out = str_pydantic_model(model)
    assert out == "ParentModel(child={'host': 'example.com'}, name=alice)"


@pydantic_available
def test_str_pydantic_model_nested_secret_included_when_exclude_secret_false_secret_is_hidden() -> (
    None
):
    model = ParentModel(
        name="alice", child=ChildModel(api_key=SecretStr("s3cr3t"), host="example.com")
    )
    out = str_pydantic_model(model, exclude_secret=False)
    assert "s3cr3t" not in out


@pydantic_available
def test_str_pydantic_model_nested_secret_included_when_exclude_secret_false_key_is_included() -> (
    None
):
    model = ParentModel(
        name="alice", child=ChildModel(api_key=SecretStr("s3cr3t"), host="example.com")
    )
    out = str_pydantic_model(model, exclude_secret=False)
    assert "api_key" in out


@pydantic_available
def test_str_pydantic_model_nested_secret_included_when_exclude_secret_false_secret_is_masked() -> (
    None
):
    model = ParentModel(
        name="alice", child=ChildModel(api_key=SecretStr("s3cr3t"), host="example.com")
    )
    out = str_pydantic_model(model, exclude_secret=False)
    assert "**********" in out


@pydantic_available
def test_str_pydantic_model_nested_secret_in_list_excluded_key_is_excluded() -> None:
    model = ParentWithChildListModel(
        name="alice",
        children=[
            ChildModel(api_key=SecretStr("s3cr3t-1"), host="a.example.com"),
            ChildModel(api_key=SecretStr("s3cr3t-2"), host="b.example.com"),
        ],
    )
    out = str_pydantic_model(model)
    assert "api_key" not in out


@pydantic_available
def test_str_pydantic_model_nested_secret_in_list_excluded_secret_is_hidden() -> None:
    model = ParentWithChildListModel(
        name="alice",
        children=[
            ChildModel(api_key=SecretStr("s3cr3t-1"), host="a.example.com"),
            ChildModel(api_key=SecretStr("s3cr3t-2"), host="b.example.com"),
        ],
    )
    out = str_pydantic_model(model)
    assert "s3cr3t" not in out


@pydantic_available
def test_str_pydantic_model_nested_secret_in_list_excluded_first_child() -> None:
    model = ParentWithChildListModel(
        name="alice",
        children=[
            ChildModel(api_key=SecretStr("s3cr3t-1"), host="a.example.com"),
            ChildModel(api_key=SecretStr("s3cr3t-2"), host="b.example.com"),
        ],
    )
    out = str_pydantic_model(model)
    assert "'host': 'a.example.com'" in out


@pydantic_available
def test_str_pydantic_model_nested_secret_in_list_excluded_second_child() -> None:
    model = ParentWithChildListModel(
        name="alice",
        children=[
            ChildModel(api_key=SecretStr("s3cr3t-1"), host="a.example.com"),
            ChildModel(api_key=SecretStr("s3cr3t-2"), host="b.example.com"),
        ],
    )
    out = str_pydantic_model(model)
    assert "'host': 'b.example.com'" in out


@pydantic_available
def test_repr_pydantic_model_nested_secret_excluded_by_default_key_is_excluded() -> None:
    model = ParentModel(
        name="alice", child=ChildModel(api_key=SecretStr("s3cr3t"), host="example.com")
    )
    out = repr_pydantic_model(model)
    assert "api_key" not in out


@pydantic_available
def test_repr_pydantic_model_nested_secret_excluded_by_default_secret_is_hidden() -> None:
    model = ParentModel(
        name="alice", child=ChildModel(api_key=SecretStr("s3cr3t"), host="example.com")
    )
    out = repr_pydantic_model(model)
    assert "s3cr3t" not in out


@pydantic_available
def test_str_pydantic_model_nested_secret_in_dict_excluded_key_is_excluded() -> None:
    model = ParentWithChildDictModel(
        name="alice",
        children={
            "a": ChildModel(api_key=SecretStr("s3cr3t-1"), host="a.example.com"),
            "b": ChildModel(api_key=SecretStr("s3cr3t-2"), host="b.example.com"),
        },
    )
    out = str_pydantic_model(model)
    assert "api_key" not in out


@pydantic_available
def test_str_pydantic_model_nested_secret_in_dict_excluded_secret_is_hidden() -> None:
    model = ParentWithChildDictModel(
        name="alice",
        children={
            "a": ChildModel(api_key=SecretStr("s3cr3t-1"), host="a.example.com"),
            "b": ChildModel(api_key=SecretStr("s3cr3t-2"), host="b.example.com"),
        },
    )
    out = str_pydantic_model(model)
    assert "s3cr3t" not in out


@pydantic_available
def test_str_pydantic_model_nested_secret_in_dict_excluded_first_child() -> None:
    model = ParentWithChildDictModel(
        name="alice",
        children={
            "a": ChildModel(api_key=SecretStr("s3cr3t-1"), host="a.example.com"),
            "b": ChildModel(api_key=SecretStr("s3cr3t-2"), host="b.example.com"),
        },
    )
    out = str_pydantic_model(model)
    assert "'host': 'a.example.com'" in out


@pydantic_available
def test_str_pydantic_model_nested_secret_in_dict_excluded_second_child() -> None:
    model = ParentWithChildDictModel(
        name="alice",
        children={
            "a": ChildModel(api_key=SecretStr("s3cr3t-1"), host="a.example.com"),
            "b": ChildModel(api_key=SecretStr("s3cr3t-2"), host="b.example.com"),
        },
    )
    out = str_pydantic_model(model)
    assert "'host': 'b.example.com'" in out
