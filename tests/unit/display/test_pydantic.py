from __future__ import annotations

import warnings
from collections import OrderedDict
from typing import Annotated, Optional

import pytest

from coola.display import repr_pydantic_model, str_pydantic_model
from coola.display.pydantic import (
    _NestedModel,
    _Options,
    _shown,
    is_secret_annotation,
    secret_field_names,
)
from coola.testing.marks import pydantic_available
from coola.utils.imports import is_pydantic_available

if is_pydantic_available():
    from pydantic import (
        BaseModel,
        ConfigDict,
        Field,
        Secret,
        SecretBytes,
        SecretStr,
        computed_field,
        field_serializer,
    )

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

    class ParentWithChildTupleModel(BaseModel):
        children: tuple[ChildModel, ...]

    class LeafModel(BaseModel):
        model_config = ConfigDict(frozen=True)

        name: str
        note: str | None = None

    class ParentWithLeafCollectionsModel(BaseModel):
        leaves_set: set[LeafModel]
        leaves_frozenset: frozenset[LeafModel]

    class UnsortedLeafModel(BaseModel):
        z: int
        a: int

    class ParentWithUnsortedLeafModel(BaseModel):
        leaf: UnsortedLeafModel

    class ParentWithLeafModel(BaseModel):
        leaf: LeafModel
        note: str | None = None

    class HiddenFieldModel(BaseModel):
        shown: int
        hidden: int = Field(default=1, repr=False)

    class ExtraModel(BaseModel):
        model_config = ConfigDict(extra="allow")

        a: int

    class SerializerModel(BaseModel):
        value: int

        @field_serializer("value")
        def serialize_value(self, value: int) -> int:
            msg = f"the serializer must not run: {value}"
            raise RuntimeError(msg)

    class ComputedModel(BaseModel):
        value: int

        @computed_field
        @property
        def double(self) -> int:
            return self.value * 2

    SECRET_ANNOTATIONS = [
        SecretStr,
        SecretBytes,
        Secret[int],
        Optional[SecretStr],  # noqa: UP045
        SecretStr | None,
        tuple[SecretStr, int],
        list[SecretStr],
        dict[str, list[SecretBytes]],
        Annotated[SecretStr, "meta"],
    ]
else:  # pragma: no cover
    SECRET_ANNOTATIONS = []


##########################################
#     Tests for is_secret_annotation     #
##########################################


@pydantic_available
@pytest.mark.parametrize("annotation", SECRET_ANNOTATIONS)
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
    assert out == "ParentModel(child=ChildModel(host=example.com), name=alice)"


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
    assert "ChildModel(host='a.example.com')" in out


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
    assert "ChildModel(host='b.example.com')" in out


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
    assert "ChildModel(host='a.example.com')" in out


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
    assert "ChildModel(host='b.example.com')" in out


#####################################################
#     Tests for nested models (str and repr)        #
#####################################################


def _child(host: str = "example.com") -> ChildModel:
    return ChildModel(api_key=SecretStr("s3cr3t"), host=host)


@pydantic_available
def test_repr_pydantic_model_nested_model_keeps_its_name() -> None:
    model = ParentModel(name="alice", child=_child())
    assert (
        repr_pydantic_model(model)
        == "ParentModel(child=ChildModel(host='example.com'), name='alice')"
    )


@pydantic_available
def test_str_pydantic_model_nested_model_in_tuple() -> None:
    model = ParentWithChildTupleModel(children=(_child("a"), _child("b")))
    assert str_pydantic_model(model) == (
        "ParentWithChildTupleModel(children=(ChildModel(host='a'), ChildModel(host='b')))"
    )


@pydantic_available
def test_str_pydantic_model_nested_model_in_set() -> None:
    model = ParentWithLeafCollectionsModel(leaves_set={LeafModel(name="a")}, leaves_frozenset=set())
    assert str_pydantic_model(model) == (
        "ParentWithLeafCollectionsModel(leaves_frozenset=frozenset(), "
        "leaves_set={LeafModel(name='a', note=None)})"
    )


@pydantic_available
def test_str_pydantic_model_nested_model_in_frozenset() -> None:
    model = ParentWithLeafCollectionsModel(leaves_set=set(), leaves_frozenset={LeafModel(name="a")})
    assert str_pydantic_model(model) == (
        "ParentWithLeafCollectionsModel(leaves_frozenset=frozenset({LeafModel(name='a', note=None)}), "
        "leaves_set=set())"
    )


@pydantic_available
def test_str_pydantic_model_nested_model_sort_true() -> None:
    model = ParentWithUnsortedLeafModel(leaf=UnsortedLeafModel(z=1, a=2))
    assert (
        str_pydantic_model(model) == "ParentWithUnsortedLeafModel(leaf=UnsortedLeafModel(a=2, z=1))"
    )


@pydantic_available
def test_str_pydantic_model_nested_model_sort_false() -> None:
    model = ParentWithUnsortedLeafModel(leaf=UnsortedLeafModel(z=1, a=2))
    assert (
        str_pydantic_model(model, sort=False)
        == "ParentWithUnsortedLeafModel(leaf=UnsortedLeafModel(z=1, a=2))"
    )


@pydantic_available
def test_str_pydantic_model_nested_model_exclude_none_false() -> None:
    model = ParentWithLeafModel(leaf=LeafModel(name="a"))
    assert (
        str_pydantic_model(model)
        == "ParentWithLeafModel(leaf=LeafModel(name=a, note=None), note=None)"
    )


@pydantic_available
def test_str_pydantic_model_nested_model_exclude_none_true() -> None:
    model = ParentWithLeafModel(leaf=LeafModel(name="a"))
    assert (
        str_pydantic_model(model, exclude_none=True)
        == "ParentWithLeafModel(leaf=LeafModel(name=a))"
    )


@pydantic_available
def test_str_pydantic_model_exclude_fields_does_not_apply_to_nested_model() -> None:
    model = ParentWithLeafModel(leaf=LeafModel(name="a"))
    assert (
        str_pydantic_model(model, exclude_fields=["note"])
        == "ParentWithLeafModel(leaf=LeafModel(name=a, note=None))"
    )


@pydantic_available
def test_str_pydantic_model_whole_output_with_nested_secrets() -> None:
    model = ParentWithChildListModel(name="alice", children=[_child("a"), _child("b")])
    assert str_pydantic_model(model, exclude_secret=False) == (
        "ParentWithChildListModel(children=[ChildModel(api_key=SecretStr('**********'), host='a'), "
        "ChildModel(api_key=SecretStr('**********'), host='b')], name=alice)"
    )


@pydantic_available
def test_str_pydantic_model_whole_output_with_nested_secrets_excluded() -> None:
    model = ParentWithChildDictModel(name="alice", children={"a": _child("a"), "b": _child("b")})
    assert str_pydantic_model(model) == (
        "ParentWithChildDictModel(children={'a': ChildModel(host='a'), 'b': ChildModel(host='b')}, "
        "name=alice)"
    )


@pydantic_available
def test_str_pydantic_model_does_not_modify_the_model() -> None:
    model = ParentModel(name="alice", child=_child())
    str_pydantic_model(model)
    assert model == ParentModel(name="alice", child=_child())


@pydantic_available
def test_str_pydantic_model_is_deterministic() -> None:
    model = ParentModel(name="alice", child=_child())
    assert str_pydantic_model(model) == str_pydantic_model(model)


#####################################################
#     Tests for the fields shown by the formatters  #
#####################################################


@pydantic_available
def test_str_pydantic_model_field_with_repr_false_is_hidden() -> None:
    assert str_pydantic_model(HiddenFieldModel(shown=1)) == "HiddenFieldModel(shown=1)"


@pydantic_available
def test_str_pydantic_model_extra_values_are_shown() -> None:
    assert str_pydantic_model(ExtraModel(a=1, b=2)) == "ExtraModel(a=1, b=2)"


@pydantic_available
def test_str_pydantic_model_exclude_extra_field_no_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = str_pydantic_model(ExtraModel(a=1, b=2), exclude_fields=["b"])
    assert out == "ExtraModel(a=1)"


@pydantic_available
def test_str_pydantic_model_does_not_run_field_serializer() -> None:
    assert str_pydantic_model(SerializerModel(value=3)) == "SerializerModel(value=3)"


@pydantic_available
def test_str_pydantic_model_computed_field_is_not_shown() -> None:
    assert str_pydantic_model(ComputedModel(value=3)) == "ComputedModel(value=3)"


############################
#     Tests for _shown     #
############################

OPTIONS = _Options(sort=True, exclude_none=False, exclude_secret=True)


@pydantic_available
@pytest.mark.parametrize("value", [1, "abc", None, 2.5, object])
def test_shown_plain_value_is_unchanged(value: object) -> None:
    assert _shown(value, OPTIONS) is value


@pydantic_available
def test_shown_model_is_nested_model() -> None:
    assert isinstance(_shown(LeafModel(name="a"), OPTIONS), _NestedModel)


@pydantic_available
@pytest.mark.parametrize("value", [[1, 2], (1, 2), {1, 2}, frozenset({1, 2}), {"a": 1}])
def test_shown_container_without_model_is_equal(value: object) -> None:
    assert _shown(value, OPTIONS) == value


@pydantic_available
@pytest.mark.parametrize("value", [[1], (1,), {1}, frozenset({1}), {"a": 1}])
def test_shown_container_keeps_its_type(value: object) -> None:
    assert type(_shown(value, OPTIONS)) is type(value)


@pydantic_available
def test_shown_list_of_lists_of_models() -> None:
    assert repr(_shown([[LeafModel(name="a")]], OPTIONS)) == "[[LeafModel(name='a', note=None)]]"


@pydantic_available
def test_shown_dict_subclass_is_unchanged() -> None:
    value = OrderedDict(a=1)
    assert _shown(value, OPTIONS) is value


##################################
#     Tests for _NestedModel     #
##################################


@pydantic_available
def test_nested_model_str() -> None:
    assert str(_NestedModel(LeafModel(name="a"), OPTIONS)) == "LeafModel(name=a, note=None)"


@pydantic_available
def test_nested_model_repr() -> None:
    assert repr(_NestedModel(LeafModel(name="a"), OPTIONS)) == "LeafModel(name='a', note=None)"


@pydantic_available
def test_nested_model_equal() -> None:
    assert _NestedModel(LeafModel(name="a"), OPTIONS) == _NestedModel(LeafModel(name="a"), OPTIONS)


@pydantic_available
def test_nested_model_different_model_is_not_equal() -> None:
    assert _NestedModel(LeafModel(name="a"), OPTIONS) != _NestedModel(LeafModel(name="b"), OPTIONS)


@pydantic_available
def test_nested_model_not_equal_to_a_model() -> None:
    assert _NestedModel(LeafModel(name="a"), OPTIONS) != LeafModel(name="a")


@pydantic_available
def test_nested_model_hash_of_equal_models_is_equal() -> None:
    assert hash(_NestedModel(LeafModel(name="a"), OPTIONS)) == hash(
        _NestedModel(LeafModel(name="a"), OPTIONS)
    )
