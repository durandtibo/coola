from __future__ import annotations

import pytest

from coola.summary import SummaryConfig
from coola.summary.metadata import BaseMetadataSummarizer, Field, format_fields
from coola.utils.imports import MissingPackageError

CONFIG = SummaryConfig()


class ListSummarizer(BaseMetadataSummarizer[list]):
    def _describe(self, data: list) -> list[Field]:
        return [f"length={len(data)}", "kind=list"]


class ColumnsSummarizer(BaseMetadataSummarizer[dict]):
    def _describe(self, data: dict) -> list[Field]:
        return ["kind=dict", ("columns", data.items())]


class RequiresPackageSummarizer(ListSummarizer):
    def _check_dependencies(self) -> None:
        msg = "'missing' package is required but not installed."
        raise MissingPackageError(msg, name="missing")


class BadRepr:
    def __repr__(self) -> str:
        msg = "broken"
        raise RuntimeError(msg)


###################################
#     Tests for format_fields     #
###################################


def test_format_fields_empty() -> None:
    assert format_fields([], 5) == ""


def test_format_fields_single() -> None:
    assert format_fields([("a", "int64")], 5) == "a: int64"


def test_format_fields_two() -> None:
    assert format_fields([("a", "int64"), ("b", "float64")], 5) == "a: int64, b: float64"


def test_format_fields_at_the_limit() -> None:
    fields = [(f"c{i}", "int") for i in range(3)]
    assert format_fields(fields, 3) == "c0: int, c1: int, c2: int"


def test_format_fields_above_the_limit() -> None:
    fields = [(f"c{i}", "int") for i in range(5)]
    assert format_fields(fields, 2) == "c0: int, c1: int, ... (+3 more)"


def test_format_fields_limit_zero() -> None:
    assert format_fields([("a", "int"), ("b", "int")], 0) == "... (+2 more)"


def test_format_fields_no_limit() -> None:
    fields = [(f"c{i}", "int") for i in range(20)]
    assert format_fields(fields, None).endswith("c19: int")


def test_format_fields_accepts_an_iterator() -> None:
    assert format_fields(iter([("a", "int")]), 5) == "a: int"


############################################
#     Tests for BaseMetadataSummarizer     #
############################################


def test_base_metadata_summarizer_is_abstract() -> None:
    with pytest.raises(TypeError, match="abstract"):
        BaseMetadataSummarizer()  # ty: ignore[call-non-callable]


def test_base_metadata_summarizer_summarize() -> None:
    assert ListSummarizer().summarize([1, 2], CONFIG, compact=False) == (
        "<class 'list'> | length=2 | kind=list"
    )


def test_base_metadata_summarizer_summarize_compact_is_ignored() -> None:
    assert ListSummarizer().summarize([1, 2], CONFIG, compact=True) == (
        "<class 'list'> | length=2 | kind=list"
    )


def test_base_metadata_summarizer_summarize_show_data() -> None:
    assert ListSummarizer().summarize([1, 2], SummaryConfig(show_data=True), compact=False) == (
        "[1, 2]"
    )


def test_base_metadata_summarizer_summarize_show_data_failing_repr() -> None:
    result = ListSummarizer().summarize(BadRepr(), SummaryConfig(show_data=True), compact=False)  # ty: ignore[invalid-argument-type]
    assert result.startswith("<BadRepr object at 0x")


def test_base_metadata_summarizer_summarize_fields_are_limited_by_max_items() -> None:
    data = {"a": "int", "b": "float", "c": "str"}
    assert ColumnsSummarizer().summarize(data, SummaryConfig(max_items=2), compact=False) == (
        "<class 'dict'> | kind=dict | columns=a: int, b: float, ... (+1 more)"
    )


def test_base_metadata_summarizer_summarize_fields_without_limit() -> None:
    data = {"a": "int", "b": "float", "c": "str"}
    assert ColumnsSummarizer().summarize(data, SummaryConfig(max_items=None), compact=False) == (
        "<class 'dict'> | kind=dict | columns=a: int, b: float, c: str"
    )


def test_base_metadata_summarizer_check_dependencies_error_is_raised() -> None:
    with pytest.raises(MissingPackageError, match="'missing' package is required"):
        RequiresPackageSummarizer()


def test_base_metadata_summarizer_equal_same_type() -> None:
    assert ListSummarizer().equal(ListSummarizer())


def test_base_metadata_summarizer_equal_false_for_child_class() -> None:
    class Child(ListSummarizer): ...

    assert not ListSummarizer().equal(Child())
