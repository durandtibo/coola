from __future__ import annotations

import pytest

from coola.summary import SummarizerRegistry
from coola.summary.metadata import (
    MAX_FIELDS,
    BaseFieldsSummarizer,
    BaseMetadataSummarizer,
    format_fields,
)
from coola.utils.imports import MissingPackageError


class ListSummarizer(BaseMetadataSummarizer[list]):
    def _describe(self, data: list) -> list[str]:
        return [f"length={len(data)}", "kind=list"]


class RequiresPackageSummarizer(ListSummarizer):
    def _check_dependencies(self) -> None:
        msg = "'missing' package is required but not installed."
        raise MissingPackageError(msg, name="missing")


@pytest.fixture
def registry() -> SummarizerRegistry:
    return SummarizerRegistry()


###################################
#     Tests for format_fields     #
###################################


def test_format_fields_empty() -> None:
    assert format_fields([]) == ""


def test_format_fields() -> None:
    assert format_fields([("a", "int64"), ("b", "float64")]) == "a: int64, b: float64"


def test_format_fields_at_the_limit() -> None:
    fields = [(f"c{i}", "int") for i in range(3)]
    assert format_fields(fields, max_fields=3) == "c0: int, c1: int, c2: int"


def test_format_fields_above_the_limit() -> None:
    fields = [(f"c{i}", "int") for i in range(5)]
    assert format_fields(fields, max_fields=3) == "c0: int, c1: int, c2: int, ... (+2 more)"


def test_format_fields_default_limit_is_ten() -> None:
    fields = [(f"c{i}", "int") for i in range(11)]
    assert format_fields(fields).endswith("c9: int, ... (+1 more)")


def test_format_fields_accepts_an_iterator() -> None:
    assert format_fields(iter([("a", 1)])) == "a: 1"


############################################
#     Tests for BaseMetadataSummarizer     #
############################################


def test_base_metadata_summarizer_is_abstract() -> None:
    with pytest.raises(TypeError, match="abstract"):
        BaseMetadataSummarizer()  # type: ignore[abstract]


def test_base_metadata_summarizer_summarize(registry: SummarizerRegistry) -> None:
    assert ListSummarizer().summarize([1, 2], registry) == "<class 'list'> | length=2 | kind=list"


def test_base_metadata_summarizer_summarize_show_data(registry: SummarizerRegistry) -> None:
    assert ListSummarizer(show_data=True).summarize([1, 2], registry) == "[1, 2]"


def test_base_metadata_summarizer_repr_default() -> None:
    assert repr(ListSummarizer()) == "ListSummarizer(show_data=False)"


def test_base_metadata_summarizer_repr_show_data() -> None:
    assert repr(ListSummarizer(show_data=True)) == "ListSummarizer(show_data=True)"


def test_base_metadata_summarizer_show_data_is_a_bool() -> None:
    assert ListSummarizer(show_data=1)._show_data is True  # type: ignore[arg-type]


def test_base_metadata_summarizer_equal_equal() -> None:
    assert ListSummarizer().equal(ListSummarizer())


def test_base_metadata_summarizer_equal_different_show_data() -> None:
    assert not ListSummarizer().equal(ListSummarizer(show_data=True))


def test_base_metadata_summarizer_equal_different_type() -> None:
    assert not ListSummarizer().equal(1)


def test_base_metadata_summarizer_equal_false_for_child_class() -> None:
    class Child(ListSummarizer): ...

    assert not ListSummarizer().equal(Child())


def test_base_metadata_summarizer_check_dependencies_error_is_raised() -> None:
    with pytest.raises(MissingPackageError, match="'missing' package is required"):
        RequiresPackageSummarizer()


def test_base_metadata_summarizer_depth_is_ignored(registry: SummarizerRegistry) -> None:
    summarizer = ListSummarizer()
    assert summarizer.summarize([1], registry, depth=7, max_depth=9) == summarizer.summarize(
        [1], registry
    )


########################################
#     Tests for BaseFieldsSummarizer   #
########################################


class DictFieldsSummarizer(BaseFieldsSummarizer[dict]):
    def _describe(self, data: dict) -> list[str]:
        return [f"fields={self._format_fields(data.items())}"]


FIELDS = {f"c{i}": "int" for i in range(12)}


def test_fields_summarizer_default_max_fields() -> None:
    assert DictFieldsSummarizer()._max_fields == MAX_FIELDS


def test_fields_summarizer_default_max_fields_truncates_the_fields(
    registry: SummarizerRegistry,
) -> None:
    out = DictFieldsSummarizer().summarize(FIELDS, registry)
    assert out.endswith("c9: int, ... (+2 more)")


def test_fields_summarizer_default_max_fields_hides_the_other_fields(
    registry: SummarizerRegistry,
) -> None:
    out = DictFieldsSummarizer().summarize(FIELDS, registry)
    assert "c10" not in out


def test_fields_summarizer_max_fields(registry: SummarizerRegistry) -> None:
    out = DictFieldsSummarizer(max_fields=3).summarize(FIELDS, registry)
    assert out == "<class 'dict'> | fields=c0: int, c1: int, c2: int, ... (+9 more)"


def test_fields_summarizer_max_fields_larger_than_the_fields_no_more_marker(
    registry: SummarizerRegistry,
) -> None:
    out = DictFieldsSummarizer(max_fields=100).summarize(FIELDS, registry)
    assert "more" not in out


def test_fields_summarizer_max_fields_larger_than_the_fields_last_field(
    registry: SummarizerRegistry,
) -> None:
    out = DictFieldsSummarizer(max_fields=100).summarize(FIELDS, registry)
    assert "c11: int" in out


def test_fields_summarizer_max_fields_zero(registry: SummarizerRegistry) -> None:
    out = DictFieldsSummarizer(max_fields=0).summarize(FIELDS, registry)
    assert out == "<class 'dict'> | fields=... (+12 more)"


def test_fields_summarizer_negative_max_fields_raises() -> None:
    with pytest.raises(ValueError, match="max_fields must be non-negative"):
        DictFieldsSummarizer(max_fields=-1)


def test_fields_summarizer_repr_default() -> None:
    assert repr(DictFieldsSummarizer()) == "DictFieldsSummarizer(show_data=False)"


def test_fields_summarizer_repr_with_max_fields() -> None:
    assert repr(DictFieldsSummarizer(max_fields=3)) == (
        "DictFieldsSummarizer(show_data=False, max_fields=3)"
    )


def test_fields_summarizer_equal_true() -> None:
    assert DictFieldsSummarizer(max_fields=3).equal(DictFieldsSummarizer(max_fields=3))


def test_fields_summarizer_equal_false_different_max_fields() -> None:
    assert not DictFieldsSummarizer(max_fields=3).equal(DictFieldsSummarizer(max_fields=4))


def test_fields_summarizer_equal_false_different_show_data() -> None:
    assert not DictFieldsSummarizer(show_data=True).equal(DictFieldsSummarizer())


def test_fields_summarizer_equal_false_different_type() -> None:
    assert not DictFieldsSummarizer().equal(ListSummarizer())


def test_fields_summarizer_show_data(registry: SummarizerRegistry) -> None:
    assert DictFieldsSummarizer(show_data=True).summarize({"a": 1}, registry) == "{'a': 1}"


class BadRepr:
    def __repr__(self) -> str:
        msg = "broken __repr__"
        raise ZeroDivisionError(msg)


def test_base_metadata_summarizer_summarize_show_data_failing_repr(
    registry: SummarizerRegistry,
) -> None:
    assert (
        ListSummarizer(show_data=True)
        .summarize(BadRepr(), registry)
        .startswith("<BadRepr object at 0x")
    )


def test_base_metadata_summarizer_summarize_show_data_truncates(
    registry: SummarizerRegistry,
) -> None:
    assert ListSummarizer(show_data=True).summarize(["x" * 2000], registry) == (
        "['" + "x" * 998 + "..."
    )


def test_base_metadata_summarizer_summarize_show_data_at_the_limit(
    registry: SummarizerRegistry,
) -> None:
    assert (
        ListSummarizer(show_data=True).summarize(["x" * 996], registry) == "['" + "x" * 996 + "']"
    )
