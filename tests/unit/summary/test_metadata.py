from __future__ import annotations

import pytest

from coola.summary import SummarizerRegistry
from coola.summary.metadata import BaseMetadataSummarizer, format_fields
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


def test_base_metadata_summarizer_repr() -> None:
    assert repr(ListSummarizer()) == "ListSummarizer(show_data=False)"
    assert repr(ListSummarizer(show_data=True)) == "ListSummarizer(show_data=True)"


def test_base_metadata_summarizer_show_data_is_a_bool() -> None:
    assert ListSummarizer(show_data=1)._show_data is True  # type: ignore[arg-type]


def test_base_metadata_summarizer_equal() -> None:
    assert ListSummarizer().equal(ListSummarizer())
    assert not ListSummarizer().equal(ListSummarizer(show_data=True))
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
