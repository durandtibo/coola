from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from coola.summary import DataclassSummarizer, SummarizerRegistry, get_default_registry, summarize
from coola.summary.default import DefaultSummarizer


@dataclass
class Point:
    x: float
    y: float


@dataclass
class Hidden:
    x: int
    cache: int = field(default=0, compare=False)


@dataclass
class Empty:
    pass


@dataclass
class Outer:
    name: str
    point: Point


@dataclass
class Wide:
    a: int = 1
    b: int = 2
    c: int = 3


class BadStr:
    def __repr__(self) -> str:
        msg = "broken"
        raise ZeroDivisionError(msg)

    __str__ = __repr__


@pytest.fixture
def registry() -> SummarizerRegistry:
    return get_default_registry()


#######################################
#     Tests for DataclassSummarizer   #
#######################################


def test_dataclass_summarizer_repr() -> None:
    assert repr(DataclassSummarizer()) == "DataclassSummarizer(max_items=5, num_spaces=2)"


def test_dataclass_summarizer_equal_true() -> None:
    assert DataclassSummarizer().equal(DataclassSummarizer())


def test_dataclass_summarizer_equal_false_different_max_items() -> None:
    assert not DataclassSummarizer().equal(DataclassSummarizer(max_items=1))


def test_dataclass_summarizer_equal_false_different_num_spaces() -> None:
    assert not DataclassSummarizer().equal(DataclassSummarizer(num_spaces=4))


def test_dataclass_summarizer_equal_false_different_type() -> None:
    assert not DataclassSummarizer().equal(DefaultSummarizer())


def test_dataclass_summarizer_summarize(registry: SummarizerRegistry) -> None:
    assert DataclassSummarizer().summarize(Point(1.0, 2.0), registry) == (
        f"{Point} (fields=2)\n  (x): 1.0\n  (y): 2.0"
    )


def test_dataclass_summarizer_summarize_includes_compare_false_fields(
    registry: SummarizerRegistry,
) -> None:
    assert "(cache): 0" in DataclassSummarizer().summarize(Hidden(1), registry)


def test_dataclass_summarizer_summarize_empty(registry: SummarizerRegistry) -> None:
    assert DataclassSummarizer().summarize(Empty(), registry) == f"{Empty} (fields=0)"


def test_dataclass_summarizer_summarize_max_items_zero(registry: SummarizerRegistry) -> None:
    assert DataclassSummarizer(max_items=0).summarize(Point(1.0, 2.0), registry) == (
        f"{Point} (fields=2) ..."
    )


def test_dataclass_summarizer_summarize_max_items_truncates(registry: SummarizerRegistry) -> None:
    assert DataclassSummarizer(max_items=2).summarize(Wide(), registry) == (
        f"{Wide} (fields=3)\n  (a): 1\n  (b): 2\n  ..."
    )


def test_dataclass_summarizer_summarize_max_items_negative_shows_all(
    registry: SummarizerRegistry,
) -> None:
    assert DataclassSummarizer(max_items=-1).summarize(Wide(), registry).endswith("(c): 3")


def test_dataclass_summarizer_summarize_num_spaces(registry: SummarizerRegistry) -> None:
    assert DataclassSummarizer(num_spaces=4).summarize(Point(1.0, 2.0), registry) == (
        f"{Point} (fields=2)\n    (x): 1.0\n    (y): 2.0"
    )


def test_dataclass_summarizer_summarize_depth_limit(registry: SummarizerRegistry) -> None:
    assert DataclassSummarizer().summarize(Point(1.0, 2.0), registry, max_depth=0) == (
        "Point(x=1.0, y=2.0)"
    )


def test_dataclass_summarizer_summarize_depth_limit_failing_str(
    registry: SummarizerRegistry,
) -> None:
    @dataclass
    class Holder:
        value: object

    result = DataclassSummarizer().summarize(Holder(BadStr()), registry, max_depth=0)
    assert isinstance(result, str)


def test_dataclass_summarizer_summarize_nested_max_depth_one(
    registry: SummarizerRegistry,
) -> None:
    assert DataclassSummarizer().summarize(Outer("a", Point(1.0, 2.0)), registry) == (
        f"{Outer} (fields=2)\n  (name): a\n  (point): Point(x=1.0, y=2.0)"
    )


def test_dataclass_summarizer_summarize_nested_max_depth_two(
    registry: SummarizerRegistry,
) -> None:
    assert DataclassSummarizer().summarize(Outer("a", Point(1.0, 2.0)), registry, max_depth=2) == (
        f"{Outer} (fields=2)\n  (name): <class 'str'> a\n"
        f"  (point): {Point} (fields=2)\n      (x): 1.0\n      (y): 2.0"
    )


def test_dataclass_summarizer_summarize_does_not_mutate_the_input(
    registry: SummarizerRegistry,
) -> None:
    point = Point(1.0, 2.0)
    DataclassSummarizer().summarize(point, registry)
    assert point == Point(1.0, 2.0)


######################################################
#     Tests for the dataclasses in the registry      #
######################################################


def test_summarize_dataclass_uses_the_default_dataclass_summarizer() -> None:
    assert summarize(Point(1.0, 2.0)) == f"{Point} (fields=2)\n  (x): 1.0\n  (y): 2.0"


def test_summarizer_registry_find_summarizer_dataclass(registry: SummarizerRegistry) -> None:
    assert isinstance(registry.find_summarizer(Point), DataclassSummarizer)


def test_summarizer_registry_find_summarizer_dataclass_registered_type_wins() -> None:
    summarizer = DefaultSummarizer(max_characters=3)
    registry = SummarizerRegistry({object: DefaultSummarizer(), Point: summarizer})
    assert registry.find_summarizer(Point) is summarizer


def test_summarizer_registry_find_summarizer_dataclass_without_marker_uses_object() -> None:
    default = DefaultSummarizer()
    registry = SummarizerRegistry({object: default})
    assert registry.find_summarizer(Point) is default


def test_summarizer_registry_find_summarizer_not_a_dataclass_uses_object() -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_summarizer(int), DefaultSummarizer)
