from __future__ import annotations

from dataclasses import InitVar, dataclass, field

import pytest

from coola.summary import DataclassSummarizer, SummaryConfig

CONFIG = SummaryConfig()


@dataclass
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class Frozen:
    x: int


@dataclass(slots=True)
class Slotted:
    x: int


@dataclass
class Empty:
    pass


@dataclass
class WithInitVar:
    x: int
    scale: InitVar[int] = 1


@dataclass
class Outer:
    name: str
    point: Point


@dataclass
class WithMissingAttribute:
    x: int = field(init=False)


class Child(Point): ...


#########################################
#     Tests for DataclassSummarizer     #
#########################################


def test_dataclass_summarizer_summarize() -> None:
    node = DataclassSummarizer().summarize(Point(1.0, 2.0), CONFIG, compact=False)
    assert (node.header, list(node.children), node.total) == (
        f"{Point} (fields=2)",
        [("x", 1.0), ("y", 2.0)],
        2,
    )


def test_dataclass_summarizer_summarize_empty() -> None:
    node = DataclassSummarizer().summarize(Empty(), CONFIG, compact=False)
    assert (node.header, list(node.children), node.total) == (f"{Empty} (fields=0)", [], 0)


def test_dataclass_summarizer_summarize_frozen() -> None:
    node = DataclassSummarizer().summarize(Frozen(1), CONFIG, compact=False)
    assert list(node.children) == [("x", 1)]


def test_dataclass_summarizer_summarize_slots() -> None:
    node = DataclassSummarizer().summarize(Slotted(1), CONFIG, compact=False)
    assert list(node.children) == [("x", 1)]


def test_dataclass_summarizer_summarize_init_var_is_not_a_field() -> None:
    node = DataclassSummarizer().summarize(WithInitVar(1, 2), CONFIG, compact=False)
    assert (list(node.children), node.total) == ([("x", 1)], 1)


def test_dataclass_summarizer_summarize_nested_value_is_not_expanded() -> None:
    node = DataclassSummarizer().summarize(Outer("a", Point(1.0, 2.0)), CONFIG, compact=False)
    assert list(node.children) == [("name", "a"), ("point", Point(1.0, 2.0))]


def test_dataclass_summarizer_summarize_subclass() -> None:
    node = DataclassSummarizer().summarize(Child(1.0, 2.0), CONFIG, compact=False)
    assert node.header == f"{Child} (fields=2)"


def test_dataclass_summarizer_summarize_dataclass_class() -> None:
    node = DataclassSummarizer().summarize(Point(1.0, 2.0), CONFIG, compact=True)
    assert node.total == 2


def test_dataclass_summarizer_summarize_children_are_lazy() -> None:
    node = DataclassSummarizer().summarize(WithMissingAttribute(), CONFIG, compact=False)
    assert node.total == 1


def test_dataclass_summarizer_summarize_missing_attribute_raises_when_read() -> None:
    node = DataclassSummarizer().summarize(WithMissingAttribute(), CONFIG, compact=False)
    with pytest.raises(AttributeError, match="x"):
        list(node.children)


def test_dataclass_summarizer_summarize_does_not_modify_the_data() -> None:
    data = Point(1.0, 2.0)
    list(DataclassSummarizer().summarize(data, CONFIG, compact=False).children)
    assert data == Point(1.0, 2.0)
