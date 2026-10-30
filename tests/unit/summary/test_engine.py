from __future__ import annotations

from collections import Counter, OrderedDict, deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, is_dataclass
from typing import Any, NamedTuple

import pytest

from coola.summary import (
    BaseSummarizer,
    DataclassSummarizer,
    DefaultSummarizer,
    Node,
    SequenceSummarizer,
    SetSummarizer,
    SummarizerRegistry,
    SummaryConfig,
    get_default_registry,
)
from coola.summary.engine import CompactRepr, render
from tests.helpers import DEEP_NESTING_DEPTH, nest

REGISTRY = get_default_registry()


def run(data: Any, **kwargs: Any) -> str:
    return render(data, REGISTRY, SummaryConfig(**kwargs))


class Coord(NamedTuple):
    x: int
    y: int


@dataclass
class Point:
    x: int
    y: int


@dataclass
class Credentials:
    name: str
    token: str = field(default="", repr=False)


@dataclass
class Holder:
    items: list


class SpyMapping(Mapping):
    r"""Count the items that are taken from a mapping."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.taken = 0

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[int]:
        for i in range(self.size):
            self.taken += 1
            yield i

    def __getitem__(self, key: int) -> int:
        return key


class SpyList(list):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size
        self.taken = 0

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[int]:
        for i in range(self.size):
            self.taken += 1
            yield i


class CountingStr:
    calls = 0

    def __str__(self) -> str:
        type(self).calls += 1
        return "counted"


class BadStr:
    def __str__(self) -> str:
        msg = "secret message"
        raise RuntimeError(msg)


class BadRepr:
    def __repr__(self) -> str:
        msg = "secret message"
        raise RuntimeError(msg)


class BadLen(list):
    def __len__(self) -> int:
        msg = "secret message"
        raise RuntimeError(msg)


@dataclass
class BadField:
    x: int
    y: int

    def __getattribute__(self, name: str) -> Any:
        if name == "y":
            msg = "secret message"
            raise RuntimeError(msg)
        return super().__getattribute__(name)


class BadIter(list):
    def __iter__(self) -> Iterator[int]:
        yield 1
        msg = "secret message"
        raise RuntimeError(msg)


class ReturnsInt(BaseSummarizer[object]):
    def summarize(self, data: object, config: SummaryConfig, *, compact: bool) -> Any:  # noqa: ARG002
        return 1


##################################
#     Tests for render: text     #
##################################


def test_render_scalar() -> None:
    assert run(42) == "<class 'int'> 42"


def test_render_scalar_with_max_depth_0() -> None:
    assert run(42, max_depth=0) == "42"


def test_render_empty_list() -> None:
    assert run([]) == "<class 'list'> (length=0)"


def test_render_empty_dict() -> None:
    assert run({}) == "<class 'dict'> (length=0)"


def test_render_list_max_depth_1() -> None:
    assert run([1, "a", [1, 2]]) == ("<class 'list'> (length=3)\n  (0): 1\n  (1): a\n  (2): [1, 2]")


@pytest.mark.parametrize(
    ("max_depth", "expected"),
    [
        (0, "{'a': [1, 2], 'b': {'c': 3}}"),
        (1, "<class 'dict'> (length=2)\n  (a): [1, 2]\n  (b): {'c': 3}"),
        (
            2,
            (
                "<class 'dict'> (length=2)\n"
                "  (a): <class 'list'> (length=2)\n"
                "    (0): 1\n"
                "    (1): 2\n"
                "  (b): <class 'dict'> (length=1)\n"
                "    (c): 3"
            ),
        ),
        (
            3,
            (
                "<class 'dict'> (length=2)\n"
                "  (a): <class 'list'> (length=2)\n"
                "    (0): <class 'int'> 1\n"
                "    (1): <class 'int'> 2\n"
                "  (b): <class 'dict'> (length=1)\n"
                "    (c): <class 'int'> 3"
            ),
        ),
    ],
)
def test_render_nested_dict_by_depth(max_depth: int, expected: str) -> None:
    assert run({"a": [1, 2], "b": {"c": 3}}, max_depth=max_depth) == expected


def test_render_whole_nested_structure() -> None:
    data = {"a": [1, 2, 3, 4, 5, 6, 7], "b": {"c": {1, 2}, "d": "x\ny"}}
    assert run(data, max_depth=3) == (
        "<class 'dict'> (length=2)\n"
        "  (a): <class 'list'> (length=7)\n"
        "    (0): <class 'int'> 1\n"
        "    (1): <class 'int'> 2\n"
        "    (2): <class 'int'> 3\n"
        "    (3): <class 'int'> 4\n"
        "    (4): <class 'int'> 5\n"
        "    ... (+2 more)\n"
        "  (b): <class 'dict'> (length=2)\n"
        "    (c): <class 'set'> (length=2)\n"
        "      (0): 1\n"
        "      (1): 2\n"
        "    (d): <class 'str'> x\n"
        "      y"
    )


def test_render_tuple() -> None:
    assert run((1, 2)) == "<class 'tuple'> (length=2)\n  (0): 1\n  (1): 2"


def test_render_set() -> None:
    assert run({3, 1, 2}) == "<class 'set'> (length=3)\n  (0): 1\n  (1): 2\n  (2): 3"


def test_render_dataclass() -> None:
    assert run(Point(1, 2)) == f"{Point} (fields=2)\n  (x): 1\n  (y): 2"


def test_render_dataclass_does_not_print_the_field_without_repr() -> None:
    assert "secret" not in run(Credentials("a", "secret"), max_depth=3)


def test_render_dataclass_at_the_display_depth() -> None:
    assert run(Holder([Point(1, 2)])) == f"{Holder} (fields=1)\n  (items): [Point(x=1, y=2)]"


def test_render_nested_dataclass() -> None:
    assert run(Holder([Point(1, 2)]), max_depth=3) == (
        f"{Holder} (fields=1)\n"
        "  (items): <class 'list'> (length=1)\n"
        f"    (0): {Point} (fields=2)\n"
        "      (x): 1\n"
        "      (y): 2"
    )


def test_render_indentation_is_regular_at_every_depth() -> None:
    assert run(nest(1, 4), max_depth=10) == (
        "<class 'list'> (length=1)\n"
        "  (0): <class 'list'> (length=1)\n"
        "    (0): <class 'list'> (length=1)\n"
        "      (0): <class 'list'> (length=1)\n"
        "        (0): <class 'int'> 1"
    )


def test_render_text_with_newline_is_aligned() -> None:
    assert run(["a\nb"]) == "<class 'list'> (length=1)\n  (0): a\n    b"


def test_render_root_text_with_newline_is_not_indented() -> None:
    assert run("a\nb") == "<class 'str'> a\nb"


def test_render_key_with_newline_is_aligned() -> None:
    assert run({"a\nb": 1}) == "<class 'dict'> (length=1)\n  (a\n    b): 1"


def test_render_non_string_keys() -> None:
    assert run({1: "a", (2, 3): "b"}) == ("<class 'dict'> (length=2)\n  (1): a\n  ((2, 3)): b")


def test_render_ordered_dict() -> None:
    assert run(OrderedDict(a=1)) == "<class 'collections.OrderedDict'> (length=1)\n  (a): 1"


def test_render_named_tuple_uses_the_field_names() -> None:
    assert run(Coord(1, 2)) == f"{Coord} (length=2)\n  (x): 1\n  (y): 2"


####################################
#     Tests for render: limits     #
####################################


@pytest.mark.parametrize(
    ("max_items", "expected"),
    [
        (0, "<class 'list'> (length=3)\n  ... (+3 more)"),
        (1, "<class 'list'> (length=3)\n  (0): 0\n  ... (+2 more)"),
        (2, "<class 'list'> (length=3)\n  (0): 0\n  (1): 1\n  ... (+1 more)"),
        (3, "<class 'list'> (length=3)\n  (0): 0\n  (1): 1\n  (2): 2"),
        (4, "<class 'list'> (length=3)\n  (0): 0\n  (1): 1\n  (2): 2"),
        (None, "<class 'list'> (length=3)\n  (0): 0\n  (1): 1\n  (2): 2"),
    ],
)
def test_render_max_items(max_items: int | None, expected: str) -> None:
    assert run([0, 1, 2], max_items=max_items) == expected


def test_render_hidden_count_is_formatted() -> None:
    assert run(list(range(1005)), max_items=1).endswith("... (+1,004 more)")


def test_render_hidden_items_of_a_dict() -> None:
    assert run({"a": 1, "b": 2}, max_items=1) == (
        "<class 'dict'> (length=2)\n  (a): 1\n  ... (+1 more)"
    )


def test_render_hidden_items_of_a_set_are_the_largest() -> None:
    assert run({1, 2, 3}, max_items=1) == "<class 'set'> (length=3)\n  (0): 1\n  ... (+2 more)"


def test_render_hidden_items_of_a_nested_node() -> None:
    assert run([[1, 2, 3]], max_depth=2, max_items=2) == (
        "<class 'list'> (length=1)\n"
        "  (0): <class 'list'> (length=3)\n"
        "    (0): 1\n"
        "    (1): 2\n"
        "    ... (+1 more)"
    )


def test_render_max_characters_cuts_a_leaf() -> None:
    assert run("abcdef", max_characters=3) == "<cl..."


def test_render_max_characters_cuts_the_text_of_a_leaf_child() -> None:
    assert run(["abcdef"], max_characters=3) == "<class 'list'> (length=1)\n  (0): abc..."


def test_render_max_characters_cuts_a_label() -> None:
    assert run({"abcdef": 1}, max_characters=3) == "<class 'dict'> (length=1)\n  (abc...): 1"


def test_render_max_characters_zero() -> None:
    assert run("abc", max_characters=0) == "..."


def test_render_max_characters_none_does_not_cut() -> None:
    assert run("a" * 2000, max_characters=None) == "<class 'str'> " + "a" * 2000


def test_render_default_max_characters_cuts_at_1000() -> None:
    assert run("a" * 2000) == ("<class 'str'> " + "a" * 2000)[:1000] + "..."


def test_render_max_characters_shortens_a_string_of_the_compact_form() -> None:
    assert run([["abcdefghijkl"]], max_characters=10) == (
        "<class 'list'> (length=1)\n  (0): ['ab...jkl']"
    )


def test_render_max_depth_larger_than_the_data() -> None:
    assert run([1], max_depth=100) == "<class 'list'> (length=1)\n  (0): <class 'int'> 1"


def test_render_no_marker_when_nothing_is_hidden() -> None:
    assert "..." not in run(list(range(5)), max_items=None, max_characters=None)


###################################
#     Tests for render: depth     #
###################################


def test_render_very_deep_structure_does_not_raise() -> None:
    assert run(nest(1, 5000), max_depth=10**6, max_items=1).count("\n") == 5000


def test_render_deep_structure_above_the_display_depth_is_on_one_line() -> None:
    assert run(nest(1, DEEP_NESTING_DEPTH), max_depth=2, max_items=1).count("\n") == 2


################################
#     Tests for the cycles     #
################################


def test_render_cycle_through_a_list() -> None:
    data: list = [1]
    data.append(data)
    assert run(data, max_depth=3) == (
        "<class 'list'> (length=2)\n  (0): <class 'int'> 1\n  (1): <cycle>"
    )


def test_render_cycle_through_a_dict() -> None:
    data: dict = {}
    data["self"] = data
    assert run(data, max_depth=3) == "<class 'dict'> (length=1)\n  (self): <cycle>"


def test_render_cycle_through_a_tuple_in_a_list() -> None:
    data: list = []
    data.append((data,))
    assert run(data, max_depth=4) == (
        "<class 'list'> (length=1)\n  (0): <class 'tuple'> (length=1)\n    (0): <cycle>"
    )


def test_render_cycle_through_a_dataclass() -> None:
    data = Holder([])
    data.items.append(data)
    assert run(data, max_depth=4) == (
        f"{Holder} (fields=1)\n  (items): <class 'list'> (length=1)\n    (0): <cycle>"
    )


def test_render_cycle_at_the_display_depth_is_not_followed() -> None:
    data: list = [1]
    data.append(data)
    assert run(data) == "<class 'list'> (length=2)\n  (0): 1\n  (1): [1, [1, [1, [...]]]]"


def test_render_shared_reference_is_not_a_cycle() -> None:
    shared = [1]
    assert run([shared, shared], max_depth=2) == (
        "<class 'list'> (length=2)\n"
        "  (0): <class 'list'> (length=1)\n"
        "    (0): 1\n"
        "  (1): <class 'list'> (length=1)\n"
        "    (0): 1"
    )


def test_render_summary_of_a_cycle_is_deterministic() -> None:
    data: list = []
    data.append(data)
    assert run(data, max_depth=5) == run(data, max_depth=5)


######################################
#     Tests for the bounded work     #
######################################


def test_render_takes_at_most_max_items_from_a_sequence() -> None:
    data = SpyList(10**6)
    run(data, max_items=3)
    assert data.taken == 3


def test_render_takes_at_most_max_items_from_a_mapping() -> None:
    data = SpyMapping(10**6)
    run(data, max_items=3)
    assert data.taken == 3


def test_render_compact_form_does_not_convert_the_children_to_a_string() -> None:
    CountingStr.calls = 0
    run([[CountingStr() for _ in range(3)]], max_depth=1)
    assert CountingStr.calls == 0


def test_render_compact_form_of_a_huge_list_is_bounded() -> None:
    assert run([list(range(2_000_000))]) == (
        "<class 'list'> (length=1)\n  (0): [0, 1, 2, 3, 4, ...]"
    )


def test_render_iterator_is_a_leaf_and_is_not_consumed() -> None:
    iterator = iter([1, 2, 3])
    run([iterator], max_depth=2)
    assert list(iterator) == [1, 2, 3]


def test_render_generator_is_a_leaf() -> None:
    generator = (i for i in range(3))
    assert run(generator).startswith("<class 'generator'> <generator object")


##################################
#     Tests for the totality     #
##################################


def test_render_failing_len_gives_a_placeholder() -> None:
    assert run(BadLen()) == f"{BadLen} <summary failed: RuntimeError>"


def test_render_failing_len_does_not_break_the_siblings() -> None:
    assert run([BadLen(), 1]) == (
        f"<class 'list'> (length=2)\n  (0): {BadLen} <summary failed: RuntimeError>\n  (1): 1"
    )


def test_render_failing_str_of_a_leaf_is_replaced_by_a_placeholder() -> None:
    assert "<BadStr object at 0x" in run([BadStr()])


def test_render_failing_attribute_of_a_dataclass_field() -> None:
    assert run(BadField(1, 2)) == (
        f"{BadField} (fields=2)\n  (x): 1\n  <summary failed: RuntimeError>\n  ... (+1 more)"
    )


def test_render_str_that_calls_summarize() -> None:
    class Reentrant:
        def __str__(self) -> str:
            return run([1])

    assert run([Reentrant()]) == (
        "<class 'list'> (length=1)\n  (0): <class 'list'> (length=1)\n      (0): 1"
    )


def test_render_failing_children_iterator() -> None:
    assert run(BadIter([1, 2, 3])) == (
        f"{BadIter} (length=3)\n  (0): 1\n  <summary failed: RuntimeError>\n  ... (+2 more)"
    )


def test_render_failing_repr_in_the_compact_form() -> None:
    assert run([BadRepr()]).startswith("<class 'list'> (length=1)\n  (0): <BadRepr object at 0x")


def test_render_failing_compact_form_of_a_container() -> None:
    assert run([BadLen()]) == (
        f"<class 'list'> (length=1)\n  (0): {BadLen} <summary failed: RuntimeError>"
    )


def test_render_exception_message_is_not_in_the_text() -> None:
    assert "secret message" not in run(BadLen())


def test_render_failing_summarizer_return_type() -> None:
    registry = SummarizerRegistry({object: ReturnsInt()})
    assert render(1, registry, SummaryConfig()) == ("<class 'int'> <summary failed: TypeError>")


def test_render_unregistered_type_raises() -> None:
    with pytest.raises(KeyError, match="int"):
        render(1, SummarizerRegistry(), SummaryConfig())


############################################
#     Tests for registries and plugins     #
############################################


def test_render_with_a_custom_registry() -> None:
    registry = SummarizerRegistry({object: DefaultSummarizer(), list: SequenceSummarizer()})
    assert render([{"a": 1}], registry, SummaryConfig()) == (
        "<class 'list'> (length=1)\n  (0): {'a': 1}"
    )


def test_render_with_a_registry_that_has_no_container_summarizer() -> None:
    registry = SummarizerRegistry({object: DefaultSummarizer()})
    assert render([1], registry, SummaryConfig()) == "<class 'list'> [1]"


def test_render_plugin_that_returns_a_str() -> None:
    class Plugin:
        def summarize(self, data: Any, config: SummaryConfig, *, compact: bool) -> str:  # noqa: ARG002
            return "plugin"

    registry = SummarizerRegistry({object: Plugin()})
    assert render(1, registry, SummaryConfig()) == "plugin"


def test_render_plugin_that_returns_a_node() -> None:
    class Plugin:
        def summarize(self, data: Any, config: SummaryConfig, *, compact: bool) -> Node:  # noqa: ARG002
            return Node("tree", [("left", 1), ("right", 2)], 2)

    registry = SummarizerRegistry({object: Plugin()})
    assert render(object(), registry, SummaryConfig()) == "tree\n  (left): 1\n  (right): 2"


def test_render_plugin_node_with_a_wrong_total() -> None:
    class Plugin:
        def summarize(self, data: Any, config: SummaryConfig, *, compact: bool) -> Node:  # noqa: ARG002
            return Node("tree", [("a", 1)], 3)

    registry = SummarizerRegistry({object: Plugin()})
    assert render(object(), registry, SummaryConfig()) == "tree\n  (a): 1\n  ... (+2 more)"


def test_render_summarizer_receives_the_compact_flag() -> None:
    class Plugin:
        def summarize(self, data: Any, config: SummaryConfig, *, compact: bool) -> str:  # noqa: ARG002
            return f"compact={compact}"

    registry = SummarizerRegistry(
        {object: DefaultSummarizer(), list: SequenceSummarizer(), int: Plugin()}
    )
    assert render([1], registry, SummaryConfig()) == (
        "<class 'list'> (length=1)\n  (0): compact=True"
    )


def test_render_does_not_modify_the_data() -> None:
    data = {"a": [1, 2, {3}]}
    run(data, max_depth=3)
    assert data == {"a": [1, 2, {3}]}


def test_render_is_deterministic() -> None:
    data = {"a": [1, {"b": {3, 1, 2}}]}
    assert run(data, max_depth=4) == run(data, max_depth=4)


def test_render_dataclass_fallback_is_used_with_a_registry_that_has_it() -> None:
    registry = SummarizerRegistry({object: DefaultSummarizer()})
    registry.register_fallback(is_dataclass, DataclassSummarizer())
    assert (
        render(Point(1, 2), registry, SummaryConfig()) == f"{Point} (fields=2)\n  (x): 1\n  (y): 2"
    )


def test_render_set_summarizer_with_a_registry() -> None:
    registry = SummarizerRegistry({object: DefaultSummarizer(), set: SetSummarizer()})
    assert (
        render({2, 1}, registry, SummaryConfig()) == "<class 'set'> (length=2)\n  (0): 1\n  (1): 2"
    )


#################################
#     Tests for CompactRepr     #
#################################


def compact(data: Any, **kwargs: Any) -> str:
    return CompactRepr(SummaryConfig(**kwargs)).repr(data)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (list(range(10)), "[0, 1, 2, ...]"),
        (tuple(range(10)), "(0, 1, 2, ...)"),
        (deque(range(10)), "deque([0, 1, 2, ...])"),
        ({i: i for i in range(10)}, "{0: 0, 1: 1, 2: 2, ...}"),
        (set(range(10)), "{0, 1, 2, ...}"),
        (frozenset(range(10)), "frozenset({0, 1, 2, ...})"),
    ],
)
def test_compact_repr_container_is_cut_at_max_items(data: Any, expected: str) -> None:
    assert compact(data, max_items=3) == expected


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ([1, 2], "[1, 2]"),
        ((1,), "(1,)"),
        ((), "()"),
        ([], "[]"),
        ({}, "{}"),
        (set(), "set()"),
        (frozenset(), "frozenset()"),
        ({"a": 1}, "{'a': 1}"),
    ],
)
def test_compact_repr_small_container(data: Any, expected: str) -> None:
    assert compact(data) == expected


def test_compact_repr_dict_keeps_the_insertion_order() -> None:
    assert compact({"b": 1, "a": 2}) == "{'b': 1, 'a': 2}"


def test_compact_repr_set_shows_the_smallest_items() -> None:
    assert compact({5, 4, 3, 2, 1, 0}, max_items=2) == "{0, 1, ...}"


def test_compact_repr_set_of_strings_is_sorted() -> None:
    assert compact({"b", "c", "a"}) == "{'a', 'b', 'c'}"


def test_compact_repr_frozenset_shows_the_smallest_items() -> None:
    assert compact(frozenset({3, 2, 1}), max_items=2) == "frozenset({1, 2, ...})"


def test_compact_repr_nested_containers_are_cut_at_3_levels() -> None:
    assert compact([[[[1]]]]) == "[[[[...]]]]"


def test_compact_repr_nested_dict_is_cut_at_3_levels() -> None:
    assert compact({"a": {"b": {"c": {"d": 1}}}}) == "{'a': {'b': {'c': {...}}}}"


def test_compact_repr_nested_set_is_cut_at_3_levels() -> None:
    assert compact([[[{1}]]]) == "[[[{...}]]]"


def test_compact_repr_nested_frozenset_is_cut_at_3_levels() -> None:
    assert compact([[[frozenset({1})]]]) == "[[[frozenset({...})]]]"


def test_compact_repr_string_is_shortened_in_the_middle() -> None:
    assert compact(["abcdefghijklmnopqrstuvwxyz"], max_characters=10) == "['ab...xyz']"


def test_compact_repr_max_items_none_shows_all_items() -> None:
    assert compact(list(range(20)), max_items=None).endswith("18, 19]")


def test_compact_repr_max_characters_none_does_not_shorten_a_string() -> None:
    assert compact(["a" * 100], max_characters=None) == f"['{'a' * 100}']"


def test_compact_repr_max_items_zero() -> None:
    assert compact([1, 2], max_items=0) == "[...]"


def test_compact_repr_subclass_is_shown_with_its_repr() -> None:
    assert compact(Counter("aab")) == "Counter({'a': 2, 'b': 1})"


def test_compact_repr_ordered_dict_is_shown_with_its_repr() -> None:
    assert compact(OrderedDict(a=1)) == "OrderedDict({'a': 1})"


def test_compact_repr_other_type_is_cut_at_max_characters() -> None:
    assert compact([Counter(range(100))], max_characters=10) == "[Counter({0...]"


def test_compact_repr_dataclass() -> None:
    assert compact(Point(1, 2)) == "Point(x=1, y=2)"


def test_compact_repr_failing_repr() -> None:
    assert compact(BadRepr()).startswith("<BadRepr object at 0x")


def test_compact_repr_is_bounded_for_a_huge_list() -> None:
    assert compact(list(range(2_000_000)), max_items=3) == "[0, 1, 2, ...]"
