from collections import OrderedDict, defaultdict

from pytest import mark, raises

from coola import Summarizer, summarizer_options
from coola.formatters import (
    DefaultFormatter,
    MappingFormatter,
    SequenceFormatter,
    SetFormatter,
)

######################################
#     Tests for DefaultFormatter     #
######################################


def test_default_formatter_str() -> None:
    assert str(DefaultFormatter()).startswith("DefaultFormatter(")


def test_default_formatter_clone_max_characters_10() -> None:
    formatter = DefaultFormatter(max_characters=10)
    formatter_cloned = formatter.clone()
    formatter.set_max_characters(20)
    assert formatter is not formatter_cloned
    assert formatter.equal(DefaultFormatter(max_characters=20))
    assert formatter_cloned.equal(DefaultFormatter(max_characters=10))


def test_default_formatter_equal_true() -> None:
    assert DefaultFormatter().equal(DefaultFormatter())


def test_default_formatter_equal_false_different_max_characters() -> None:
    assert not DefaultFormatter().equal(DefaultFormatter(max_characters=10))


def test_default_formatter_equal_false_different_type() -> None:
    assert not DefaultFormatter().equal(42)


def test_default_formatter_format_str() -> None:
    assert DefaultFormatter().format(Summarizer(), "abc") == "<class 'str'> abc"


def test_default_formatter_format_int() -> None:
    assert DefaultFormatter().format(Summarizer(), 1) == "<class 'int'> 1"


def test_default_formatter_format_float() -> None:
    assert DefaultFormatter().format(Summarizer(), 1.2) == "<class 'float'> 1.2"


@mark.parametrize("max_characters", (-1, -10))
def test_default_formatter_format_max_characters_neg(max_characters: int) -> None:
    assert (
        DefaultFormatter(max_characters=max_characters).format(
            Summarizer(), "abcdefghijklmnopqrstuvwxyz"
        )
        == "<class 'str'> abcdefghijklmnopqrstuvwxyz"
    )


def test_default_formatter_format_max_characters_0() -> None:
    assert (
        DefaultFormatter(max_characters=0).format(Summarizer(), "abcdefghijklmnopqrstuvwxyz")
        == "<class 'str'> ..."
    )


def test_default_formatter_format_max_characters_10() -> None:
    assert (
        DefaultFormatter(max_characters=10).format(Summarizer(), "abcdefghijklmnopqrstuvwxyz")
        == "<class 'str'> abcdefghij..."
    )


def test_default_formatter_format_max_characters_10_with_indent() -> None:
    assert (
        DefaultFormatter(max_characters=10).format(Summarizer(), "abc\tdefghijklmnopqrstuvwxyz")
        == "<class 'str'> abc\tdefghi..."
    )


def test_default_formatter_format_max_characters_26() -> None:
    assert (
        DefaultFormatter(max_characters=26).format(Summarizer(), "abcdefghijklmnopqrstuvwxyz")
        == "<class 'str'> abcdefghijklmnopqrstuvwxyz"
    )


def test_default_formatter_format_max_characters_100() -> None:
    assert (
        DefaultFormatter(max_characters=100).format(Summarizer(), "abcdefghijklmnopqrstuvwxyz")
        == "<class 'str'> abcdefghijklmnopqrstuvwxyz"
    )


@mark.parametrize("max_depth", (0, -1, -2))
def test_default_formatter_format_max_depth_0(max_depth: int) -> None:
    assert (
        DefaultFormatter().format(Summarizer(), "abcdefghijklmnopqrstuvwxyz", max_depth=max_depth)
        == "abcdefghijklmnopqrstuvwxyz"
    )


def test_default_formatter_format_max_depth_0_max_characters_10() -> None:
    assert (
        DefaultFormatter(max_characters=10).format(
            Summarizer(), "abcdefghijklmnopqrstuvwxyz", max_depth=0
        )
        == "abcdefghij..."
    )


def test_default_formatter_load_state_dict() -> None:
    formatter = DefaultFormatter()
    formatter.load_state_dict({"max_characters": 10})
    assert formatter.equal(DefaultFormatter(max_characters=10))


def test_default_formatter_state_dict() -> None:
    assert DefaultFormatter().state_dict() == {"max_characters": -1}


def test_default_formatter_get_max_characters() -> None:
    assert DefaultFormatter().get_max_characters() == -1


@mark.parametrize("max_characters", (-1, 0, 1, 10))
def test_default_formatter_set_max_characters_int(max_characters: int) -> None:
    formatter = DefaultFormatter()
    assert formatter.get_max_characters() == -1
    formatter.set_max_characters(max_characters)
    assert formatter.get_max_characters() == max_characters


def test_default_formatter_set_max_characters_incorrect_type() -> None:
    formatter = DefaultFormatter()
    with raises(TypeError, match="Incorrect type for max_characters. Expected int value"):
        formatter.set_max_characters(4.2)


######################################
#     Tests for MappingFormatter     #
######################################


def test_mapping_formatter_str() -> None:
    assert str(MappingFormatter()).startswith("MappingFormatter(")


def test_mapping_formatter_clone_max_items_10() -> None:
    formatter = MappingFormatter(max_items=10, num_spaces=4)
    formatter_cloned = formatter.clone()
    formatter.set_max_items(20)
    formatter.set_num_spaces(6)
    assert formatter is not formatter_cloned
    assert formatter.equal(MappingFormatter(max_items=20, num_spaces=6))
    assert formatter_cloned.equal(MappingFormatter(max_items=10, num_spaces=4))


def test_mapping_formatter_equal_true() -> None:
    assert MappingFormatter().equal(MappingFormatter())


def test_mapping_formatter_equal_false_different_max_items() -> None:
    assert not MappingFormatter().equal(MappingFormatter(max_items=10))


def test_mapping_formatter_equal_false_different_num_spaces() -> None:
    assert not MappingFormatter().equal(MappingFormatter(num_spaces=4))


def test_mapping_formatter_equal_false_different_type() -> None:
    assert not MappingFormatter().equal(42)


def test_mapping_formatter_format_dict_empty() -> None:
    assert MappingFormatter().format(Summarizer(), {}) == "<class 'dict'> {}"


def test_mapping_formatter_format_dict_1() -> None:
    assert (
        MappingFormatter().format(Summarizer(), {"key": "value"})
        == "<class 'dict'> (length=1)\n  (key): value"
    )


def test_mapping_formatter_format_dict_2() -> None:
    assert MappingFormatter().format(Summarizer(), {1: "one line", "two": "two\nlines"}) == (
        "<class 'dict'> (length=2)\n  (1): one line\n  (two): two\n    lines"
    )


def test_mapping_formatter_format_length_5() -> None:
    assert MappingFormatter().format(Summarizer(), {f"key{i}": f"value{i}" for i in range(5)}) == (
        "<class 'dict'> (length=5)\n  (key0): value0\n  (key1): value1\n  (key2): value2\n"
        "  (key3): value3\n  (key4): value4"
    )


def test_mapping_formatter_format_length_10() -> None:
    assert MappingFormatter().format(Summarizer(), {f"key{i}": f"value{i}" for i in range(10)}) == (
        "<class 'dict'> (length=10)\n"
        "  (key0): value0\n"
        "  (key1): value1\n"
        "  (key2): value2\n"
        "  (key3): value3\n"
        "  (key4): value4\n"
        "  ..."
    )


def test_mapping_formatter_format_all_items() -> None:
    assert MappingFormatter(max_items=-1).format(
        Summarizer(), {f"key{i}": f"value{i}" for i in range(10)}
    ) == (
        "<class 'dict'> (length=10)\n"
        "  (key0): value0\n"
        "  (key1): value1\n"
        "  (key2): value2\n"
        "  (key3): value3\n"
        "  (key4): value4\n"
        "  (key5): value5\n"
        "  (key6): value6\n"
        "  (key7): value7\n"
        "  (key8): value8\n"
        "  (key9): value9"
    )


def test_mapping_formatter_format_length_10_max_items_5_max_depth_2() -> None:
    assert MappingFormatter(max_items=5).format(
        Summarizer(), {f"key{i}": i for i in range(10)}, max_depth=2
    ) == (
        "<class 'dict'> (length=10)\n"
        "  (key0): <class 'int'> 0\n"
        "  (key1): <class 'int'> 1\n"
        "  (key2): <class 'int'> 2\n"
        "  (key3): <class 'int'> 3\n"
        "  (key4): <class 'int'> 4\n"
        "  ..."
    )


def test_mapping_formatter_format_max_items_3() -> None:
    assert MappingFormatter(max_items=3).format(
        Summarizer(), {f"key{i}": f"value{i}" for i in range(10)}
    ) == (
        "<class 'dict'> (length=10)\n"
        "  (key0): value0\n"
        "  (key1): value1\n"
        "  (key2): value2\n"
        "  ..."
    )


def test_mapping_formatter_format_nested_dict() -> None:
    assert MappingFormatter().format(
        Summarizer(), {"key0": {"key0": 0, "key1": 1, "key2": 1}, "key1": {1: 2, 3: 4}}
    ) == (
        "<class 'dict'> (length=2)\n  (key0): {'key0': 0, 'key1': 1, 'key2': 1}\n"
        "  (key1): {1: 2, 3: 4}"
    )


def test_mapping_formatter_format_nested_dict_max_depth_2() -> None:
    assert MappingFormatter().format(
        Summarizer(), {"key0": {"key0": 0, "key1": 1, "key2": 1}, "key1": {1: 2, 3: 4}}, max_depth=2
    ) == (
        "<class 'dict'> (length=2)\n"
        "  (key0): <class 'dict'> (length=3)\n"
        "      (key0): 0\n"
        "      (key1): 1\n"
        "      (key2): 1\n"
        "  (key1): <class 'dict'> (length=2)\n"
        "      (1): 2\n"
        "      (3): 4"
    )


def test_mapping_formatter_format_nested_dict_max_depth_3() -> None:
    assert MappingFormatter().format(
        Summarizer(), {"key0": {"key0": 0, "key1": 1, "key2": 1}, "key1": {1: 2, 3: 4}}, max_depth=3
    ) == (
        "<class 'dict'> (length=2)\n"
        "  (key0): <class 'dict'> (length=3)\n"
        "      (key0): <class 'int'> 0\n"
        "      (key1): <class 'int'> 1\n"
        "      (key2): <class 'int'> 1\n"
        "  (key1): <class 'dict'> (length=2)\n"
        "      (1): <class 'int'> 2\n"
        "      (3): <class 'int'> 4"
    )


@mark.parametrize("max_depth", (0, -1, -2))
def test_mapping_formatter_format_nested_dict_max_depth_0(max_depth: int) -> None:
    assert MappingFormatter().format(
        Summarizer(),
        {"key0": {"key0": 0, "key1": 1, "key2": 1}, "key1": {1: 2, 3: 4}},
        max_depth=max_depth,
    ) == ("{'key0': {'key0': 0, 'key1': 1, 'key2': 1}, 'key1': {1: 2, 3: 4}}")


def test_mapping_formatter_format_nested_dict_max_characters() -> None:
    formatter = MappingFormatter()
    with summarizer_options(max_characters=5):
        assert formatter.format(
            Summarizer(), {"key0": {"key0": 0, "key1": 1, "key2": 1}, "key1": {1: 2, 3: 4}}
        ) == ("<class 'dict'> (length=2)\n  (key0): {'key...\n  (key1): {1: 2...")


def test_mapping_formatter_format_num_spaces_4() -> None:
    assert MappingFormatter(num_spaces=4).format(
        Summarizer(), {1: "one line", "two": "two\nlines"}
    ) == ("<class 'dict'> (length=2)\n    (1): one line\n    (two): two\n        lines")


def test_mapping_formatter_format_defaultdict_1() -> None:
    d = defaultdict(int)
    d["key"] = 1
    assert (
        MappingFormatter().format(Summarizer(), d)
        == "<class 'collections.defaultdict'> (length=1)\n  (key): 1"
    )


def test_mapping_formatter_format_ordereddict_1() -> None:
    assert (
        MappingFormatter().format(
            Summarizer(), OrderedDict([("key1", "value1"), ("key2", "value2")])
        )
        == "<class 'collections.OrderedDict'> (length=2)\n  (key1): value1\n  (key2): value2"
    )


def test_mapping_formatter_load_state_dict() -> None:
    formatter = MappingFormatter()
    formatter.load_state_dict({"max_items": 10, "num_spaces": 4})
    assert formatter.equal(MappingFormatter(max_items=10, num_spaces=4))


def test_mapping_formatter_state_dict() -> None:
    assert MappingFormatter().state_dict() == {"max_items": 5, "num_spaces": 2}


def test_mapping_formatter_get_max_items() -> None:
    assert MappingFormatter().get_max_items() == 5


@mark.parametrize("max_items", (-1, 0, 1, 10))
def test_mapping_formatter_set_max_items_int(max_items: int) -> None:
    formatter = MappingFormatter()
    assert formatter.get_max_items() == 5
    formatter.set_max_items(max_items)
    assert formatter.get_max_items() == max_items


def test_mapping_formatter_set_max_items_incorrect_type() -> None:
    formatter = MappingFormatter()
    with raises(TypeError, match="Incorrect type for max_items. Expected int value"):
        formatter.set_max_items(4.2)


def test_mapping_formatter_get_num_spaces() -> None:
    assert MappingFormatter().get_num_spaces() == 2


@mark.parametrize("num_spaces", (0, 1, 10))
def test_mapping_formatter_set_num_spaces_int(num_spaces: int) -> None:
    formatter = MappingFormatter()
    assert formatter.get_num_spaces() == 2
    formatter.set_num_spaces(num_spaces)
    assert formatter.get_num_spaces() == num_spaces


def test_mapping_formatter_set_num_spaces_incorrect_type() -> None:
    formatter = MappingFormatter()
    with raises(TypeError, match="Incorrect type for num_spaces. Expected int value"):
        formatter.set_num_spaces(4.2)


@mark.parametrize("num_spaces", (-1, -2))
def test_mapping_formatter_set_num_spaces_incorrect_value(num_spaces: int) -> None:
    formatter = MappingFormatter()
    with raises(ValueError, match="Incorrect value for num_spaces. Expected a positive integer"):
        formatter.set_num_spaces(num_spaces)


#######################################
#     Tests for SequenceFormatter     #
#######################################


def test_sequence_formatter_str() -> None:
    assert str(SequenceFormatter()).startswith("SequenceFormatter(")


def test_sequence_formatter_clone_max_items_10() -> None:
    formatter = SequenceFormatter(max_items=10, num_spaces=4)
    formatter_cloned = formatter.clone()
    formatter.set_max_items(20)
    formatter.set_num_spaces(6)
    assert formatter is not formatter_cloned
    assert formatter.equal(SequenceFormatter(max_items=20, num_spaces=6))
    assert formatter_cloned.equal(SequenceFormatter(max_items=10, num_spaces=4))


def test_sequence_formatter_equal_true() -> None:
    assert SequenceFormatter().equal(SequenceFormatter())


def test_sequence_formatter_equal_false_different_max_items() -> None:
    assert not SequenceFormatter().equal(SequenceFormatter(max_items=10))


def test_sequence_formatter_equal_false_different_num_spaces() -> None:
    assert not SequenceFormatter().equal(SequenceFormatter(num_spaces=10))


def test_sequence_formatter_equal_false_different_type() -> None:
    assert not SequenceFormatter().equal(42)


def test_sequence_formatter_format_list_empty() -> None:
    assert SequenceFormatter().format(Summarizer(), []) == "<class 'list'> []"


def test_sequence_formatter_format_list_1() -> None:
    assert (
        SequenceFormatter().format(Summarizer(), ["abc"]) == "<class 'list'> (length=1)\n  (0): abc"
    )


def test_sequence_formatter_format_list_2() -> None:
    assert SequenceFormatter().format(Summarizer(), ["one line", "two\nlines"]) == (
        "<class 'list'> (length=2)\n  (0): one line\n  (1): two\n    lines"
    )


def test_sequence_formatter_format_length_5() -> None:
    assert SequenceFormatter().format(Summarizer(), list(range(5))) == (
        "<class 'list'> (length=5)\n  (0): 0\n  (1): 1\n  (2): 2\n  (3): 3\n  (4): 4"
    )


def test_sequence_formatter_format_length_10() -> None:
    assert SequenceFormatter().format(Summarizer(), list(range(10))) == (
        "<class 'list'> (length=10)\n"
        "  (0): 0\n"
        "  (1): 1\n"
        "  (2): 2\n"
        "  (3): 3\n"
        "  (4): 4\n"
        "  ..."
    )


def test_sequence_formatter_format_length_10_max_items_5_max_depth_2() -> None:
    assert SequenceFormatter(max_items=5).format(Summarizer(), list(range(10)), max_depth=2) == (
        "<class 'list'> (length=10)\n"
        "  (0): <class 'int'> 0\n"
        "  (1): <class 'int'> 1\n"
        "  (2): <class 'int'> 2\n"
        "  (3): <class 'int'> 3\n"
        "  (4): <class 'int'> 4\n"
        "  ..."
    )


def test_sequence_formatter_format_all_items() -> None:
    assert SequenceFormatter(max_items=-1).format(Summarizer(), list(range(10))) == (
        "<class 'list'> (length=10)\n"
        "  (0): 0\n"
        "  (1): 1\n"
        "  (2): 2\n"
        "  (3): 3\n"
        "  (4): 4\n"
        "  (5): 5\n"
        "  (6): 6\n"
        "  (7): 7\n"
        "  (8): 8\n"
        "  (9): 9"
    )


def test_sequence_formatter_format_nested_list() -> None:
    assert SequenceFormatter().format(Summarizer(), [[0, 1, 2], ["abc", "def"]]) == (
        "<class 'list'> (length=2)\n  (0): [0, 1, 2]\n  (1): ['abc', 'def']"
    )


def test_sequence_formatter_format_nested_list_max_depth_2() -> None:
    assert SequenceFormatter().format(Summarizer(), [[0, 1, 2], ["abc", "def"]], max_depth=2) == (
        "<class 'list'> (length=2)\n"
        "  (0): <class 'list'> (length=3)\n"
        "      (0): 0\n"
        "      (1): 1\n"
        "      (2): 2\n"
        "  (1): <class 'list'> (length=2)\n"
        "      (0): abc\n"
        "      (1): def"
    )


def test_sequence_formatter_format_nested_list_max_depth_3() -> None:
    assert SequenceFormatter().format(Summarizer(), [[0, 1, 2], ["abc", "def"]], max_depth=3) == (
        "<class 'list'> (length=2)\n"
        "  (0): <class 'list'> (length=3)\n"
        "      (0): <class 'int'> 0\n"
        "      (1): <class 'int'> 1\n"
        "      (2): <class 'int'> 2\n"
        "  (1): <class 'list'> (length=2)\n"
        "      (0): <class 'str'> abc\n"
        "      (1): <class 'str'> def"
    )


@mark.parametrize("max_depth", (0, -1, -2))
def test_sequence_formatter_format_nested_list_max_depth_0(max_depth: int) -> None:
    assert SequenceFormatter().format(
        Summarizer(), [[0, 1, 2], ["abc", "def"]], max_depth=max_depth
    ) == ("[[0, 1, 2], ['abc', 'def']]")


def test_sequence_formatter_format_nested_list_max_characters() -> None:
    formatter = SequenceFormatter()
    with summarizer_options(max_characters=5):
        assert formatter.format(Summarizer(), [[0, 1, 2], ["abc", "def"]]) == (
            "<class 'list'> (length=2)\n  (0): [0, 1...\n  (1): ['abc..."
        )


def test_sequence_formatter_format_num_spaces_4() -> None:
    assert (
        SequenceFormatter(num_spaces=4).format(Summarizer(), ["one line", "two\nlines"])
        == "<class 'list'> (length=2)\n    (0): one line\n    (1): two\n        lines"
    )


def test_sequence_formatter_format_tuple() -> None:
    assert (
        SequenceFormatter().format(Summarizer(), ("abc", "def"))
        == "<class 'tuple'> (length=2)\n  (0): abc\n  (1): def"
    )


def test_sequence_formatter_load_state_dict() -> None:
    formatter = SequenceFormatter()
    formatter.load_state_dict({"max_items": 10, "num_spaces": 4})
    assert formatter.equal(SequenceFormatter(max_items=10, num_spaces=4))


def test_sequence_formatter_state_dict() -> None:
    assert SequenceFormatter().state_dict() == {"max_items": 5, "num_spaces": 2}


def test_sequence_formatter_get_max_items() -> None:
    assert SequenceFormatter().get_max_items() == 5


@mark.parametrize("max_items", (-1, 0, 1, 10))
def test_sequence_formatter_set_max_items_int(max_items: int) -> None:
    formatter = SequenceFormatter()
    assert formatter.get_max_items() == 5
    formatter.set_max_items(max_items)
    assert formatter.get_max_items() == max_items


def test_sequence_formatter_set_max_items_incorrect_type() -> None:
    formatter = SequenceFormatter()
    with raises(TypeError, match="Incorrect type for max_items. Expected int value"):
        formatter.set_max_items(4.2)


def test_sequence_formatter_get_num_spaces() -> None:
    assert SequenceFormatter().get_num_spaces() == 2


@mark.parametrize("num_spaces", (0, 1, 10))
def test_sequence_formatter_set_num_spaces_int(num_spaces: int) -> None:
    formatter = SequenceFormatter()
    assert formatter.get_num_spaces() == 2
    formatter.set_num_spaces(num_spaces)
    assert formatter.get_num_spaces() == num_spaces


def test_sequence_formatter_set_num_spaces_incorrect_type() -> None:
    formatter = SequenceFormatter()
    with raises(TypeError, match="Incorrect type for num_spaces. Expected int value"):
        formatter.set_num_spaces(4.2)


@mark.parametrize("num_spaces", (-1, -2))
def test_sequence_formatter_set_num_spaces_incorrect_value(num_spaces: int) -> None:
    formatter = SequenceFormatter()
    with raises(ValueError, match="Incorrect value for num_spaces. Expected a positive integer"):
        formatter.set_num_spaces(num_spaces)


##################################
#     Tests for SetFormatter     #
##################################


def test_set_formatter_str() -> None:
    assert str(SetFormatter()).startswith("SetFormatter(")


def test_set_formatter_clone_max_items_10() -> None:
    formatter = SetFormatter(max_items=10, num_spaces=4)
    formatter_cloned = formatter.clone()
    formatter.set_max_items(20)
    formatter.set_num_spaces(6)
    assert formatter is not formatter_cloned
    assert formatter.equal(SetFormatter(max_items=20, num_spaces=6))
    assert formatter_cloned.equal(SetFormatter(max_items=10, num_spaces=4))


def test_set_formatter_equal_true() -> None:
    assert SetFormatter().equal(SetFormatter())


def test_set_formatter_equal_false_different_max_items() -> None:
    assert not SetFormatter().equal(SetFormatter(max_items=10))


def test_set_formatter_equal_false_different_type() -> None:
    assert not SetFormatter().equal(42)


def test_set_formatter_format_empty() -> None:
    assert SetFormatter().format(Summarizer(), set()) == "<class 'set'> set()"


def test_set_formatter_format_1() -> None:
    assert SetFormatter().format(Summarizer(), {"abc"}) == "<class 'set'> (length=1)\n  (0): abc"


def test_set_formatter_format_2() -> None:
    s = SetFormatter().format(Summarizer(), {"one line", "two\nlines"})
    assert s in {
        "<class 'set'> (length=2)\n  (0): one line\n  (1): two\n    lines",
        "<class 'set'> (length=2)\n  (0): two\n    lines\n  (1): one line",
    }


def test_set_formatter_format_length_5() -> None:
    s = SetFormatter().format(Summarizer(), set(range(5)))
    assert s.startswith("<class 'set'> (length=5)\n  (0): ")
    assert len(s.split("\n")) == 6


def test_set_formatter_format_length_10() -> None:
    s = SetFormatter().format(Summarizer(), set(range(10)))
    assert s.startswith("<class 'set'> (length=10)\n  (0): ")
    assert s.endswith("...")
    assert len(s.split("\n")) == 7


def test_set_formatter_format_all_items() -> None:
    s = SetFormatter(max_items=-1).format(Summarizer(), set(range(10)))
    assert s.startswith("<class 'set'> (length=10)\n  (0): ")
    assert not s.endswith("...")
    assert len(s.split("\n")) == 11


def test_set_formatter_format_length_10_max_items_5_max_depth_2() -> None:
    s = SetFormatter(max_items=5).format(Summarizer(), set(range(10)), max_depth=2)
    assert s.startswith("<class 'set'> (length=10)\n  (0): <class 'int'>")
    assert s.endswith("...")
    assert len(s.split("\n")) == 7


def test_set_formatter_format_nested() -> None:
    s = SetFormatter().format(Summarizer(), {(0, 1, 2), ("abc", "def")})
    assert s in {
        "<class 'set'> (length=2)\n  (0): (0, 1, 2)\n  (1): ('abc', 'def')",
        "<class 'set'> (length=2)\n  (0): ('abc', 'def')\n  (1): (0, 1, 2)",
    }


def test_set_formatter_format_nested_max_depth_2() -> None:
    s = SetFormatter().format(Summarizer(), {(0, 1, 2), ("abc", "def")}, max_depth=2)
    assert s in {
        "<class 'set'> (length=2)\n"
        "  (0): <class 'tuple'> (length=3)\n"
        "      (0): 0\n"
        "      (1): 1\n"
        "      (2): 2\n"
        "  (1): <class 'tuple'> (length=2)\n"
        "      (0): abc\n"
        "      (1): def",
        "<class 'set'> (length=2)\n"
        "  (0): <class 'tuple'> (length=2)\n"
        "      (0): abc\n"
        "      (1): def\n"
        "  (1): <class 'tuple'> (length=3)\n"
        "      (0): 0\n"
        "      (1): 1\n"
        "      (2): 2",
    }


def test_set_formatter_format_nested_max_depth_3() -> None:
    s = SetFormatter().format(Summarizer(), {(0, 1, 2), ("abc", "def")}, max_depth=3)
    assert s in {
        "<class 'set'> (length=2)\n"
        "  (0): <class 'tuple'> (length=3)\n"
        "      (0): <class 'int'> 0\n"
        "      (1): <class 'int'> 1\n"
        "      (2): <class 'int'> 2\n"
        "  (1): <class 'tuple'> (length=2)\n"
        "      (0): <class 'str'> abc\n"
        "      (1): <class 'str'> def",
        "<class 'set'> (length=2)\n"
        "  (0): <class 'tuple'> (length=2)\n"
        "      (0): <class 'str'> abc\n"
        "      (1): <class 'str'> def\n"
        "  (1): <class 'tuple'> (length=3)\n"
        "      (0): <class 'int'> 0\n"
        "      (1): <class 'int'> 1\n"
        "      (2): <class 'int'> 2",
    }


@mark.parametrize("max_depth", (0, -1, -2))
def test_set_formatter_format_nested_max_depth_0(max_depth: int) -> None:
    s = SetFormatter().format(Summarizer(), {(0, 1, 2), ("abc", "def")}, max_depth=max_depth)
    assert s in {"{(0, 1, 2), ('abc', 'def')}", "{('abc', 'def'), (0, 1, 2)}"}


def test_set_formatter_format_nested_max_characters() -> None:
    formatter = SetFormatter()
    with summarizer_options(max_characters=5):
        s = formatter.format(Summarizer(), {(0, 1, 2), ("abc", "def")})
        assert s in {
            "<class 'set'> (length=2)\n  (0): (0, 1...\n  (1): ('abc...",
            "<class 'set'> (length=2)\n  (0): ('abc...\n  (1): (0, 1...",
        }


def test_set_formatter_format_num_spaces_4() -> None:
    s = SetFormatter(num_spaces=4).format(Summarizer(), {"one line", "two\nlines"})
    assert s in {
        "<class 'set'> (length=2)\n    (0): one line\n    (1): two\n        lines",
        "<class 'set'> (length=2)\n    (0): two\n        lines\n    (1): one line",
    }


def test_set_formatter_load_state_dict() -> None:
    formatter = SetFormatter()
    formatter.load_state_dict({"max_items": 10, "num_spaces": 4})
    assert formatter.equal(SetFormatter(max_items=10, num_spaces=4))


def test_set_formatter_state_dict() -> None:
    assert SetFormatter().state_dict() == {"max_items": 5, "num_spaces": 2}


def test_set_formatter_get_max_items() -> None:
    assert SetFormatter().get_max_items() == 5


@mark.parametrize("max_items", (-1, 0, 1, 10))
def test_set_formatter_set_max_items_int(max_items: int) -> None:
    formatter = SetFormatter()
    assert formatter.get_max_items() == 5
    formatter.set_max_items(max_items)
    assert formatter.get_max_items() == max_items


def test_set_formatter_set_max_items_incorrect_type() -> None:
    formatter = SetFormatter()
    with raises(TypeError, match="Incorrect type for max_items. Expected int value"):
        formatter.set_max_items(4.2)


def test_set_formatter_get_num_spaces() -> None:
    assert SetFormatter().get_num_spaces() == 2


@mark.parametrize("num_spaces", (0, 1, 10))
def test_set_formatter_set_num_spaces_int(num_spaces: int) -> None:
    formatter = SetFormatter()
    assert formatter.get_num_spaces() == 2
    formatter.set_num_spaces(num_spaces)
    assert formatter.get_num_spaces() == num_spaces


def test_set_formatter_set_num_spaces_incorrect_type() -> None:
    formatter = SetFormatter()
    with raises(TypeError, match="Incorrect type for num_spaces. Expected int value"):
        formatter.set_num_spaces(4.2)


@mark.parametrize("num_spaces", (-1, -2))
def test_set_formatter_set_num_spaces_incorrect_value(num_spaces: int) -> None:
    formatter = SetFormatter()
    with raises(ValueError, match="Incorrect value for num_spaces. Expected a positive integer"):
        formatter.set_num_spaces(num_spaces)
