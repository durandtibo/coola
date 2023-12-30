from collections.abc import Mapping, Sequence
from unittest.mock import Mock, patch

import pytest

from coola import Summarizer, set_summarizer_options, summarizer_options
from coola.formatters import (
    BaseFormatter,
    DefaultFormatter,
    MappingFormatter,
    SequenceFormatter,
    SetFormatter,
)


@pytest.fixture(autouse=True)
def _reset() -> None:
    state = Summarizer.state_dict()
    try:
        yield
    finally:
        Summarizer.load_state_dict(state)


################################
#     Tests for Summarizer     #
################################


def test_summarizer_str() -> None:
    assert str(Summarizer()).startswith("Summarizer(")


def test_summarizer_registry_default() -> None:
    assert len(Summarizer.registry) >= 7
    assert isinstance(Summarizer.registry[Mapping], MappingFormatter)
    assert isinstance(Summarizer.registry[Sequence], SequenceFormatter)
    assert isinstance(Summarizer.registry[dict], MappingFormatter)
    assert isinstance(Summarizer.registry[list], SequenceFormatter)
    assert isinstance(Summarizer.registry[object], DefaultFormatter)
    assert isinstance(Summarizer.registry[set], SetFormatter)
    assert isinstance(Summarizer.registry[tuple], SequenceFormatter)


def test_summarizer_summary_bool() -> None:
    assert Summarizer().summary(True) == "<class 'bool'> True"


def test_summarizer_summary_int() -> None:
    assert Summarizer().summary(42) == "<class 'int'> 42"


def test_summarizer_summary_max_depth_1() -> None:
    assert (
        Summarizer().summary([[0, 1, 2], {"key1": "abc", "key2": "def"}])
        == "<class 'list'> (length=2)\n  (0): [0, 1, 2]\n  (1): {'key1': 'abc', 'key2': 'def'}"
    )


def test_summarizer_summary_max_depth_2() -> None:
    assert Summarizer().summary([[0, 1, 2], {"key1": "abc", "key2": "def"}], max_depth=2) == (
        "<class 'list'> (length=2)\n"
        "  (0): <class 'list'> (length=3)\n"
        "      (0): 0\n"
        "      (1): 1\n"
        "      (2): 2\n"
        "  (1): <class 'dict'> (length=2)\n"
        "      (key1): abc\n"
        "      (key2): def"
    )


def test_summarizer_summary_max_depth_3() -> None:
    assert Summarizer().summary([[0, 1, 2], {"key1": "abc", "key2": "def"}], max_depth=3) == (
        "<class 'list'> (length=2)\n"
        "  (0): <class 'list'> (length=3)\n"
        "      (0): <class 'int'> 0\n"
        "      (1): <class 'int'> 1\n"
        "      (2): <class 'int'> 2\n"
        "  (1): <class 'dict'> (length=2)\n"
        "      (key1): <class 'str'> abc\n"
        "      (key2): <class 'str'> def"
    )


@patch.dict(Summarizer.registry, {}, clear=True)
def test_summarizer_add_formatter() -> None:
    formatter = Mock(spec=BaseFormatter)
    Summarizer.add_formatter(int, formatter)
    assert Summarizer.registry[int] == formatter


@patch.dict(Summarizer.registry, {}, clear=True)
def test_summarizer_add_formatter_duplicate_exist_ok_true() -> None:
    formatter = Mock(spec=BaseFormatter)
    Summarizer.add_formatter(int, Mock(spec=BaseFormatter))
    Summarizer.add_formatter(int, formatter, exist_ok=True)
    assert Summarizer.registry[int] == formatter


@patch.dict(Summarizer.registry, {}, clear=True)
def test_summarizer_add_formatter_duplicate_exist_ok_false() -> None:
    formatter = Mock(spec=BaseFormatter)
    Summarizer.add_formatter(int, Mock(spec=BaseFormatter))
    with pytest.raises(RuntimeError, match="A formatter (.*) is already registered"):
        Summarizer.add_formatter(int, formatter)


def test_summarizer_has_formatter_true() -> None:
    assert Summarizer.has_formatter(dict)


def test_summarizer_has_formatter_false() -> None:
    assert not Summarizer.has_formatter(int)


def test_summarizer_find_formatter_direct() -> None:
    assert isinstance(Summarizer.find_formatter(dict), MappingFormatter)


def test_summarizer_find_formatter_indirect() -> None:
    assert isinstance(Summarizer.find_formatter(str), DefaultFormatter)


def test_summarizer_find_formatter_incorrect_type() -> None:
    with pytest.raises(TypeError, match="Incorrect data type:"):
        Summarizer.find_formatter(Mock(__mro__=[]))


def test_summarizer_load_state_dict() -> None:
    Summarizer.load_state_dict({object: {"max_characters": 10}})
    assert Summarizer.registry[object].equal(DefaultFormatter(max_characters=10))


def test_summarizer_state_dict() -> None:
    state = Summarizer.state_dict()
    assert len(state) >= 6
    assert isinstance(state, dict)
    assert isinstance(state[Mapping], dict)
    assert isinstance(state[Sequence], dict)
    assert isinstance(state[dict], dict)
    assert isinstance(state[list], dict)
    assert isinstance(state[object], dict)
    assert isinstance(state[tuple], dict)


def test_summarizer_set_max_characters() -> None:
    Summarizer.set_max_characters(10)
    assert Summarizer.registry[object].equal(DefaultFormatter(max_characters=10))


def test_summarizer_set_max_items() -> None:
    Summarizer.set_max_items(10)
    assert Summarizer.registry[Mapping].equal(MappingFormatter(max_items=10))
    assert Summarizer.registry[Sequence].equal(SequenceFormatter(max_items=10))


def test_summarizer_set_num_spaces() -> None:
    Summarizer.set_num_spaces(4)
    assert Summarizer.registry[Mapping].equal(MappingFormatter(num_spaces=4))
    assert Summarizer.registry[Sequence].equal(SequenceFormatter(num_spaces=4))


############################################
#     Tests for set_summarizer_options     #
############################################


def test_set_summarizer_options() -> None:
    set_summarizer_options(max_characters=10, max_items=2, num_spaces=4)
    assert Summarizer.registry[object].equal(DefaultFormatter(max_characters=10))
    assert Summarizer.registry[Mapping].equal(MappingFormatter(max_items=2, num_spaces=4))
    assert Summarizer.registry[Sequence].equal(SequenceFormatter(max_items=2, num_spaces=4))


def test_set_summarizer_options_empty() -> None:
    state = Summarizer.state_dict()
    set_summarizer_options()
    assert state == Summarizer.state_dict()


@pytest.mark.parametrize("max_characters", [-1, 0, 1, 10])
def test_set_summarizer_options_max_characters(max_characters: int) -> None:
    set_summarizer_options(max_characters=max_characters)
    assert Summarizer.registry[object].equal(DefaultFormatter(max_characters=max_characters))


@pytest.mark.parametrize("max_items", [0, 1, 2])
def test_set_summarizer_options_max_items(max_items: int) -> None:
    set_summarizer_options(max_items=max_items)
    assert Summarizer.registry[Mapping].equal(MappingFormatter(max_items=max_items))
    assert Summarizer.registry[Sequence].equal(SequenceFormatter(max_items=max_items))


@pytest.mark.parametrize("num_spaces", [0, 1, 2])
def test_set_summarizer_options_num_spaces(num_spaces: int) -> None:
    set_summarizer_options(num_spaces=num_spaces)
    assert Summarizer.registry[Mapping].equal(MappingFormatter(num_spaces=num_spaces))
    assert Summarizer.registry[Sequence].equal(SequenceFormatter(num_spaces=num_spaces))


########################################
#     Tests for summarizer_options     #
########################################


def test_summarizer_options() -> None:
    assert Summarizer.registry[object].equal(DefaultFormatter())
    assert Summarizer.registry[Mapping].equal(MappingFormatter())
    assert Summarizer.registry[Sequence].equal(SequenceFormatter())
    with summarizer_options(max_characters=10, max_items=2, num_spaces=4):
        assert Summarizer.registry[object].equal(DefaultFormatter(max_characters=10))
        assert Summarizer.registry[Mapping].equal(MappingFormatter(max_items=2, num_spaces=4))
        assert Summarizer.registry[Sequence].equal(SequenceFormatter(max_items=2, num_spaces=4))
    assert Summarizer.registry[object].equal(DefaultFormatter())
    assert Summarizer.registry[Mapping].equal(MappingFormatter())
    assert Summarizer.registry[Sequence].equal(SequenceFormatter())


def test_summarizer_options_empty() -> None:
    state = Summarizer.state_dict()
    with summarizer_options():
        assert state == Summarizer.state_dict()


@pytest.mark.parametrize("max_characters", [-1, 0, 1, 10])
def test_summarizer_options_max_characters(max_characters: int) -> None:
    assert Summarizer.registry[object].equal(DefaultFormatter())
    with summarizer_options(max_characters=max_characters):
        assert Summarizer.registry[object].equal(DefaultFormatter(max_characters=max_characters))
    assert Summarizer.registry[object].equal(DefaultFormatter())


@pytest.mark.parametrize("max_items", [0, 1, 10])
def test_summarizer_options_max_items(max_items: int) -> None:
    assert Summarizer.registry[Mapping].equal(MappingFormatter())
    assert Summarizer.registry[Sequence].equal(SequenceFormatter())
    with summarizer_options(max_items=max_items):
        assert Summarizer.registry[Mapping].equal(MappingFormatter(max_items=max_items))
        assert Summarizer.registry[Sequence].equal(SequenceFormatter(max_items=max_items))
    assert Summarizer.registry[Mapping].equal(MappingFormatter())
    assert Summarizer.registry[Sequence].equal(SequenceFormatter())


@pytest.mark.parametrize("num_spaces", [0, 1, 10])
def test_summarizer_options_num_spaces(num_spaces: int) -> None:
    assert Summarizer.registry[Mapping].equal(MappingFormatter())
    assert Summarizer.registry[Sequence].equal(SequenceFormatter())
    with summarizer_options(num_spaces=num_spaces):
        assert Summarizer.registry[Mapping].equal(MappingFormatter(num_spaces=num_spaces))
        assert Summarizer.registry[Sequence].equal(SequenceFormatter(num_spaces=num_spaces))
    assert Summarizer.registry[Mapping].equal(MappingFormatter())
    assert Summarizer.registry[Sequence].equal(SequenceFormatter())
