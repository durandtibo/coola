from __future__ import annotations

from coola.equality.handler.format import (
    format_mapping_difference,
    format_sequence_difference,
    format_shape_difference,
    format_type_difference,
    format_value_difference,
)

###############################################
#     Tests for format_mapping_difference     #
###############################################


def test_format_mapping_difference_different_keys() -> None:
    msg = format_mapping_difference(missing_keys={"b"}, additional_keys={"c"})
    assert msg == (
        "mappings have different keys:\n  missing keys    : ['b']\n  additional keys : ['c']"
    )


def test_format_mapping_difference_additional_keys() -> None:
    msg = format_mapping_difference(additional_keys={"c"})
    assert msg == "mappings have different keys:\n  additional keys : ['c']"


def test_format_mapping_difference_missing_keys() -> None:
    msg = format_mapping_difference(missing_keys={"b"})
    assert msg == "mappings have different keys:\n  missing keys    : ['b']"


def test_format_mapping_difference_different_value() -> None:
    msg = format_mapping_difference(different_value_key="b")
    assert msg == "mappings have different values for key 'b'"


def test_format_mapping_difference_missing_and_different() -> None:
    msg = format_mapping_difference(
        missing_keys={"b"}, additional_keys={"c"}, different_value_key="a"
    )
    assert msg == (
        "mappings have different keys:\n"
        "  missing keys    : ['b']\n"
        "  additional keys : ['c']\n"
        "mappings have different values for key 'a'"
    )


################################################
#     Tests for format_sequence_difference     #
################################################


def test_format_sequence_difference_with_index() -> None:
    msg = format_sequence_difference([1, 2, 3], [1, 2, 4], different_index=2)
    assert msg == "sequences have different values at index 2"


def test_format_sequence_difference_different_lengths() -> None:
    msg = format_sequence_difference([1, 2], [1, 2, 3], different_index=None)
    assert msg == "sequences have different lengths: 2 vs 3"


def test_format_sequence_difference_with_index_and_lengths() -> None:
    msg = format_sequence_difference([1, 2], [1, 2, 3], different_index=1)
    assert msg == "sequences have different values at index 1"


def test_format_sequence_difference_different() -> None:
    msg = format_sequence_difference([1, 2], [1, 4])
    assert msg == "sequences are different:\n  actual   : [1, 2]\n  expected : [1, 4]"


#############################################
#     Tests for format_shape_difference     #
#############################################


def test_format_shape_difference() -> None:
    msg = format_shape_difference((2, 3), (2, 4))
    assert msg == "objects have different shapes:\n  actual   : (2, 3)\n  expected : (2, 4)"


def test_format_shape_difference_1d_vs_2d() -> None:
    msg = format_shape_difference((5,), (5, 1))
    assert msg == "objects have different shapes:\n  actual   : (5,)\n  expected : (5, 1)"


############################################
#     Tests for format_type_difference     #
############################################


def test_format_type_difference() -> None:
    msg = format_type_difference(list, tuple)
    assert msg == (
        "objects have different types:\n  actual   : <class 'list'>\n  expected : <class 'tuple'>"
    )


def test_format_type_difference_int_float() -> None:
    msg = format_type_difference(int, float)
    assert msg == (
        "objects have different types:\n  actual   : <class 'int'>\n  expected : <class 'float'>"
    )


#############################################
#     Tests for format_value_difference     #
#############################################


def test_format_value_difference_default_name() -> None:
    msg = format_value_difference(1, 2)
    assert msg == "objects are different:\n  actual   : 1\n  expected : 2"


def test_format_value_difference_custom_name() -> None:
    msg = format_value_difference(1.0, 2.0, name="numbers")
    assert msg == "numbers are different:\n  actual   : 1.0\n  expected : 2.0"


def test_format_value_difference_arrays() -> None:
    msg = format_value_difference([1, 2, 3], [1, 2, 4], name="lists")
    assert msg == "lists are different:\n  actual   : [1, 2, 3]\n  expected : [1, 2, 4]"
