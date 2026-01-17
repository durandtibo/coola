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


def test_format_mapping_difference_missing_keys() -> None:
    msg = format_mapping_difference(
        missing_keys={"b"},
        additional_keys={"c"},
    )
    assert "mappings have different keys:" in msg
    assert "missing keys    : ['b']" in msg
    assert "additional keys : ['c']" in msg


def test_format_mapping_difference_different_value() -> None:
    msg = format_mapping_difference(
        different_value_key="b",
    )
    assert "mappings have different values for key 'b'" in msg


def test_format_mapping_difference_missing_and_different() -> None:
    msg = format_mapping_difference(
        missing_keys={"b"},
        additional_keys={"c"},
        different_value_key="a",
    )
    assert "mappings have different keys:" in msg
    assert "missing keys" in msg
    assert "additional keys" in msg


################################################
#     Tests for format_sequence_difference     #
################################################


def test_format_sequence_difference_with_index() -> None:
    msg = format_sequence_difference([1, 2, 3], [1, 2, 4], different_index=2)
    assert "sequences have different values at index 2" in msg


def test_format_sequence_difference_different_lengths() -> None:
    msg = format_sequence_difference([1, 2], [1, 2, 3], different_index=None)
    assert "sequences have different lengths: 2 vs 3" in msg


def test_format_sequence_difference_with_index_and_lengths() -> None:
    msg = format_sequence_difference([1, 2], [1, 2, 3], different_index=1)
    assert "sequences have different values at index 1" in msg


#############################################
#     Tests for format_shape_difference     #
#############################################


def test_format_shape_difference() -> None:
    msg = format_shape_difference((2, 3), (2, 4))
    assert "objects have different shapes:" in msg
    assert "actual   : (2, 3)" in msg
    assert "expected : (2, 4)" in msg


def test_format_shape_difference_1d_vs_2d() -> None:
    msg = format_shape_difference((5,), (5, 1))
    assert "objects have different shapes:" in msg
    assert "actual   : (5,)" in msg
    assert "expected : (5, 1)" in msg


############################################
#     Tests for format_type_difference     #
############################################


def test_format_type_difference() -> None:
    msg = format_type_difference(list, tuple)
    assert "objects have different types:" in msg
    assert "actual   : <class 'list'>" in msg
    assert "expected : <class 'tuple'>" in msg


def test_format_type_difference_int_float() -> None:
    msg = format_type_difference(int, float)
    assert "objects have different types:" in msg
    assert "actual   : <class 'int'>" in msg
    assert "expected : <class 'float'>" in msg


#############################################
#     Tests for format_value_difference     #
#############################################


def test_format_value_difference_default_name() -> None:
    msg = format_value_difference(1, 2)
    assert "objects are different:" in msg
    assert "actual   : 1" in msg
    assert "expected : 2" in msg


def test_format_value_difference_custom_name() -> None:
    msg = format_value_difference(1.0, 2.0, name="numbers")
    assert "numbers are different:" in msg
    assert "actual   : 1.0" in msg
    assert "expected : 2.0" in msg


def test_format_value_difference_arrays() -> None:
    msg = format_value_difference([1, 2, 3], [1, 2, 4], name="arrays")
    assert "arrays are different:" in msg
    assert "actual   : [1, 2, 3]" in msg
    assert "expected : [1, 2, 4]" in msg
