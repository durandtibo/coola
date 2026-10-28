from __future__ import annotations

import pytest

from coola.identifier.crockford import decode, encode

############################
#     Tests for encode     #
############################


def test_encode_zero() -> None:
    assert encode(0, length=4) == "0000"


def test_encode_pads_on_the_left() -> None:
    assert encode(31, length=3) == "00Z"


def test_encode_maximum_value_of_the_length() -> None:
    assert encode(32**3 - 1, length=3) == "ZZZ"


def test_encode_uses_upper_case() -> None:
    assert encode(10, length=1) == "A"


@pytest.mark.parametrize("value", [-1, 32**3])
def test_encode_value_out_of_range_raises(value: int) -> None:
    with pytest.raises(ValueError, match="value must fit in 3 Crockford Base32 characters"):
        encode(value, length=3)


############################
#     Tests for decode     #
############################


def test_decode_zero() -> None:
    assert decode("0000") == 0


def test_decode_value() -> None:
    assert decode("00Z") == 31


def test_decode_is_case_insensitive() -> None:
    assert decode("00z") == 31


@pytest.mark.parametrize(
    ("text", "expected"), [("O", 0), ("o", 0), ("I", 1), ("i", 1), ("L", 1), ("l", 1)]
)
def test_decode_confusable_letters(text: str, expected: int) -> None:
    assert decode(text) == expected


@pytest.mark.parametrize("value", [0, 1, 31, 32, 1023, 2**100 + 12345])
def test_decode_inverts_encode(value: int) -> None:
    assert decode(encode(value, length=21)) == value


@pytest.mark.parametrize(
    "text",
    [
        "U",
        "u",
        "-",
        " ",
        "\N{LATIN SMALL LETTER DOTLESS I}",
        "\N{LATIN SMALL LETTER Y WITH DIAERESIS}",
    ],
)
def test_decode_character_outside_the_alphabet_raises(text: str) -> None:
    with pytest.raises(ValueError, match="value contains a character outside"):
        decode(text)


def test_decode_empty_string_raises() -> None:
    with pytest.raises(ValueError, match="value must not be empty"):
        decode("")


def test_decode_uses_the_name_in_the_message() -> None:
    with pytest.raises(ValueError, match="ulid must not be empty"):
        decode("", name="ulid")
