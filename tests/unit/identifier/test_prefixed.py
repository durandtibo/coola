from __future__ import annotations

import re

import pytest

from coola.identifier.prefixed import generate_prefixed_id
from coola.identifier.ulid import generate_ulid

########################################
#     Tests for generate_prefixed_id  #
########################################


def test_generate_prefixed_id_returns_str() -> None:
    assert isinstance(generate_prefixed_id("cus"), str)


def test_generate_prefixed_id_uses_prefix() -> None:
    assert generate_prefixed_id("cus").startswith("cus_")


def test_generate_prefixed_id_default_generator_is_ulid() -> None:
    value = generate_prefixed_id("cus")
    suffix = value.removeprefix("cus_")
    assert re.match(r"^[0-9A-HJKMNP-TV-Z]{26}$", suffix)


def test_generate_prefixed_id_is_unique() -> None:
    assert generate_prefixed_id("cus") != generate_prefixed_id("cus")


def test_generate_prefixed_id_uses_given_generator() -> None:
    assert generate_prefixed_id("evt", generator=lambda: "42") == "evt_42"


def test_generate_prefixed_id_custom_generator_called_each_time() -> None:
    counter = iter(range(2))
    value1 = generate_prefixed_id("evt", generator=lambda: str(next(counter)))
    value2 = generate_prefixed_id("evt", generator=lambda: str(next(counter)))
    assert value1 == "evt_0"
    assert value2 == "evt_1"


def test_generate_prefixed_id_empty_prefix_raises() -> None:
    with pytest.raises(ValueError, match="prefix must not be empty"):
        generate_prefixed_id("")


def test_generate_prefixed_id_prefix_with_underscore_raises() -> None:
    with pytest.raises(ValueError, match="prefix must not contain '_'"):
        generate_prefixed_id("cus_tom")


def test_generate_prefixed_id_with_generate_ulid_matches_pattern() -> None:
    value = generate_prefixed_id("rec", generator=generate_ulid)
    assert re.match(r"^rec_[0-9A-HJKMNP-TV-Z]{26}$", value)


def test_generate_prefixed_id_empty_generator_output_raises() -> None:
    with pytest.raises(ValueError, match="generator\\(\\) must return a non-empty string"):
        generate_prefixed_id("cus", generator=lambda: "")
