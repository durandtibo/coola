r"""Property-based tests for ``coola.identifier``."""

from __future__ import annotations

from hypothesis import assume, given, strategies as st

from coola.identifier import generate_checksummed_id, verify_checksummed_id
from coola.identifier.crockford import ALPHABET

###################################
#     Checksummed identifiers     #
###################################

lengths = st.integers(1, 40)
group_sizes = st.integers(1, 10)
separators = st.sampled_from(["-", " ", "_", "."])


@given(length=lengths, group_size=group_sizes, sep=separators)
def test_checksummed_id_is_verified(length: int, group_size: int, sep: str) -> None:
    identifier = generate_checksummed_id(length=length, group_size=group_size, sep=sep)
    assert verify_checksummed_id(identifier, sep=sep)


@given(length=lengths, group_size=group_sizes, sep=separators)
def test_checksummed_id_only_has_the_alphabet_and_the_separator(
    length: int, group_size: int, sep: str
) -> None:
    identifier = generate_checksummed_id(length=length, group_size=group_size, sep=sep)
    assert set(identifier) <= set(ALPHABET + sep)


@given(length=lengths, group_size=group_sizes, data=st.data())
def test_checksummed_id_detects_a_single_substitution(
    length: int, group_size: int, data: st.DataObject
) -> None:
    identifier = generate_checksummed_id(length=length, group_size=group_size, sep="")
    index = data.draw(st.integers(0, len(identifier) - 1))
    char = data.draw(st.sampled_from(ALPHABET))
    assume(char != identifier[index])
    corrupted = identifier[:index] + char + identifier[index + 1 :]
    assert not verify_checksummed_id(corrupted, sep="")


@given(length=lengths, group_size=group_sizes, data=st.data())
def test_checksummed_id_detects_a_swap_of_adjacent_characters(
    length: int, group_size: int, data: st.DataObject
) -> None:
    identifier = generate_checksummed_id(length=length, group_size=group_size, sep="")
    index = data.draw(st.integers(0, len(identifier) - 2))
    assume(identifier[index] != identifier[index + 1])
    corrupted = (
        identifier[:index] + identifier[index + 1] + identifier[index] + identifier[index + 2 :]
    )
    assert not verify_checksummed_id(corrupted, sep="")


@given(identifier=st.one_of(st.none(), st.integers(), st.binary(), st.text()))
def test_verify_checksummed_id_never_raises(identifier: object) -> None:
    assert verify_checksummed_id(identifier) in {True, False}
