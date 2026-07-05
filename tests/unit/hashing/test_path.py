from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from coola.hashing import HasherRegistry, PathHasher, hash_path


@pytest.fixture
def registry() -> HasherRegistry:
    return HasherRegistry({object: PathHasher()})


###############################
#     Tests for PathHasher    #
###############################


def test_path_hasher_repr() -> None:
    assert repr(PathHasher()) == "PathHasher()"


def test_path_hasher_str() -> None:
    assert str(PathHasher()) == "PathHasher()"


def test_path_hasher_hash_returns_str(tmp_path: Path, registry: HasherRegistry) -> None:
    assert isinstance(PathHasher().hash(tmp_path / "file.txt", registry=registry), str)


def test_path_hasher_hash_default_length(tmp_path: Path, registry: HasherRegistry) -> None:
    assert len(PathHasher().hash(tmp_path / "file.txt", registry=registry)) == 64


@pytest.mark.parametrize(
    "length",
    [
        pytest.param(16, id="16"),
        pytest.param(32, id="32"),
        pytest.param(64, id="64-default"),
        pytest.param(128, id="128"),
    ],
)
def test_path_hasher_hash_length(length: int, tmp_path: Path, registry: HasherRegistry) -> None:
    result = PathHasher().hash(tmp_path / "file.txt", registry=registry, length=length)
    assert len(result) == length


def test_path_hasher_hash_is_deterministic(tmp_path: Path, registry: HasherRegistry) -> None:
    hasher = PathHasher()
    path = tmp_path / "file.txt"
    assert hasher.hash(path, registry=registry) == hasher.hash(path, registry=registry)


def test_path_hasher_hash_equal_paths_same_hash(tmp_path: Path, registry: HasherRegistry) -> None:
    hasher = PathHasher()
    path_a = tmp_path / "file.txt"
    path_b = tmp_path / "file.txt"
    assert hasher.hash(path_a, registry=registry) == hasher.hash(path_b, registry=registry)


def test_path_hasher_hash_different_paths_different_hashes(
    tmp_path: Path, registry: HasherRegistry
) -> None:
    hasher = PathHasher()
    path_a = tmp_path / "a.txt"
    path_b = tmp_path / "b.txt"
    assert hasher.hash(path_a, registry=registry) != hasher.hash(path_b, registry=registry)


def test_path_hasher_hash_does_not_use_registry(tmp_path: Path, registry: HasherRegistry) -> None:
    empty_registry = HasherRegistry()
    hasher = PathHasher()
    path = tmp_path / "file.txt"
    assert hasher.hash(path, registry=registry) == hasher.hash(path, registry=empty_registry)


def test_path_hasher_hash_matches_hash_path(tmp_path: Path, registry: HasherRegistry) -> None:
    hasher = PathHasher()
    path = tmp_path / "file.txt"
    assert hasher.hash(path, registry=registry) == hash_path(path)


##############################
#     Tests for hash_path    #
##############################

# --- Return type and format ---


def test_hash_path_returns_str(tmp_path: Path) -> None:
    assert isinstance(hash_path(tmp_path / "file.txt"), str)


def test_hash_path_returns_lowercase_hex(tmp_path: Path) -> None:
    assert all(c in "0123456789abcdef" for c in hash_path(tmp_path / "file.txt"))


def test_hash_path_default_length_is_64(tmp_path: Path) -> None:
    assert len(hash_path(tmp_path / "file.txt")) == 64


@pytest.mark.parametrize(
    "length",
    [
        pytest.param(2, id="min"),
        pytest.param(16, id="16"),
        pytest.param(32, id="32"),
        pytest.param(64, id="64-default"),
        pytest.param(128, id="max"),
    ],
)
def test_hash_path_output_length_matches_requested(length: int, tmp_path: Path) -> None:
    assert len(hash_path(tmp_path / "file.txt", length=length)) == length


# --- Determinism ---


def test_hash_path_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "file.txt"
    assert hash_path(path) == hash_path(path)


def test_hash_path_equal_paths_same_hash(tmp_path: Path) -> None:
    assert hash_path(tmp_path / "file.txt") == hash_path(tmp_path / "file.txt")


# --- Sensitivity ---


def test_hash_path_different_paths_produce_different_hashes(tmp_path: Path) -> None:
    assert hash_path(tmp_path / "a.txt") != hash_path(tmp_path / "b.txt")


def test_hash_path_same_prefix_different_paths_produce_different_hashes(tmp_path: Path) -> None:
    assert hash_path(tmp_path / "file1.txt") != hash_path(tmp_path / "file2.txt")


def test_hash_path_is_case_sensitive(tmp_path: Path) -> None:
    assert hash_path(tmp_path / "File.txt") != hash_path(tmp_path / "file.txt")


# --- Resolution behavior ---


def test_hash_path_resolves_relative_segments(tmp_path: Path) -> None:
    direct = tmp_path / "file.txt"
    indirect = tmp_path / "sub" / ".." / "file.txt"
    assert hash_path(direct) == hash_path(indirect)


def test_hash_path_resolves_relative_to_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    relative = Path("file.txt")
    absolute = tmp_path / "file.txt"
    assert hash_path(relative) == hash_path(absolute)


def test_hash_path_resolve_failure_uses_posix_of_unresolved_path(tmp_path: Path) -> None:
    path = tmp_path / "file.txt"
    with patch.object(Path, "resolve", side_effect=OSError("boom")):
        result = hash_path(path)

    from coola.hashing import hash_string

    assert result == hash_string(path.as_posix())


# --- Invalid length ---


@pytest.mark.parametrize(
    "length",
    [
        pytest.param(0, id="zero"),
        pytest.param(1, id="odd-below-min"),
        pytest.param(3, id="odd-small"),
        pytest.param(63, id="odd-near-max"),
        pytest.param(130, id="above-max"),
        pytest.param(-2, id="negative-even"),
        pytest.param(-1, id="negative-odd"),
    ],
)
def test_hash_path_raises_for_invalid_length(length: int, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=str(abs(length))):
        hash_path(tmp_path / "file.txt", length=length)
