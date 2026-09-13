from __future__ import annotations

import pytest

from coola.utils.lru import LRUCache

##############################
#     Tests for LRUCache     #
##############################


def test_lru_cache_init_invalid_maxsize() -> None:
    with pytest.raises(ValueError, match=r"maxsize must be greater than 0"):
        LRUCache(maxsize=0)


def test_lru_cache_maxsize() -> None:
    assert LRUCache[str, int](maxsize=3).maxsize == 3


def test_lru_cache_len_empty() -> None:
    assert len(LRUCache[str, int](maxsize=3)) == 0


def test_lru_cache_setitem_getitem() -> None:
    cache = LRUCache[str, int](maxsize=3)
    cache["a"] = 1
    assert cache["a"] == 1
    assert len(cache) == 1


def test_lru_cache_getitem_missing_key_raises_keyerror() -> None:
    cache = LRUCache[str, int](maxsize=3)
    with pytest.raises(KeyError):
        _ = cache["missing"]


def test_lru_cache_setitem_updates_existing_key() -> None:
    cache = LRUCache[str, int](maxsize=3)
    cache["a"] = 1
    cache["a"] = 2
    assert cache["a"] == 2
    assert len(cache) == 1


def test_lru_cache_contains() -> None:
    cache = LRUCache[str, int](maxsize=3)
    cache["a"] = 1
    assert "a" in cache
    assert "b" not in cache


def test_lru_cache_iter_order() -> None:
    cache = LRUCache[str, int](maxsize=3)
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3
    assert list(cache) == ["a", "b", "c"]


def test_lru_cache_evicts_least_recently_used_on_insert() -> None:
    cache = LRUCache[str, int](maxsize=2)
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3  # evicts "a"
    assert "a" not in cache
    assert list(cache) == ["b", "c"]


def test_lru_cache_getitem_marks_key_as_most_recently_used() -> None:
    cache = LRUCache[str, int](maxsize=2)
    cache["a"] = 1
    cache["b"] = 2
    _ = cache["a"]  # "a" becomes most-recently-used
    cache["c"] = 3  # evicts "b", not "a"
    assert "b" not in cache
    assert "a" in cache
    assert "c" in cache


def test_lru_cache_setitem_existing_key_marks_as_most_recently_used() -> None:
    cache = LRUCache[str, int](maxsize=2)
    cache["a"] = 1
    cache["b"] = 2
    cache["a"] = 10  # "a" becomes most-recently-used
    cache["c"] = 3  # evicts "b", not "a"
    assert "b" not in cache
    assert cache["a"] == 10
    assert "c" in cache


def test_lru_cache_get_existing_key() -> None:
    cache = LRUCache[str, int](maxsize=3)
    cache["a"] = 1
    assert cache.get("a") == 1


def test_lru_cache_get_missing_key_returns_none() -> None:
    cache = LRUCache[str, int](maxsize=3)
    assert cache.get("missing") is None


def test_lru_cache_get_missing_key_returns_default() -> None:
    cache = LRUCache[str, int](maxsize=3)
    assert cache.get("missing", default=-1) == -1


def test_lru_cache_get_marks_key_as_most_recently_used() -> None:
    cache = LRUCache[str, int](maxsize=2)
    cache["a"] = 1
    cache["b"] = 2
    cache.get("a")  # "a" becomes most-recently-used
    cache["c"] = 3  # evicts "b", not "a"
    assert "b" not in cache
    assert "a" in cache


def test_lru_cache_delitem() -> None:
    cache = LRUCache[str, int](maxsize=3)
    cache["a"] = 1
    del cache["a"]
    assert "a" not in cache
    assert len(cache) == 0


def test_lru_cache_delitem_missing_key_raises_keyerror() -> None:
    cache = LRUCache[str, int](maxsize=3)
    with pytest.raises(KeyError):
        del cache["missing"]


def test_lru_cache_pop_existing_key() -> None:
    cache = LRUCache[str, int](maxsize=3)
    cache["a"] = 1
    assert cache.pop("a") == 1
    assert "a" not in cache


def test_lru_cache_pop_missing_key_raises_keyerror() -> None:
    cache = LRUCache[str, int](maxsize=3)
    with pytest.raises(KeyError):
        cache.pop("missing")


def test_lru_cache_pop_missing_key_with_default() -> None:
    cache = LRUCache[str, int](maxsize=3)
    assert cache.pop("missing", -1) == -1


def test_lru_cache_clear() -> None:
    cache = LRUCache[str, int](maxsize=3)
    cache["a"] = 1
    cache["b"] = 2
    cache.clear()
    assert len(cache) == 0
    assert list(cache) == []


def test_lru_cache_keys() -> None:
    cache = LRUCache[str, int](maxsize=3)
    cache["a"] = 1
    cache["b"] = 2
    assert list(cache.keys()) == ["a", "b"]


def test_lru_cache_values() -> None:
    cache = LRUCache[str, int](maxsize=3)
    cache["a"] = 1
    cache["b"] = 2
    assert list(cache.values()) == [1, 2]


def test_lru_cache_items() -> None:
    cache = LRUCache[str, int](maxsize=3)
    cache["a"] = 1
    cache["b"] = 2
    assert list(cache.items()) == [("a", 1), ("b", 2)]


def test_lru_cache_eq_true_same_content() -> None:
    cache1 = LRUCache[str, int](maxsize=3)
    cache1["a"] = 1
    cache2 = LRUCache[str, int](maxsize=10)
    cache2["a"] = 1
    assert cache1 == cache2


def test_lru_cache_eq_false_different_content() -> None:
    cache1 = LRUCache[str, int](maxsize=3)
    cache1["a"] = 1
    cache2 = LRUCache[str, int](maxsize=3)
    assert cache1 != cache2


def test_lru_cache_eq_true_dict() -> None:
    cache = LRUCache[str, int](maxsize=3)
    cache["a"] = 1
    assert cache == {"a": 1}


def test_lru_cache_eq_false_dict() -> None:
    cache = LRUCache[str, int](maxsize=3)
    cache["a"] = 1
    assert cache != {"a": 2}


def test_lru_cache_eq_false_other_type() -> None:
    cache = LRUCache[str, int](maxsize=3)
    assert cache != 42


def test_lru_cache_eq_empty() -> None:
    assert LRUCache[str, int](maxsize=3) == {}


def test_lru_cache_repr() -> None:
    assert repr(LRUCache[str, int](maxsize=3)).startswith("LRUCache(")


def test_lru_cache_str() -> None:
    assert str(LRUCache[str, int](maxsize=3)).startswith("LRUCache(")
