from __future__ import annotations

from coola.nested import convert_to_dict_of_lists, convert_to_list_of_dicts

##############################################
#     Tests for convert_to_dict_of_lists     #
##############################################


def test_convert_to_dict_of_lists_empty_list() -> None:
    assert convert_to_dict_of_lists([]) == {}


def test_convert_to_dict_of_lists_empty_dict() -> None:
    assert convert_to_dict_of_lists([{}]) == {}


def test_convert_to_dict_of_lists() -> None:
    assert convert_to_dict_of_lists(
        [{"key1": 1, "key2": 10}, {"key1": 2, "key2": 20}, {"key1": 3, "key2": 30}]
    ) == {
        "key1": [1, 2, 3],
        "key2": [10, 20, 30],
    }


##############################################
#     Tests for convert_to_list_of_dicts     #
##############################################


def test_convert_to_list_of_dicts_empty_dict() -> None:
    assert convert_to_list_of_dicts({}) == []


def test_convert_to_list_of_dicts_empty_list() -> None:
    assert convert_to_list_of_dicts({"key1": [], "key2": []}) == []


def test_convert_to_list_of_dicts() -> None:
    assert convert_to_list_of_dicts({"key1": [1, 2, 3], "key2": [10, 20, 30]}) == [
        {"key1": 1, "key2": 10},
        {"key1": 2, "key2": 20},
        {"key1": 3, "key2": 30},
    ]
