__all__ = ["str_dict", "str_indent"]

from typing import Any


def str_dict(data: dict[str, Any], sorted_keys: bool = False, indent: int = 0) -> str:
    r"""Converts a dict to a pretty string representation.

    This function was designed for flat dictionary. If you have a
    nested dictionary, you may consider other functions. Note that
    this function works for nested dict but the output may not be
    nice.

    Args:
        data (dict): Specifies the input dictionary.
        sorted_keys (bool, optional): Specifies if the key of the dict
            are sorted or not. Default: ``False``
        indent (int, optional): Specifies the indentation. The value
            should be greater or equal to 0. Default: ``0``

    Returns:
        str: The string representation.

        Example usage:

    .. code-block:: python

        >>> from coola.format import str_dict
        >>> str_dict({"my_key": "my_value"})
        'my_key : my_value'
        >>> str_dict({"key1": "value1", "key2": "value2"})
        'key1 : value1\nkey2 : value2'
    """
    if indent < 0:
        raise ValueError(f"The indent has to be greater or equal to 0 (received: {indent})")
    if not data:
        return ""

    max_length = max([len(key) for key in data.keys()])
    output = []
    for key in sorted(data.keys()) if sorted_keys else data.keys():
        output.append(f"{' ' * indent + str(key) + ' ' * (max_length - len(key))} : {data[key]}")
    return "\n".join(output)


def str_indent(original: Any, num_spaces: int = 2) -> str:
    r"""Add indentations if the original string is a multi-lines string.

    Args:
        original: Specifies the original string. If the inputis not a
            string, it will be converted to a string with the function
            ``str``.
        num_spaces (int, optional): Specifies the number of spaces
            used for the indentation. Default: ``2``.

    Returns:
        str: The indented string.

    Example usage:

    .. code-block:: python

        >>> from coola.format import str_indent
        >>> print(f"\t{str_indent('string1\nstring2', 4)}")
            string1
            string2
    """
    formatted_str = str(original).split("\n")
    if len(formatted_str) == 1:  # single line
        return formatted_str[0]
    first = formatted_str.pop(0)
    formatted_str = "\n".join([(num_spaces * " ") + line for line in formatted_str])
    return first + "\n" + formatted_str
