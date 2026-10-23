# Display Utilities

:book: This page describes the `coola.display` package, which provides shared helpers to build
readable `__repr__`/`__str__` implementations and to configure colored logging output.

**Prerequisites:** You'll need to know a bit of Python.
For a refresher, see the [Python tutorial](https://docs.python.org/tutorial/).

## Overview

The `coola.display` package provides:

- **`InlineDisplayMixin`**: a mixin that renders `__repr__`/`__str__` on a single line
- **`MultilineDisplayMixin`**: a mixin that renders `__repr__`/`__str__` as an indented,
  multiline block
- **`BaseDisplayMixin`**: the abstract base class of the mixins, which defines `__repr__`, `__str__` and
  the `_display_fields` hook
- **`str_pydantic_model`** / **`repr_pydantic_model`**: formatted, single-line string/repr
  representations of [pydantic](https://docs.pydantic.dev/) models, with automatic masking of
  `SecretStr` fields
- **`configure_colorlog_logging`**: configure the root logger with colored output when
  [`colorlog`](https://pypi.org/project/colorlog/) is installed

## Display mixins

`InlineDisplayMixin` and `MultilineDisplayMixin` remove the boilerplate of writing consistent
`__repr__`/`__str__` methods. Subclasses only need to override `_display_fields`, which
returns a dict mapping field names to their values. The default returns an empty dict, so a
stateless class needs no override.

`InlineDisplayMixin` is best suited for simple objects with a few, short arguments:

```pycon

>>> from coola.display import InlineDisplayMixin
>>> from typing import Any
>>> class MyClass(InlineDisplayMixin):
...     def __init__(self, key1: str, key2: str) -> None:
...         self.key1 = key1
...         self.key2 = key2
...
...     def _display_fields(self) -> dict[str, Any]:
...         return {"key1": self.key1, "key2": self.key2}
>>> obj = MyClass(key1="value1", key2="value2")
>>> print(repr(obj))
MyClass(key1='value1', key2='value2')
>>> print(str(obj))
MyClass(key1=value1, key2=value2)

```

`MultilineDisplayMixin` is best suited for objects with several arguments or deeply nested
values, where a single-line format would be hard to read:

```pycon

>>> from coola.display import MultilineDisplayMixin
>>> from typing import Any
>>> class MyClass(MultilineDisplayMixin):
...     def __init__(self, key1: str, key2: str) -> None:
...         self.key1 = key1
...         self.key2 = key2
...
...     def _display_fields(self) -> dict[str, Any]:
...         return {"key1": self.key1, "key2": self.key2}
>>> obj = MyClass(key1="value1", key2="value2")
>>> print(repr(obj))
MyClass(
  (key1): value1
  (key2): value2
)

```

A class with no field to show does not need to override anything:

```pycon

>>> from coola.display import InlineDisplayMixin
>>> class MyClass(InlineDisplayMixin):
...     pass
>>> print(repr(MyClass()))
MyClass()

```

To create a new kind of mixin, inherit from `BaseDisplayMixin` and implement `_render_fields`,
which receives the fields and a `use_repr` flag and returns the formatted string.
`__repr__` and `__str__` are already defined by `BaseDisplayMixin`.

## Properties

| Property             | Statement                                                                                                         |
| -------------------- | ----------------------------------------------------------------------------------------------------------------- |
| One hook             | A subclass only overrides `_display_fields`. `__repr__` and `__str__` come from the mixin.                        |
| Consistent format    | All the objects that use a mixin share the same layout: `Name(key=value)` inline, or one `(key): value` per line. |
| `repr` vs `str`      | `__repr__` shows the `repr` of the values, and `__str__` shows their `str`.                                       |
| Safe terminal output | The colored logging is used only if `colorlog` is installed and the output is a terminal. Otherwise it is plain.  |
| Respects the logger  | `logging.basicConfig` does nothing if the root logger already has handlers, unless `force=True`.                  |

## Formatting pydantic models

`str_pydantic_model` and `repr_pydantic_model` produce a single-line, sorted representation of
a pydantic model. Fields typed as `SecretStr` (including `Optional[SecretStr]`) are masked by
default, which makes them safe to use in logs:

```pycon

>>> from pydantic import BaseModel, SecretStr
>>> from coola.display import str_pydantic_model
>>> class Config(BaseModel):
...     name: str
...     token: SecretStr
>>> str_pydantic_model(Config(name="my-app", token="s3cr3t"))
'Config(name=my-app)'

```

Use `exclude_none=True` to omit fields whose value is `None`, `exclude_fields` to omit specific
fields by name, or `exclude_secret=False` to include the masked `SecretStr` value instead of
dropping the field entirely.

## Configuring colored logging

`configure_colorlog_logging` configures the root logger, using a colored formatter when the
optional [`colorlog`](https://pypi.org/project/colorlog/) dependency is installed and the
output is attached to a terminal. It falls back to plain `logging.basicConfig` otherwise (e.g.
when output is redirected to a file, or running in CI), to avoid emitting raw ANSI escape codes
into non-interactive output:

```pycon

>>> from coola.display.colorlog import configure_colorlog_logging
>>> configure_colorlog_logging()

```

!!! note
`logging.basicConfig` is a no-op if the root logger already has handlers configured. Pass
`force=True` to remove existing handlers and reconfigure unconditionally.

See the [reference](../refs/display.md) for the complete API.
