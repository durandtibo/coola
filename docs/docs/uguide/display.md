# Display Utilities

:book: This page describes the `coola.display` package, which provides shared helpers to build
readable `__repr__`/`__str__` implementations, to format pydantic models without their secrets, and
to write to the terminal with `rich` and `colorlog`.

**Prerequisites:** You'll need to know a bit of Python.
For a refresher, see the [Python tutorial](https://docs.python.org/tutorial/).

## Overview

The `coola.display` package has three kinds of content, with one rule each:

| Kind                | Names                                                                                                                        | Dependency         | Rule                                                                                              |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ------------------ | ------------------------------------------------------------------------------------------------- |
| Mixins              | `InlineDisplayMixin`, `MultilineDisplayMixin`, `BaseDisplayMixin`                                                            | none               | Imported eagerly; a class only overrides `_display_fields`.                                       |
| Pydantic formatting | `str_pydantic_model`, `repr_pydantic_model`                                                                                  | `pydantic`         | Exported lazily from `coola.display`; a missing library raises `MissingPackageError` when called. |
| Terminal output     | `configure_colorlog_logging`; `configure_rich_logging`, `make_progressbar`, `make_spinner`, `print_markdown`, `print_pretty` | `colorlog`, `rich` | `coola.display.rich` exports lazily; a function never runs without its library (see below).       |

## Properties

These are the guarantees of the package. Each one is enforced by the code, not by the discipline of
the callers.

| Property             | Statement                                                                                                                                                                                                                           |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| One hook             | A class that uses a mixin only overrides `_display_fields`. `__repr__` and `__str__` come from the mixin; a class with no field overrides nothing.                                                                                  |
| Consistent format    | All the objects that use a mixin share the same layout: `Name(key=value)` inline, or one `(key): value` per line. `__repr__` shows the `repr` of the values and `__str__` their `str`. The name is the `__qualname__` of the class. |
| Pure                 | Formatting never modifies the object and has no side effect (no logging, no I/O, no global state). Two calls on the same object give the same string.                                                                               |
| Secrets never leak   | A pydantic field typed as a secret (`SecretStr`, `SecretBytes`, `Secret`, or a type that contains one) is omitted by default, at any depth. With `exclude_secret=False` the masked value of pydantic is shown, never the secret.    |
| Honest options       | An option that names a field that does not exist (`exclude_fields`) warns, at the line of the caller. An option out of its range raises `ValueError`.                                                                               |
| Optional libraries   | `pydantic`, `rich` and `colorlog` are imported when they are used, never when `coola` or `coola.display` is imported. A missing library raises `MissingPackageError` when a function that needs it is called, never at import time. |
| Safe terminal output | Colour is used only if the library is installed **and** the output is a terminal (`colorlog`), or the library decides (`rich` strips the styles of a file). Otherwise the output is plain.                                          |
| Respects the logger  | `logging.basicConfig` does nothing if the root logger already has handlers, unless `force=True`. Both `configure_*_logging` functions have the same options for this: `level`, `force` and `**kwargs` forwarded.                    |
| One console          | The `rich` helpers (`print_*`, progress bars, logging) write to the console of `rich` (`rich.get_console()`) unless a `console` is passed, so their output does not interleave.                                                     |
| Keyword-only options | Every option after the first required arguments (and `level` for the logging functions) is keyword-only.                                                                                                                            |

Not guaranteed:

- **Self-referencing objects.** `repr` of an object whose `_display_fields` contains itself raises
  `RecursionError`, like any `__repr__` that recurses.
- **The layout of a multiline value.** A value whose `str` has several lines is shown with its lines
  indented under the key; this is a readability rule, not a format to parse.
- **The look of the `rich` output** (box characters, colours, widths) depends on the terminal and
  on the version of `rich`.

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

## Formatting pydantic models

`str_pydantic_model` and `repr_pydantic_model` produce a single-line, sorted representation of
a pydantic model. Fields typed as `SecretStr` (including `Optional[SecretStr]`) are omitted by
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

The two functions have the same options and use `str` or `repr` for the values:

- `sort` (default `True`) sorts the fields by name.
- `exclude_none=True` omits the fields whose value is `None`.
- `exclude_secret=False` shows the value masked by pydantic instead of omitting the field.
- `exclude_fields` omits fields by name. A name that is not a field of the model emits a
  `RuntimeWarning`.

The fields are read from the model: no `field_serializer` runs, a field declared with
`Field(repr=False)` is hidden (as in the `repr` of pydantic), the values of `extra="allow"` are
shown, and a `computed_field` is not shown because it is not a field. A nested model keeps its name
and is formatted with `sort`, `exclude_none` and `exclude_secret` of the root model, also inside
lists, tuples, sets and dicts:

```pycon

>>> from pydantic import BaseModel, SecretStr
>>> from coola.display import str_pydantic_model
>>> class Database(BaseModel):
...     host: str
...     password: SecretStr
>>> class Config(BaseModel):
...     name: str
...     database: Database
>>> str_pydantic_model(Config(name="my-app", database=Database(host="db", password="s3cr3t")))
'Config(database=Database(host=db), name=my-app)'

```

`coola.display.pydantic` also provides `secret_field_names(model)` and `is_secret_annotation(annotation)`,
which define what a secret is for `display` and `summary`.

## Printing with rich

`coola.display.rich` needs the optional [`rich`](https://pypi.org/project/rich/) package
(`pip install rich`). `import coola.display.rich` works without it and imports nothing; the first
call of a function raises `MissingPackageError`. The functions write to `rich.get_console()` unless
a `console` is passed:

```pycon

>>> from io import StringIO
>>> from rich.console import Console
>>> from coola.display.rich import print_markdown, print_pretty
>>> console = Console(file=StringIO(), width=20)
>>> print_pretty({"key": "value"}, panel=False, console=console)
>>> console.file.getvalue().strip()
"{'key': 'value'}"

```

- `print_pretty(data, *, title, title_align, box, panel, console)` renders any object with
  `rich.pretty.Pretty`.
- `print_markdown(msg, *, title, title_align, box, panel, max_length, console)` renders a Markdown
  string. `max_length` (at least 1) keeps the beginning of a long message: the cut prefers a
  paragraph break, then a line break, then a space (each only if it keeps at least half of
  `max_length`), closes a code fence left open, and appends the number of characters that are not
  shown. An inline marker such as `**` can be left unbalanced when the cut falls inside a paragraph.
- `make_progressbar(*, transient, console)` and `make_spinner(*, transient, console)` return a
  `rich.progress.Progress` with a standard layout, to use as a context manager.
- `configure_rich_logging(level, *, fmt, datefmt, rich_tracebacks, markup, force, console, **kwargs)`
  configures the root logger with a `rich.logging.RichHandler` (`**kwargs` go to the handler). The
  root logger is not changed if `rich` is missing.

## Configuring colored logging

`configure_colorlog_logging(level, *, force, **kwargs)` configures the root logger, using a colored
formatter when the optional [`colorlog`](https://pypi.org/project/colorlog/) dependency is installed
and the output is attached to a terminal. It falls back to plain `logging.basicConfig` otherwise
(e.g. when output is redirected to a file, or running in CI), to avoid emitting raw ANSI escape codes
into non-interactive output:

```pycon

>>> from coola.display.colorlog import configure_colorlog_logging
>>> configure_colorlog_logging()

```

`colorlog` is optional and `rich` is required, because for a logging helper "no colour" is a correct
result and a missing `rich` is not.

!!! note
`logging.basicConfig` is a no-op if the root logger already has handlers configured. Pass
`force=True` to remove existing handlers and reconfigure unconditionally.

See the [reference](../refs/display.md) for the complete API.
