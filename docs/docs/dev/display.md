# Architecture of `coola.display`

:book: This page describes the design of the `coola.display` package for contributors: the invariants, the contracts between the parts, and the trade-offs of the main design choices. The rules that users rely on are in the [user guide](../uguide/display.md#properties); the API is in the [reference](../refs/display.md).

## Constraints

1. **Simple and easy to maintain.** The package is small and mostly docstrings. It does not walk two
   structures in parallel, and it has no layer that only exists for one caller.
2. **`rich` and `colorlog` are optional.** `import coola`, `import coola.display` and
   `import coola.display.rich` work without them and do not import them. Calling a function that needs
   a missing library raises `MissingPackageError` with the `pip install coola[rich]` hint, not a bare
   `ModuleNotFoundError`. The tests of `coola.display.rich` are skipped when `rich` is not installed
   instead of failing at collection.
3. **Core package.** `coola.display` sits in the core group (`registry`, `utils`, `validation`,
   `formatting`): it imports no feature package (`lint-imports` contract "Core packages must not
   import feature packages"). It may import `utils`, `validation` and `formatting`.
4. **Consistent with the siblings.** Options are keyword-only, arguments are validated with
   `coola.validation`, helpers that exist elsewhere are reused (`coola.formatting`), and the optional
   libraries follow the pattern of `coola.utils.fallback`.

## Architecture

```
coola/display/
    __init__.py       BaseDisplayMixin, InlineDisplayMixin, MultilineDisplayMixin (eager, no dependency)
                      str_pydantic_model, repr_pydantic_model                     (lazy, pydantic)
    mixin.py          the three mixins
    pydantic.py       format a pydantic model without its secrets                   imports pydantic
    colorlog.py       configure_colorlog_logging                                    imports colorlog
    rich/
        __init__.py   the five public names, lazy                                  imports nothing
        logging.py    configure_rich_logging                                        imports rich
        print_.py     print_markdown, print_pretty                                  imports rich
        progressbar.py make_progressbar, make_spinner                               imports rich
coola/utils/fallback/rich.py   stand-ins for the names of rich that the three modules import
```

The package has three kinds of content with one rule each:

| Kind                | Dependency         | Rule                                                                                                                                |
| ------------------- | ------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| Mixins              | none               | imported eagerly; used by other packages.                                                                                           |
| Pydantic formatting | `pydantic`         | the functions are exported lazily from `coola.display`; the module guards its import.                                               |
| Terminal output     | `rich`, `colorlog` | `coola.display.rich` exports lazily like `coola.display`; each module guards its import; a function never runs without its library. |

### The rule for an optional library

`pydantic.py`, `colorlog.py` and the `rich` modules follow the same rule:

1. **The package `__init__` exports lazily.** `coola/display/rich/__init__.py` uses `lazy_exports`
   (as `coola/display/__init__.py` does for the pydantic functions). `import coola.display.rich`
   imports no module of `rich`; `from coola.display.rich import print_pretty` imports
   `coola.display.rich.print_`.
2. **A module guards its import** with the fallback of `coola.utils.fallback`:

   ```python
   if TYPE_CHECKING or is_rich_available():  # pragma: no cover
       from rich import get_console
       from rich.markdown import Markdown
       ...
   else:  # pragma: no cover
       from coola.utils.fallback.rich import Markdown, Panel, Pretty, get_console, ...
   ```

   `import coola.display.rich.print_` therefore never fails.

3. **The fallback fails at the first use**, with a `MissingPackageError` whose message says that the
   `rich` package is required and gives the hint ``pip install coola[rich]`` (the message of
   `raise_package_missing_error`). `coola/utils/fallback/rich.py` (about 25 lines, built with
   `make_fake_class` and `make_fake_function`) defines `get_console`, `RichHandler`, `Markdown`,
   `Panel`, `Pretty`, `Progress` and the six columns as fakes and `MINIMAL`, `ROUNDED` as `None`.
   Every public function calls a fake in its first statement (`Pretty(data)`, `SpinnerColumn()`,
   `get_console()`), before it changes anything: `configure_rich_logging` does not touch the root
   logger when `rich` is missing.
4. **`check_rich()` is not called.** The fallback is the single mechanism, as for `pydantic`; a second
   check would be dead code. (`check_rich` stays in `coola.utils.imports` for the other packages.)
5. **`colorlog` works without the library.** The function falls back to plain `logging.basicConfig`,
   because for a logging helper "no colour" is a correct result and not an error. This is the
   difference between the two logging functions: `rich` is required, `colorlog` is optional.

Why the fallback and not an import inside each function: it is the convention of the repository
(`reducer`, `rounding`, `utils.tensor`, `display.pydantic`, `display.colorlog`), the names stay at the
top of the module where `ruff` and `ty` see them, and the tests need no patching of
`builtins.__import__`. The cost is one file of 25 lines to keep in sync with the imported names. The
alternative, `check_rich()` and `from rich import ...` inside each function, needs no fake names but
is a second convention and hides the imports in function bodies.

### The mixins

`BaseDisplayMixin` defines `__repr__`, `__str__` and a default `_display_fields`; the two concrete
mixins implement `_render_fields`. `MultilineDisplayMixin` uses `repr_indent(repr_mapping(fields))`,
which indents the continuation lines of a multiline value (`(y): a\n  b`). This is intended and covered
by a test.

The mixins do not guard against self-reference. A `reprlib.recursive_repr`-style guard needs a per-call
set of ids on every `__repr__`, for a case that does not occur in the objects that use the mixins
(configurations, loaders, transformers, registries); it would also change `__str__` and hide a bug in
the caller's `_display_fields`.

### Formatting a pydantic model

The function formats the **fields of the model**, not a dump of it. One private function builds the
mapping, the two public functions choose `repr` or `str`:

```python
def _format_model(model, *, line_fn, sort, exclude_none, exclude_secret, exclude_fields) -> str:
    fields = {name: getattr(model, name) for name, info in type(model).model_fields.items()
              if info.repr}                                   # Field(repr=False) is honoured
    fields |= model.model_extra or {}                         # extra="allow" values
    if exclude_secret:
        for name in secret_field_names(model): fields.pop(name, None)
    ...exclude_none, exclude_fields (warn on unknown names), sort...
    return f"{type(model).__qualname__}({line_fn({k: _shown(v) for k, v in fields.items()})})"
```

with one small recursive helper for the values:

```python
def _shown(value):
    """A nested model is formatted like the root; lists, tuples, sets and dicts are rebuilt."""
```

- **Nested models keep their name** (`P(c=C(name=a), items=[C(name=b)])`). A nested model is formatted
  with the same options as the root (`sort`, `exclude_none`, `exclude_secret`); the shown value is a
  tiny object whose `__repr__`/`__str__` is the formatted string, so the generic `repr_mapping_line`
  and `str_mapping_line` of `coola.formatting` need no change.
- **No parallel walk, no `cast`**: the value is read once, from the model, and a nested secret is
  dropped by the recursive call on the nested model. Dumping the model (`model_dump()`) and masking the
  secrets in the dump would need a second walk over the model and the dump in parallel.
- **No serializer runs.** `getattr` reads the attribute; a `field_serializer`, an alias or a
  `computed_field` has no effect (a `computed_field` is not a field, so it is not shown; pydantic's
  `repr` shows it, the function does not: it is a property that may be expensive). The function is a
  safe formatter for logs, whose value is that it cannot raise or leak through user code. Keeping
  `model_dump()` would keep `computed_field` at the price of the parallel walk and of running the
  serializers.
- **`secret_field_names` and `is_secret_annotation` are public in `coola.display.pydantic`.** They are
  the definition of "secret" shared by `display` and `summary`, with their own tests and doctests.
  A separate `coola.pydantic` module for the pydantic knowledge shared by several packages is not
  justified for two functions.
- **One core, two wrappers.** `str_pydantic_model` and `repr_pydantic_model` mirror `str`/`repr` and
  `str_mapping`/`repr_mapping` of `formatting`. The five options are documented once in
  `str_pydantic_model` and `repr_pydantic_model` points to it ("same options"). A generic
  `format_pydantic_model(model, *, use_repr=...)` would not remove code, since the wrappers are five
  lines over the private function.

### Terminal output

- **`configure_colorlog_logging(level=logging.INFO, *, force=False, **kwargs)`.** It accepts the
  plain `logging.basicConfig` options (`handlers`, `filename`, `filemode`, `format`, `style`), handles
  `stream` and `datefmt`, and checks that the stream is a terminal before it colours.
- **`configure_rich_logging(level=logging.INFO, *, fmt, datefmt, rich_tracebacks, markup, force,
console=None, **kwargs)`.** The handler is `RichHandler(console=console or get_console())`;
  `**kwargs` go to `RichHandler` and an unknown name raises `TypeError`.
- **`make_progressbar(*, transient=False, console=None)` / `make_spinner(*, transient=True,
console=None)`** take the same `console` argument.
- **`print_pretty` / `print_markdown`** take `title`, `title_align`, `box`, `panel` and `console`.
  `print_markdown(max_length=...)` truncates a Markdown string. `max_length` is validated
  (`validate_ge(max_length, 1)`, so a negative value raises `ValueError` instead of cutting from the
  end) and the cut is done by `_truncate_markdown(msg, max_length)`, which stays in `print_.py` (it is
  about Markdown, so it does not belong in `formatting.text`; `truncate_str` is not used because it
  cuts at a character and appends a suffix). The cut:
  1. snaps to the last paragraph break (`\n\n`), else the last line break, else the last space, of the
     kept text, each only if it keeps at least half of `max_length`; otherwise it cuts at
     `max_length`;
  2. closes a code fence left open by the cut (an odd number of lines that start with ``` or `~~~`
     gets a closing fence), so the rest of the panel is not swallowed by an open block;
  3. appends the notice `*… (N more characters)*` as a separate paragraph (`N` is the number of
     characters of `msg` that are not shown).

  The cut is not guaranteed to keep an inline marker (`**`, `*`, `` ` ``) balanced when it falls
  inside a paragraph: the preference for paragraph and line breaks makes it rare, and it is documented
  instead of repaired with a heuristic. A plain character cut would be simpler but can drop the closing
  `**` or a code fence.

- **The `rich` modules are four files** (`logging`, `print_`, `progressbar`, `__init__`). Merging them
  would not shorten them: the three cover three different `rich` APIs and the split keeps each import
  list short.
- **Two logging functions, not one `configure_logging(backend=...)`.** They have different options
  (`rich_tracebacks`, `markup` against the colour of the formatter), and a dispatcher adds a name and
  a string option for two lines of caller code.

## Public API

```python
from coola.display import (
    BaseDisplayMixin,
    InlineDisplayMixin,
    MultilineDisplayMixin,
    repr_pydantic_model,  # (model, *, sort=True, exclude_none=False, exclude_secret=True, exclude_fields=None)
    str_pydantic_model,  # same options
)
from coola.display.pydantic import is_secret_annotation, secret_field_names
from coola.display.colorlog import (
    configure_colorlog_logging,
)  # (level=INFO, *, force=False, **kwargs)
from coola.display.rich import (
    configure_rich_logging,  # (level=INFO, *, fmt, datefmt, rich_tracebacks, markup, force, console=None, **kwargs)
    make_progressbar,  # (*, transient=False, console=None)
    make_spinner,  # (*, transient=True, console=None)
    print_markdown,  # (msg, *, title, title_align, box, panel, max_length=None, console)
    print_pretty,  # (data, *, title, title_align, box, panel, console)
)
```

Behaviours worth knowing:

| Situation                                                       | Behaviour                                                                                         |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `import coola.display.rich` without `rich`                      | the import works; the first call raises `MissingPackageError` (`pip install coola[rich]`)          |
| `import coola.display.rich` with `rich`                         | it imports no `rich` module until a name is used                                                  |
| `configure_colorlog_logging(logging.INFO, True)`                | `TypeError`: `force` is keyword-only                                                              |
| `str_pydantic_model(P(c=C(...)))`                               | shows `c=C(name=a)`: a nested model has its name, with the options of the root                    |
| a field with `Field(repr=False)`                                | hidden, like in pydantic's `repr`; no user code runs                                              |
| a `computed_field`                                              | not shown (it is not a field)                                                                     |
| `print_markdown(msg, max_length=-1)`                            | `ValueError` for `max_length < 1`; the cut prefers paragraph and line breaks and closes a fence   |
| `configure_rich_logging`, `make_*`                              | write to `get_console()` unless `console=` chooses another one                                    |

## Dependents

| Package                                  | Use                                                                                    |
| ---------------------------------------- | -------------------------------------------------------------------------------------- |
| `io`, `recursive`, `hashing`, `registry` | `InlineDisplayMixin` and `_display_fields`.                                            |
| `summary.pydantic`                       | `secret_field_names`.                                                                  |
| `coola.utils.fallback`                   | `rich.py` holds the fakes.                                                             |
| `coola.testing.marks`                    | `rich_available` and `colorlog_available` skip the `rich` and `colorlog` tests.        |

## Not covered

- **A `Table` or `Tree` helper** for `rich` (for the summary of `coola.summary`): out of scope;
  `print_pretty` takes any object.
- **Theming of the colours of `colorlog`** (the `log_colors` mapping is fixed): add an option when a
  user asks.
- **Log in JSON or to a file**: `configure_*_logging` forward `handlers`, `filename`, `format` to
  `logging.basicConfig`; nothing else is provided.
- **Moving the logging helpers out of `display`** (to `coola.utils` or a `coola.logging` package):
  `display` is where the user looks for terminal output and the two modules are small; revisit if the
  package grows a third logging backend.
- **`coola.utils.logging.CaptureHandler`** is a test helper for loggers and is unrelated to the
  display of records.
