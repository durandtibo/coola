# Architecture of `coola.factory`

:book: This page describes the design of the `coola.factory` package for contributors: the invariants, the contracts between the parts, and the decisions behind them. The rules that users rely on are in the [user guide](../../uguide/factory.md#properties); the API is in the [reference](../../refs/factory.md).

## Overview

The package builds an object from a fully qualified name. It is small on purpose and has no state:
it keeps no registry, cache or singleton, and it imports the standard library and
`coola.utils.imports` only. A configuration names the code to run, so every change is judged first
by what the data can make the process do. The three entry points (`factory`, `resolve_object` and
`is_object_config`) are compositions of functions that each own one invariant: `import_object` is the
only place that imports a target (allow-list checked before the import, one `ImportError` for a
missing target), and `instantiate_object` is the only place that calls one (abstract classes
refused, the `_init_` rule, the target's keywords never shadowed by a parameter of the API). The
allow-list is a filter on a path and always comes from the caller's code, never from the data. See
the [user guide](../../uguide/factory.md#properties) for the properties of the package.

## Design constraints

1. **Simple and easy to maintain.** The package is 649 lines, 214 of them code. The target removes
   duplicated logic (three copies of "what does this target build", two imports of the same target,
   the dict-subclass special case, the `__new__` branch) instead of adding code next to it.
2. **Major release.** Breaking changes are fine and compatibility is not a design input: no adapter,
   no alias, no deprecation period. The release notes list the changes
   (`docs/docs/migration/v1.1_to_v2.0.md`).
3. **Security is the reason the package exists.** A configuration names the code to run. Every
   change is judged first by "what can the data make the process do", then by simplicity.
4. **No new dependency, no state, no optional library.** The package imports the standard library
   and `coola.utils.imports` only. It keeps no registry, no cache and no singleton: there is nothing
   to lock, reset or fork.
5. **Same errors from every entry point.** `factory`, `resolve_object` and `is_object_config` must
   report the same cause with the same exception and the same message.

## Architecture in detail

```
factory(_target_, *args, _init_="__init__", _allowed_prefixes_=None, **kwargs)
resolve_object(obj, cls=object, *, allowed_prefixes=None, strict=False)
is_object_config(config, cls, *, allowed_prefixes=None, import_target=False)
        |  three entry points; none of them imports, checks or calls on its own
        v
import_object(path, *, allowed_prefixes)     coola/factory/instantiation.py
        |  the only place that imports a target; one "does not exist" error
        v
builds(target, cls) -> True | False | None   coola/factory/target.py
        |  the only place that reads a return hint: class, function, union, generic
        v
instantiate_object(target, /, *args, _init_, **kwargs)     coola/factory/instantiation.py
           the only place that calls a target: class (constructor or public class method) or callable
```

The three entry points are compositions of three functions, each of which owns one invariant:

| Function             | Owns                                                                                                                        | Used by                                       |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| `import_object`      | the allow-list (before the import), the `ImportError` for a missing module or attribute (and only for the target itself)    | the three entry points                        |
| `builds`             | the answer to "does this target build a `cls`?": `True`, `False`, or `None` when it cannot be known                         | `resolve_object` (refuse), `is_object_config` |
| `instantiate_object` | how a target is called: abstract classes refused, `_init_` rule, result type of an alternative constructor, positional-only | `factory`, `resolve_object`                   |

### `constants.py`

```python
OBJECT_INIT = "_init_"
OBJECT_PREFIXES = "_allowed_prefixes_"  # new: the literal was repeated in three places
OBJECT_TARGET = "_target_"
```

### `import_object` (`instantiation.py`)

One change in behaviour: the translation that `factory` does before moves into the function, so that
every caller gets the same error.

```python
module_name, _, attr = object_path.rpartition(".")
try:
    module = importlib.import_module(module_name or attr)
except ModuleNotFoundError as err:
    # Only the module of the target (or one of its parents) is "missing". A dependency of the
    # target module that is not installed keeps its own error.
    if err.name and (object_path == err.name or object_path.startswith(err.name + ".")):
        msg = f"The target object does not exist: {object_path}"
        raise ImportError(msg, name=err.name) from err
    raise
if not module_name:  # "collections": a module
    return module
try:
    return getattr(module, attr)
except AttributeError as err:
    msg = f"The target object does not exist: {object_path}"
    raise ImportError(msg, name=object_path) from err
```

The checks before the import (not a string, allow-list, empty module name) are unchanged. `factory`
loses its `try/except ImportError` block (12 lines).

### `instantiate_object` (`instantiation.py`)

One public function, no private `_instantiate_class_object`, the first parameter positional-only
, the `_init_` rule:

```python
def instantiate_object(
    obj: Callable[..., Any] | type, /, *args: Any, _init_: str = "__init__", **kwargs: Any
) -> Any:
    if not callable(obj):
        raise TypeError(f"Incorrect type: {obj}. The valid types are class and callable")
    if not isinstance(obj, type):
        return obj(*args, **kwargs)  # function, partial, callable instance: `_init_` is ignored
    if inspect.isabstract(obj):
        raise TypeError(f"Cannot instantiate the class {obj} because it is an abstract class.")
    if _init_ == "__init__":
        return obj(*args, **kwargs)
    if not isinstance(_init_, str) or _init_.startswith("_"):
        raise TypeError(
            f"`_init_` must be '__init__' or the name of a public class method (received: {_init_!r})"
)
    method = getattr(obj, _init_, None)
    if not callable(method):
        raise TypeError(f"{obj} does not have a callable `{_init_}` attribute")
    result = method(*args, **kwargs)
    if not isinstance(result, obj):
        raise TypeError(
            f"`{_init_}` of {obj} did not return an instance of {obj} (received: {result!r})"
)
    return result
```

The positional-only `obj` makes `instantiate_object(C, obj=1, cls=2)` work: both go to the target. The
`"__new__"` branch and its `Warning` are deleted.

### `builds` (new module `target.py`)

```python
def builds(target: Any, cls: type | tuple[type, ...]) -> bool | None:
    """Indicate if calling ``target`` builds an instance of ``cls``, from the types only.

    Returns ``True`` if the classes that ``target`` builds include a subclass of ``cls``, ``False``
    if none of them is, and ``None`` if it cannot be known (a built-in function, a function without
    return hint, a hint that cannot be resolved or that is not a class, a union with such a member).
    """
    if inspect.isfunction(target):
        try:
            target = get_type_hints(target).get("return")
        except (NameError, SyntaxError, TypeError):
            return None
    if target is None:
        return None
    members = get_args(target) if get_origin(target) in (Union, UnionType) else (target,)
    classes = tuple(get_origin(member) or member for member in members)  # list[int] -> list
    if not all(isinstance(c, type) for c in classes):
        return None
    return any(issubclass(c, cls) for c in classes)
```

It is a public function of a module (not `_builds` imported across modules: `CLAUDE.md`), with its
own tests, and it is **not** exported by `coola.factory` (an implementation helper shared by two
modules of the package). It replaces `_check_target`'s hint code and the second half of
`is_object_config`.

### `factory`, `resolve_object`, `is_object_config` (`resolve.py`, `config.py`)

```python
def factory(
    _target_: str,
    *args,
    _init_: str = "__init__",
    _allowed_prefixes_: Sequence[str] | None = None,
    **kwargs,
) -> Any:
    if _allowed_prefixes_ is not None:
        _refuse_nested_configurations(args, kwargs)
    target = import_object(_target_, allowed_prefixes=_allowed_prefixes_)
    return instantiate_object(target, *args, _init_=_init_, **kwargs)


def resolve_object(obj, cls=object, *, allowed_prefixes=None, strict=False):
    if isinstance(obj, dict) and OBJECT_TARGET in obj:  # the key is reserved: it is a configuration
        obj = _build(obj, cls, allowed_prefixes=allowed_prefixes, strict=strict)
    if not isinstance(obj, cls):
        raise TypeError(_not_an_instance_message(obj, cls))
    return obj


def _build(config, cls, *, allowed_prefixes, strict):
    kwargs = {k: v for k, v in config.items() if k not in _RESERVED}  # the input is not modified
    init = config.get(OBJECT_INIT, "__init__")
    if allowed_prefixes is not None:
        _refuse_nested_configurations((), kwargs)
    target = import_object(config[OBJECT_TARGET], allowed_prefixes=allowed_prefixes)
    if cls is not object and init == "__init__":
        built = builds(target, cls)
        if built is False or (built is None and strict):
            raise TypeError(
                ...
)  # the two messages of the previous version, naming the target; "it was not called"
    return instantiate_object(target, _init_=init, **kwargs)


def is_object_config(config, cls, *, allowed_prefixes=None, import_target=False):
    if not isinstance(config, dict) or not isinstance(config.get(OBJECT_TARGET), str):
        return False
    path = config[OBJECT_TARGET]
    if not (import_target or allowed_prefixes is not None or is_module_imported(path)):
        return False
    return builds(import_object(path, allowed_prefixes=allowed_prefixes), cls) is True
```

Notes on the sketch:

- `resolve_object` no longer calls `factory`. `factory` is the `**`-friendly API for code that has
  the arguments; `resolve_object` is the API for configurations, and the check between the import and
  the call needs the imported object. The duplication is the four lines of `_build`.
- `_RESERVED = (OBJECT_TARGET, OBJECT_INIT, OBJECT_PREFIXES)`. `_allowed_prefixes_` in a configuration
  is dropped silently (documented): the policy is the caller's.
- `_refuse_nested_configurations(args, kwargs)` raises `ValueError` if a value of the arguments is a
  `dict` with `_target_`. Only the top level is scanned.
- `_not_an_instance_message` adds, for a `dict`, "(a configuration needs the `_target_` key)", so
  the useful part of the previous "missing the `_target_` key" error is kept in the single `TypeError`.
- An explicit `_init_="__init__"` is checked like the default (before: the presence of the key skips the
  check).
- The `logger.info("Initializing a ...")` line is kept (the siblings log at the same level).

### Error summary (the contract that the tests assert)

| Situation                                              | Exception     | Raised before the call? |
| ------------------------------------------------------ | ------------- | ----------------------- |
| `_target_` not a string (`factory`, `resolve_object`)  | `TypeError`   | yes                     |
| `_target_` not a string (`is_object_config`)           | none: `False` | yes                     |
| `_target_` outside `allowed_prefixes`                  | `ImportError` | yes, before the import  |
| module or attribute of the target missing              | `ImportError` | yes                     |
| dependency of the target module missing                | the original  | yes                     |
| argument that is a configuration, with an allow-list   | `ValueError`  | yes, before the import  |
| target is a class / function that does not build `cls` | `TypeError`   | yes                     |
| unknown result type with `strict=True`                 | `TypeError`   | yes                     |
| abstract class, non-callable, bad `_init_`             | `TypeError`   | yes                     |
| exception of the target                                | propagates    | no                      |
| result is not an instance of `cls`                     | `TypeError`   | no (after the call)     |

## Public API

```python
from coola.factory import (
    OBJECT_INIT,  # "_init_"
    OBJECT_PREFIXES,  # "_allowed_prefixes_" (new)
    OBJECT_TARGET,  # "_target_"
    factory,  # (_target_, *args, _init_="__init__", _allowed_prefixes_=None, **kwargs)
    import_object,  # (object_path, *, allowed_prefixes=None)
    instantiate_object,  # (obj, /, *args, _init_="__init__", **kwargs)
    is_object_config,  # (config, cls, *, allowed_prefixes=None, import_target=False)
    resolve_object,  # (obj, cls=object, *, allowed_prefixes=None, strict=False)
)
```

Behaviour changes (for the migration page):

| Change                                                                                        | Before                                                        | After                                                    |
| --------------------------------------------------------------------------------------------- | ------------------------------------------------------------- | -------------------------------------------------------- |
| A target can receive `obj` or `cls`                                                           | `TypeError` from the API                                      | works                                                    |
| `_init_="__new__"` or a name that starts with `_`                                             | accepted                                                      | `TypeError`                                              |
| `_allowed_prefixes_` in the configuration of `resolve_object`                                 | used when the caller passes no list                           | ignored                                                  |
| `factory(..., _allowed_prefixes_=[...])` with an argument that is `{"_target_": ...}`         | passed to the target                                          | `ValueError`                                             |
| a `dict` with `_target_` and a `dict` subclass `cls`                                          | returned unbuilt                                              | built                                                    |
| a `dict` without `_target_`                                                                   | `TypeError` ("missing the `_target_` key")                    | returned if it is an instance of `cls`, else `TypeError` |
| a `dict` subclass instance that has a `_target_` key (for example `Counter({"_target_": 2})`) | returned if it is an instance of `cls`                        | resolved as a configuration (the key is reserved)        |
| missing module or attribute, any `cls`                                                        | `ModuleNotFoundError` / `ImportError: Module ... has no attr` | `ImportError: The target object does not exist: ...`     |
| target that returns `A \| B` with no subclass of `cls`                                        | called, then rejected                                         | refused before the call                                  |
| `is_object_config({"_target_": 5}, int)`                                                      | `TypeError`                                                   | `False`                                                  |
| explicit `_init_="__init__"` with `resolve_object`                                            | target not checked                                            | target checked                                           |

## Dependents

| Package                                           | Change                                                                                                                                                                                                                                                                                                                                                        |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `io.base` (`resolve_loader`, `resolve_saver`)     | Unchanged code. A dict without `_target_` now fails with `Received object is not a BaseLoader instance (received: <class 'dict'>; a configuration needs the `_target_` key)` instead of "missing the `_target_` key". No test of `io` matches the old message; the two tests of `tests/unit/factory/test_resolve.py` that do (lines 465 and 502) are updated. |
| `io.base` (`is_loader_config`, `is_saver_config`) | Unchanged code. Their docstrings point to `is_object_config`, which keeps the same contract for `allowed_prefixes` and `import_target`.                                                                                                                                                                                                                       |
| `utils.imports.import_object`                     | Unchanged. Both docstrings cross-reference each other and state the path syntax.                                                                                                                                                                                                                                                                         |
| `pyproject.toml` (import-linter)                  | No change: `coola.factory` imports `coola.utils.imports` only, as before.                                                                                                                                                                                                                                                                                      |

## Decision record (settled)

Summary (every decision follows the recommendation):

| #   | Decision                                             | Outcome                                 | Steps |
| --- | ---------------------------------------------------- | --------------------------------------- | ----- |
| D1  | A dict with `_target_` is always a configuration     | yes                                     | 6     |
| D2  | `_init_`: no `__new__`, no name that starts with `_` | yes                                     | 4     |
| D3  | The allow-list is never read from the data           | yes                                     | 3     |
| D4  | Refuse a nested configuration under an allow-list    | yes                                     | 3     |
| D5  | Configurations are `dict`, not `Mapping`             | `dict`                                  | 6     |
| D6  | Check the defining module of the target              | no (the list filters paths)             | none  |
| D7  | Rename one of the two `import_object`                | no, cross-reference                     | 2     |
| D8  | Attribute paths (`pkg.module.Class.method`)          | no                                      | none  |
| D9  | Location of `builds`                                 | `coola/factory/target.py`, not exported | 5     |

**D1. A `dict` with the key `_target_` is always a configuration.** **Decided: yes.** It replaces
the rule that depends on `cls`. The key is reserved, as in Hydra: a dict that has it cannot be
passed through `resolve_object` as data, and the guide says so. Consequence: the dict-subclass tests
and the guide warning are deleted, `Mapping`, `dict` and `object` work the same, and the explanation
of the rule is one sentence. Rejected alternative: keep the previous rule; it needs the `issubclass` special case,
fails for generics, and leaves  open.

**D2. Remove `_init_="__new__"` and the underscore names.** **Decided: yes.** `cls(...)` runs
`__new__` and `__init__`; `cls.__new__(cls, ...)` alone gives a partially initialised object, and the
data can choose it. A public class method (`from_config`, `fromkeys`, `now`) is what `_init_` is for.
Breaking for anybody who wrote `_init_: __new__` (the migration page says to use a class method).

**D3. The allow-list is never read from the data.** **Decided: yes.** The key
`_allowed_prefixes_` stays as a keyword of `factory` (so `factory(**config, ...)` keeps working), but
`resolve_object` ignores it in a configuration. A policy that the data can narrow looks like a
feature, and it is a hole in the day somebody lets the data widen it. Rejected alternative: keep reading it
when the caller passes none (today); costs nothing, keeps the smell.

**D4. Refuse a nested configuration when an allow-list is active.** **Decided: yes.** It closes the
exact case of  (an allowed function that resolves its arguments) in 5 lines, and it fails closed:
whoever needs the nested object resolves it first, with the same list, as the guide already tells.
Limits: it scans the top level of the arguments only (a configuration deeper in a list is not found),
and it does not help against a prefix that contains `import_object`. It is a mitigation, not a proof;
the guide says which prefixes must never be allowed. Rejected alternative: documentation only; cheaper, but the
fix of  then exposes .

**D5. `dict` or `Mapping` for configurations?** **Decided: `dict`.** JSON and YAML load into
`dict`; a `DictConfig` is converted by the caller (`to_container`). Accepting any `Mapping` costs one
word but widens what counts as "a configuration" in a security-relevant check.

**D6. Check the module that defines the target?** **Decided: no.** Comparing `target.__module__` to
the prefixes would stop a re-export from another package, but it does not close  (the functions
of `coola.factory` are defined under `coola`), and it rejects legitimate aliases (`os.path.join` is
defined in `posixpath`, `numpy` functions in private modules of other names). The guide states that
the list filters paths.

**D7. Rename one of the two `import_object`?** **Decided: no, cross-reference.** The colon form is
the internal syntax of the lazy registries, the dotted form is the user-facing one; renaming
`coola.utils.imports.import_object` touches `registry`, `random`, `reducer` and their tests for a
naming gain. Both docstrings state the path syntax and link to each other.

**D8. Support attribute paths (`pkg.module.Class.method`)?** **Decided: no.** It needs a
loop over prefixes with the "does not exist" detection of at each step. A class method
is reached with `_init_`, a nested class is rare. (`import_object("collections.OrderedDict.fromkeys")`
raises `ImportError: The target object does not exist`, with the prototype.) Three reasons: (1) the
fallback loop must tell an expected failure (`collections.OrderedDict` is not a module) from a real
one (a module that exists but whose own dependency is missing), which is the `err.name` rule of
 repeated at every step; (2) `_init_` already reaches class methods, and D2 keeps it
public; (3) walking attributes widens what a prefix reaches: `"mypkg"` would allow
`mypkg.module.Class.helper.os.system`, whereas before only the last segment goes through `getattr`.
Cost accepted: a nested class (`pkg.Outer.Inner`) needs a module-level alias.

**D9. Where does `builds` live?** **Decided: `coola/factory/target.py`, public name, not
exported.** It is a helper of two modules of one package; `coola.utils` is for helpers shared by
several packages (`CLAUDE.md`), and `builds` has no other user.

## Not covered

- **Nested resolution** (`recursive=True`, Hydra's `_recursive_`): each level would need the
  allow-list, the `strict` check and the cycle/depth rules of `recursive`. The nested object is
  resolved by the caller.
- **Positional arguments in a configuration** (`_args_`): a configuration is keyword-only. `factory`
  accepts `*args` for code that has them.
- **Partial application** (`_partial_`) and interpolation (`${...}`): out of scope; use
  `functools.partial` on the result.
- **A registry of aliases** (`"loader": "coola.io.JsonLoader"`): that is `coola.registry`.
- **Sandboxing** the target: not possible in-process. The allow-list is a filter on a path.
- **`strict` by default**: the check after the call stays the contract for a target without a hint;
  `io` users who want the strict mode pass it.
- **Entry points for targets**: none exist before and none are proposed.
