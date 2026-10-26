# `coola.factory`: target state and the changes to reach it

This document describes the design that `coola.factory` should have, then the changes to make to
the current implementation (`main`, `b82b66ea`) to get there. The design follows the rules of the
user guide (`docs/docs/uguide/factory.md`, **Properties**) and the architecture guide
(`docs/docs/dev/architecture.md`: strict contracts, lazy imports, no hidden global state), and it
stays aligned with the packages that use it (`io`) and with the style of the sibling packages
(`recursive`, `iterator`: one place that owns the invariants, keyword-only options, errors that name
the cause). Its aim is a small package that is easy to maintain: **one way to import a target, one
way to call it, one way to ask what it builds**. Breaking changes are accepted, and the future state
is more important than the past one.

**Status: the nine decisions of section 9 are settled** (D1 to D9, all as recommended). The design
below is the one to implement; section 9 records why, and what each decision costs.

Evidence comes from probes run on `main` (Python 3.14, scripts kept outside the repository) and from
a throw-away prototype of the target modules. Items marked **[verified]** were reproduced;
**[reading]** comes from the code. Nothing in `src`, `tests` or `docs` was changed.

## Constraints

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

## 1. Properties of the sub-package

These are the rules of the user guide (the five rows of its **Properties** table, completed). Every
design choice below exists to make one of them true **by construction** (in one function) instead of
by the discipline of each entry point.

| Property                                    | Statement                                                                                                                                                                                                                                                                                                                                                                                  | Enforced by                       |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------- |
| Explicit target                             | The object to import or call is named only by the `_target_` string (`package.module.Name`). Every other key is an argument, passed unchanged, by keyword. The reserved keys are `_target_`, `_init_` and `_allowed_prefixes_`.                                                                                                                                                            | `factory`, `resolve_object`       |
| No collision                                | No name of a parameter of the API can shadow an argument of the target: the target receives every key that is not reserved, whatever its name (`obj`, `cls`, `config`, ...).                                                                                                                                                                                                               | positional-only parameters        |
| Top level only                              | `_target_` is resolved once per call, at the top level. A nested dictionary is data and is passed unchanged. There is no recursion and no hidden import.                                                                                                                                                                                                                                   | `resolve_object`                  |
| Import once, call once                      | The target is imported once per call and called once. Nothing else is imported or called, in particular nothing that the checks need.                                                                                                                                                                                                                                                      | `import_object`, `resolve_object` |
| Fails loudly                                | A target that does not exist raises `ImportError` with one message, whichever entry point is used. A wrong type raises `TypeError`. A default is never substituted for a missing object, and an exception of the target propagates.                                                                                                                                                        | `import_object`                   |
| Check before call                           | When `cls` is given, a target that is known not to build a `cls` (a class that is not a subclass, a function whose return hint is not a subclass) is refused **before** it is called. The result is checked again after the call.                                                                                                                                                          | `builds`, `resolve_object`        |
| Strict mode                                 | With `strict=True` a target whose result type cannot be known beforehand (no usable return hint, a built-in) is refused before it is called.                                                                                                                                                                                                                                               | `builds`, `resolve_object`        |
| Opt-in restriction                          | `allowed_prefixes` is checked against the string **before** the module is imported. An empty list allows nothing; a bare string is one prefix. Without it, any importable object can be called.                                                                                                                                                                                            | `import_object`                   |
| Policy comes from code                      | The allow-list is an argument of the caller. `resolve_object`, the API that receives configurations, ignores the key `_allowed_prefixes_`, so the data cannot set, widen or narrow the list. `factory` has the list as a keyword for code that unpacks its own arguments (`factory(**args, _allowed_prefixes_=...)`); whoever unpacks untrusted data into `factory` owns that choice (D3). | `resolve_object`                  |
| No nested configuration under an allow-list | With `allowed_prefixes`, an argument that is itself a configuration (a dict with `_target_`) is refused: the callee could resolve it without the allow-list. Resolve it first, with the same list.                                                                                                                                                                                         | `factory`, `resolve_object`       |
| Public constructors only                    | `_init_` is `"__init__"` (default) or the name of a public class method or static method that returns an instance of the class. A name that starts with `_` is refused. The data cannot reach `__new__`, `__subclasses__`, ...                                                                                                                                                             | `instantiate_object`              |
| Predicates do not import                    | `is_object_config` imports nothing by default: a target whose module is not imported is not a configuration. `allowed_prefixes` or `import_target=True` opt in. A malformed configuration is `False`, a violation of the list raises.                                                                                                                                                      | `is_object_config`                |
| Pure, stateless                             | The configuration is never modified. No global state: the functions can be called from any thread, and after a `fork`, with no lock.                                                                                                                                                                                                                                                       | the functions are free of state   |
| Lazy                                        | Nothing is imported at import time except the standard library; a target is imported when it is named.                                                                                                                                                                                                                                                                                     | `import_object`                   |
| Keyword-only options                        | `allowed_prefixes`, `import_target`, `strict`, `_init_` and `_allowed_prefixes_` are keyword-only.                                                                                                                                                                                                                                                                                         | the signatures                    |

Not guaranteed, and documented as such:

- **An allow-list is a filter on a path, not a sandbox.** It limits _what_ is called, not the
  arguments, and not what an allowed function does. A prefix that contains a function that imports
  or resolves objects from its arguments (any `resolve_*`, `import_object`, `importlib`, `pickle`,
  `eval`, `os.system`, ...) makes the list worthless. In particular `"coola"` or `"coola.factory"`
  must never be an allowed prefix **[verified]** (F5). The guide says so.
- **Re-exports.** A path is checked, not the module that defines the object: `mypkg.helper` allowed
  by `"mypkg"` may be a function defined elsewhere (decision D6).
- **The result type of a target without a return hint** is known only after the call (unless
  `strict=True`).
- **Constructors with side effects.** Calling the target is calling user code: its side effects
  happen, and a wrong result type is detected after them.

## 2. What differs today

| #   | Finding                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Status         |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| F1  | **A target cannot receive an argument named `obj` or `cls`.** `instantiate_object(obj, *args, _init_=..., **kwargs)` and the private `_instantiate_class_object(cls, ...)` take their first parameter by name, so `factory("pkg.C", cls=1)` raises `_instantiate_class_object() got multiple values for argument 'cls'` and `factory("pkg.f", obj=1)` raises the same for `obj`. A parameter named `cls` or `obj` is common (`cls` of a loss function, `obj` of a visitor). The failure is a `TypeError` that blames the API, not the data.                                                                                                                                                                                                                                                                                                                                                                                                                           | **[verified]** |
| F2  | **`resolve_object` imports the target twice and reports a missing target in three ways.** `_check_target` calls `import_object` and then `factory` imports again. With `cls=object` a missing module raises `ImportError: The target object does not exist: nomod.X` (the translation is in `factory`); with any other `cls` the check runs first and raises the raw `ModuleNotFoundError: No module named 'nomod'`, or `ImportError: Module 'collections' has no attribute 'Nope'` for a missing attribute. `is_object_config` raises the raw errors too. The translation belongs to the import, not to one caller.                                                                                                                                                                                                                                                                                                                                                  | **[verified]** |
| F3  | **"What does this target build?" is written twice, and the two copies disagree.** `is_object_config` (`config.py`) handles a function return hint, a union (`A \| B`) and a generic (`list[int]`); `_check_target` (`resolve.py`) handles only a class or a function returning a class. For `def make() -> Cat \| None` and `cls=Animal`: `is_object_config` says `False`, `resolve_object` **calls** `make` and rejects the result afterwards. For `def make() -> Dog \| None` with `strict=True`: `resolve_object` refuses it with "not a class and has no return type hint", although the hint exists. Both call `get_type_hints` with the same `try/except`.                                                                                                                                                                                                                                                                                                      | **[verified]** |
| F4  | **The "is this dict a configuration?" rule is a special case with holes.** Whether a `dict` is a configuration depends on `cls`: for a `dict` subclass `cls` an instance is returned as is, otherwise any dict is a configuration. So `resolve_object({"_target_": "collections.OrderedDict", "a": 1}, dict)` returns the configuration **unbuilt** (`{'_target_': ..., 'a': 1}`); `resolve_object({"a": 1}, Mapping)` raises "missing the `_target_` key" for a valid `Mapping`; `resolve_object({"a": 1})` raises instead of returning the dict; `cls=list[int]` raises `issubclass() arg 1 must be a class` from the special case instead of the `isinstance` error. The guide documents the rule in a warning, and the warning is not rendered (F9).                                                                                                                                                                                                              | **[verified]** |
| F5  | **The allow-list is defeated by the entry points under the allowed prefix.** `factory("coola.io.resolve_loader", loader={"_target_": "os.getcwd"}, _allowed_prefixes_=["coola.io"])` **calls `os.getcwd`** (it then fails because the result is not a loader): the nested configuration is resolved by `resolve_loader`, which has no allow-list. With `os.system` instead of `os.getcwd` that is code execution under an allow-list that looks restrictive. Every `resolve_*`, and `factory` and `import_object` themselves (`factory("coola.factory.import_object", object_path="os:getcwd", ...)` returns `os.getcwd`), have the same property. Not exploitable through `io` alone, because the prefix must have been chosen too wide; but nothing tells the user. Fixing F1 alone makes `factory("coola.factory.resolve_object", obj={"_target_": "os.getcwd"}, _allowed_prefixes_=["coola"])` work **[verified]**: today it is blocked by the F1 collision only. | **[verified]** |
| F6  | **The configuration can carry its own policy and choose any attribute.** `_allowed_prefixes_` is read from the data when the caller passes none (`_check_target` and `factory` both read it): a policy that comes from the data it protects. `_init_` takes any attribute name from the data: `__new__` (a documented footgun: `__init__` is skipped and the object is partially built), `__class__`, `__subclasses__`, `mro`, ... The class is called with the arguments, and the result type is checked after the call. `_init_=5` raises `attribute name must be string, not 'int'`. The `"__new__"` branch (`init_fn(cls, ...)`) and its `Warning` section exist for a use that `cls(...)` already covers.                                                                                                                                                                                                                                                        | **[verified]** |
| F7  | **A predicate that raises.** `is_object_config({"_target_": 5}, int)` raises `TypeError: object_path is not a string`; a function named `is_*` that answers a boolean about arbitrary data should answer `False` for malformed data. (A target that violates `allowed_prefixes` should still raise: that is a policy decision, not malformed data.)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | **[verified]** |
| F8  | **Two public functions named `import_object`.** `coola.factory.import_object("collections.Counter")` (dotted path, strict errors, allow-list) and `coola.utils.imports.import_object("collections:Counter")` (colon path, used by the lazy registries). Same name, different syntax and errors; a wrong import of one gives a `ValueError` or an `ImportError` depending on which.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | **[reading]**  |
| F9  | **The user guide is partly broken or stale.** The `!!! warning` of "Resolving an object or a configuration" has an unindented body, so the admonition renders empty and the text is a loose paragraph. The `OBJECT_INIT` example imports `OBJECT_INIT` and passes `_init_="__init__"`, which is the default and shows nothing (no class method example, no `__new__` example). The **Properties** table is after the security section and lacks the rules above. `docs/docs/dev/architecture.md` does not describe the package (it appears only in a list of feature packages). `uguide/factory.md` mentions `resolve_loader`/`is_loader_config` options without a link.                                                                                                                                                                                                                                                                                              | **[verified]** |
| F10 | **Tests.** `tests/unit/factory` has 1,622 lines and covers each branch, but there is no integration test (no real `io` flow with an allow-list, no test that `is_object_config` really imports nothing, no thread test) and no property test. No unit test passes a keyword named `obj` or `cls` to a target (F1), and none combines an allow-list with a target that resolves its own arguments (F5).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | **[verified]** |

## 3. Architecture

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

### 3.1 `constants.py`

```python
OBJECT_INIT = "_init_"
OBJECT_PREFIXES = "_allowed_prefixes_"  # new: the literal was repeated in three places
OBJECT_TARGET = "_target_"
```

### 3.2 `import_object` (`instantiation.py`)

One change in behaviour: the translation that `factory` does today moves into the function, so that
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

### 3.3 `instantiate_object` (`instantiation.py`)

One public function, no private `_instantiate_class_object`, the first parameter positional-only
(F1), the `_init_` rule (F6):

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
`"__new__"` branch and its `Warning` are deleted (decision D2).

### 3.4 `builds` (new module `target.py`)

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

### 3.5 `factory`, `resolve_object`, `is_object_config` (`resolve.py`, `config.py`)

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
            )  # the two messages of today, naming the target; "it was not called"
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
  is dropped silently (documented): the policy is the caller's (F6, D3).
- `_refuse_nested_configurations(args, kwargs)` raises `ValueError` if a value of the arguments is a
  `dict` with `_target_`. Only the top level is scanned (D4).
- `_not_an_instance_message` adds, for a `dict`, "(a configuration needs the `_target_` key)", so
  the useful part of today's "missing the `_target_` key" error is kept in the single `TypeError`.
- An explicit `_init_="__init__"` is checked like the default (today the presence of the key skips the
  check).
- The `logger.info("Initializing a ...")` line is kept (the siblings log at the same level).

### 3.6 Error summary (the contract that the tests assert)

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

## 4. Public API after the change

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

## 5. Dependents

| Package                                           | Change                                                                                                                                                                                                                                                                                                                                                        |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `io.base` (`resolve_loader`, `resolve_saver`)     | Unchanged code. A dict without `_target_` now fails with `Received object is not a BaseLoader instance (received: <class 'dict'>; a configuration needs the `_target_` key)` instead of "missing the `_target_` key". No test of `io` matches the old message; the two tests of `tests/unit/factory/test_resolve.py` that do (lines 465 and 502) are updated. |
| `io.base` (`is_loader_config`, `is_saver_config`) | Unchanged code. Their docstrings point to `is_object_config`, which keeps the same contract for `allowed_prefixes` and `import_target`.                                                                                                                                                                                                                       |
| `utils.imports.import_object`                     | Unchanged (D7). Both docstrings cross-reference each other and state the path syntax.                                                                                                                                                                                                                                                                         |
| Docs                                              | `uguide/factory.md` (rewrite, F9), `dev/architecture.md` (one paragraph on the package), `migration/v1.1_to_v2.0.md` (section 3), `refs/factory.md` unchanged (`::: coola.factory`).                                                                                                                                                                          |
| `pyproject.toml` (import-linter)                  | No change: `coola.factory` imports `coola.utils.imports` only, as today.                                                                                                                                                                                                                                                                                      |

## 6. Evidence: probes and a prototype

The probes of F1 to F7 were run on `main`; the same probes were run on a prototype of the modules of
section 3 (a single file kept outside the repository, not the repository's tests). **[verified]**

| Probe                                                                                                   | `main`                                                    | Prototype                                                 |
| ------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- | --------------------------------------------------------- |
| `factory("pkg.C", cls=1)`, `factory("pkg.f", obj=1)`, `instantiate_object(C, cls=1)`                    | `TypeError ... got multiple values for argument`          | works, the target receives the keyword                    |
| `resolve_object({"_target_": "nomod.X"}, int)`                                                          | `ModuleNotFoundError: No module named 'nomod'`            | `ImportError: The target object does not exist: nomod.X`  |
| `resolve_object({"_target_": "collections.Nope"}, int)`                                                 | `ImportError: Module 'collections' has no attribute`      | same "does not exist" message                             |
| `import_module` calls for `resolve_object({"_target_": "collections.Counter"}, Counter)`                | 2                                                         | 1                                                         |
| `resolve_object({"a": 1}, Mapping)` / `resolve_object({"a": 1})`                                        | `TypeError` (missing `_target_`)                          | `{'a': 1}`                                                |
| `resolve_object({"_target_": "collections.OrderedDict", "a": 1}, dict)`                                 | `{'_target_': ..., 'a': 1}` (unbuilt)                     | `OrderedDict({'a': 1})`                                   |
| `resolve_object(Counter({"_target_": 2}), Counter)`                                                     | `Counter({'_target_': 2})`                                | `TypeError` (reserved key; documented)                    |
| `make() -> Cat \| None`, `cls=Animal`                                                                   | `make` is **called**, then `TypeError`                    | refused, not called                                       |
| `make() -> Dog \| None`, `cls=Animal`, `strict=True`                                                    | refused ("no return type hint")                           | built                                                     |
| `is_object_config({"_target_": 5}, int)`                                                                | `TypeError`                                               | `False`                                                   |
| `_init_=5`, `_init_="__class__"`                                                                        | `attribute name must be string` / a confusing `TypeError` | one `TypeError` that names the rule                       |
| `{"_target_": "collections.Counter", "_allowed_prefixes_": ["os"]}`                                     | `ImportError` (the data narrows the list)                 | `Counter()` (ignored: the caller decides)                 |
| `factory("coola.factory.resolve_object", obj={"_target_": "os.getcwd"}, _allowed_prefixes_=["coola"])`  | blocked by F1 only; works once F1 is fixed                | `ValueError` (nested configuration)                       |
| `factory("coola.io.resolve_loader", loader={"_target_": "os.getcwd"}, _allowed_prefixes_=["coola.io"])` | **calls `os.getcwd`**                                     | `ValueError`                                              |
| `factory("coola.factory.import_object", object_path="os:getcwd", _allowed_prefixes_=["coola"])`         | returns `os.getcwd`                                       | unchanged (import only, nothing called): documented limit |

Speed is not a design input (a call takes 2.7 µs for `factory` and 4.4 µs for `resolve_object` with
a class check, dominated by `importlib`), so there is no benchmark gate. The new `resolve_object`
imports once and caches nothing; it is not slower.

## 7. Changes to reach the target state

One branch and one pull request per step (`CLAUDE.md`). The order keeps every step releasable and
**never exposes F5**: the step that fixes F1 ships with the step that closes the allow-list.

| Step | Change                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | **Fix the guide (F9).** Indent the admonition, remove the unused `OBJECT_INIT` import, add a class method example, move the **Properties** table to the top and complete it with section 1, add the "allow-list is not a sandbox" rule with the `coola.io` example (F5), add the architecture paragraph. No code change.                                                                                                                                                      |
| 2    | **One import, one error (F2, F8).** The "does not exist" translation moves into `import_object`; `factory` loses its `try/except`; cross-reference the two `import_object` docstrings. Tests: the three error messages through the three entry points.                                                                                                                                                                                                                        |
| 3    | **Close the allow-list and fix the collisions (F1, F5, F6).** `_refuse_nested_configurations` in `factory` and `resolve_object` (D4), positional-only `obj`, `_allowed_prefixes_` ignored in a configuration and keyword in `factory` (D3), `OBJECT_PREFIXES`. **One PR**, because F1 alone opens F5. The refusal is added to the current `resolve_object` (before step 6 rewrites it). Regression tests for each (check that they fail on `main` with `git stash push src`). |
| 4    | **`_init_` rule (F6, D2).** Delete the `__new__` branch and the private `_instantiate_class_object`; refuse names that start with `_`; tests for every branch.                                                                                                                                                                                                                                                                                                                |
| 5    | **One answer to "what does it build?" (F3, F7).** New `target.py` with `builds`; `is_object_config` and `resolve_object` use it; `is_object_config` returns `False` for a non-string target. Regression tests for the union cases.                                                                                                                                                                                                                                            |
| 6    | **The dict rule and one import (F4, F2).** `resolve_object` treats a `dict` with `_target_` as a configuration, imports once and calls `instantiate_object` itself (D1). Update the two message tests of `tests/unit/factory/test_resolve.py`. Delete the dict-subclass tests and the guide warning.                                                                                                                                                                          |
| 7    | **Tests (F10).** Integration and property tests of section 8.                                                                                                                                                                                                                                                                                                                                                                                                                 |
| 8    | **Documentation.** Migration page (section 4 table), `dev/architecture.md` paragraph, status table of `review/suggestions.md` with the PR numbers, docstrings (`Security:` sections state the limits of section 1).                                                                                                                                                                                                                                                           |

## 8. Tests and gates

Unit tests mirror the source tree (`tests/unit/factory/`); the conventions of `CLAUDE.md` apply (one
behaviour per test, literal expected values, a regression test for every bug, no local imports).

- **`import_object`**: typical path, module path, nested module (`os.path`), empty path and empty
  module name, non-string, allow-list (equal, child, sibling that shares a string prefix, empty list,
  bare string), missing module, missing attribute, **a missing dependency of an existing module keeps
  its own error** (a module on `tmp_path` that imports a missing package), the message and `name` of
  each `ImportError`.
- **`instantiate_object`**: class, function, `functools.partial`, callable instance, non-callable,
  abstract class; **a keyword named `obj` and a keyword named `cls` reach the target** (regression
  for F1, for a class and for a function); a public class method and a static method; `_init_` that
  starts with `_`, that is not a string, that is missing, that is not callable; an alternative
  constructor that returns another type; `_init_` ignored for a function.
- **`builds`** (`test_target.py`, new): class, subclass, unrelated class, function with a class hint,
  union (`A | B`, `Union[A, B]`, `A | None`), generic (`list[int]`), tuple of classes for `cls`,
  `None` for a function without hint, a built-in, an unresolvable forward reference, `Any`, a union
  with a member that is not a class. The whole table is one parametrised test.
- **`resolve_object`**: instance returned as is (same object), configuration built, tuple of classes,
  `cls=object`, a dict without `_target_` (returned for `dict`/`Mapping`/`object`, `TypeError` and the
  hint for another `cls`), a dict with `_target_` and `cls=dict` is built (F4), the key `_target_`
  in a `Counter` is resolved, a generic `cls` raises the `isinstance` error; allow-list: the caller's
  list wins over `_allowed_prefixes_`, the key is ignored without a list, the input is not modified;
  **nested configuration refused with an allow-list and passed without** (F5); the target is **not
  called** when refused (a class with a counter in `__init__`, as `Tracked` in the current tests);
  union return hints (F3); `strict`; explicit `_init_="__init__"` is checked.
- **`is_object_config`**: the current cases, plus a non-string target, and the same answers as
  `resolve_object` for the same configurations (a parametrised table of `(config, cls)` where
  `is_object_config` is `True` iff `resolve_object` builds it, for the cases where the answer is
  known).
- **Properties** (`tests/properties/test_factory.py`, new, seeded Hypothesis): for a builtin type
  `T` in `{list, dict, set, frozenset, tuple, int, str}` and a value `v` of `T`,
  `resolve_object({"_target_": f"builtins.{T.__name__}", **kwargs}, T)` equals `T(**kwargs)` where
  the type takes keywords (`dict`), `factory` equals the direct call, and `resolve_object(v, T)` is
  `v` (identity) for any value that is not a dict with `_target_`; the configuration is not modified
  (deep copy compared).
- **Integration** (`tests/integration/factory/`, new, no patching):
  - **the target is imported once** (F2): a module written to `tmp_path` whose top-level code appends
    to a list, resolved through `resolve_object` with a class `cls`; the list has one entry (unit
    tests may count `importlib.import_module` calls with a patch, but the integration test is the
    one that proves it);
  - **`is_object_config` imports nothing** unless opted in: the same module is not in `sys.modules`
    after the call (this is the assertion that the 27.2 fix needs; the unit test patches);
  - **`allowed_prefixes` through `io`**: `resolve_loader` with a nested configuration under an
    allow-list raises before the nested target is imported (F5);
  - **threads**: `run_threads` on `resolve_object` with a lock around the shared list of results
    (docstring "Test that ..."), and one `fork` test only if a global state ever appears.
- **Doctests** run as part of the docstrings; the guide examples are doctests. A security example
  shows the refusal of a nested configuration.
- Coverage: 100 % of `src/coola/factory` (`--cov=coola`), no `# pragma: no cover`.

## 9. Decisions (settled)

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
the rule that depends on `cls` (F4). The key is reserved, as in Hydra: a dict that has it cannot be
passed through `resolve_object` as data, and the guide says so. Consequence: the dict-subclass tests
and the guide warning are deleted, `Mapping`, `dict` and `object` work the same, and the explanation
of the rule is one sentence. Rejected alternative: keep today's rule; it needs the `issubclass` special case,
fails for generics, and leaves F4 open.

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
exact case of F5 (an allowed function that resolves its arguments) in 5 lines, and it fails closed:
whoever needs the nested object resolves it first, with the same list, as the guide already tells.
Limits: it scans the top level of the arguments only (a configuration deeper in a list is not found),
and it does not help against a prefix that contains `import_object`. It is a mitigation, not a proof;
the guide says which prefixes must never be allowed. Rejected alternative: documentation only; cheaper, but the
fix of F1 then exposes F5.

**D5. `dict` or `Mapping` for configurations?** **Decided: `dict`.** JSON and YAML load into
`dict`; a `DictConfig` is converted by the caller (`to_container`). Accepting any `Mapping` costs one
word but widens what counts as "a configuration" in a security-relevant check.

**D6. Check the module that defines the target?** **Decided: no.** Comparing `target.__module__` to
the prefixes would stop a re-export from another package, but it does not close F5 (the functions
of `coola.factory` are defined under `coola`), and it rejects legitimate aliases (`os.path.join` is
defined in `posixpath`, `numpy` functions in private modules of other names). The guide states that
the list filters paths.

**D7. Rename one of the two `import_object`?** **Decided: no, cross-reference.** The colon form is
the internal syntax of the lazy registries, the dotted form is the user-facing one; renaming
`coola.utils.imports.import_object` touches `registry`, `random`, `reducer` and their tests for a
naming gain. Both docstrings state the path syntax and link to each other.

**D8. Support attribute paths (`pkg.module.Class.method`)?** **Decided: no.** It needs a
loop over prefixes with the "does not exist" detection of section 3.2 at each step. A class method
is reached with `_init_`, a nested class is rare. (`import_object("collections.OrderedDict.fromkeys")`
raises `ImportError: The target object does not exist`, with the prototype.) Three reasons: (1) the
fallback loop must tell an expected failure (`collections.OrderedDict` is not a module) from a real
one (a module that exists but whose own dependency is missing), which is the `err.name` rule of
section 3.2 repeated at every step; (2) `_init_` already reaches class methods, and D2 keeps it
public; (3) walking attributes widens what a prefix reaches: `"mypkg"` would allow
`mypkg.module.Class.helper.os.system`, whereas today only the last segment goes through `getattr`.
Cost accepted: a nested class (`pkg.Outer.Inner`) needs a module-level alias.

**D9. Where does `builds` live?** **Decided: `coola/factory/target.py`, public name, not
exported.** It is a helper of two modules of one package; `coola.utils` is for helpers shared by
several packages (`CLAUDE.md`), and `builds` has no other user.

## 10. Not covered

- **Nested resolution** (`recursive=True`, Hydra's `_recursive_`): each level would need the
  allow-list, the `strict` check and the cycle/depth rules of `recursive`. The nested object is
  resolved by the caller (27.1 of `suggestions.md`).
- **Positional arguments in a configuration** (`_args_`): a configuration is keyword-only. `factory`
  accepts `*args` for code that has them.
- **Partial application** (`_partial_`) and interpolation (`${...}`): out of scope; use
  `functools.partial` on the result.
- **A registry of aliases** (`"loader": "coola.io.JsonLoader"`): that is `coola.registry`.
- **Sandboxing** the target: not possible in-process. The allow-list is a filter on a path.
- **`strict` by default**: the check after the call stays the contract for a target without a hint;
  `io` users who want the strict mode pass it.
- **Entry points for targets**: none exist today and none are proposed.

## Appendix: effect on the size of the package

| Module             | Lines today (code) | Expected after (code) | Why                                                                                               |
| ------------------ | ------------------ | --------------------- | ------------------------------------------------------------------------------------------------- |
| `constants.py`     | 23 (4)             | 25 (5)                | `OBJECT_PREFIXES`                                                                                 |
| `instantiation.py` | 215 (62)           | about 190 (50)        | one function for the call, no `__new__` branch and `Warning`, the "does not exist" error moves in |
| `target.py`        | 0                  | about 60 (14)         | `builds` (replaces two copies of 10 lines)                                                        |
| `resolve.py`       | 279 (98)           | about 220 (48)        | no `_check_target`, no dict-subclass rule, `factory` without `try/except` and `pop`               |
| `config.py`        | 114 (36)           | about 90 (12)         | the type test is `builds`                                                                         |
| `__init__.py`      | 18 (14)            | 20 (15)               | `OBJECT_PREFIXES`                                                                                 |
| **Total**          | **649 (214)**      | **about 605 (144)**   | about 70 fewer lines of code, with more checks (nested configuration, `_init_` rule)              |

The line counts are estimates from the prototype and from the code to delete; the gate is the
behaviour (sections 1 and 8), not the size.
