# shared-makefiles

Shared/reusable Makefiles for formatting and linting common file types across projects.

Each file is self-contained, include-guarded, and configurable via variables — pull in only what
you need.

## Usage

Include the files you need in your project's `Makefile`:

```makefile
include yaml.mk
include makefile.mk
include shell.mk
include markdown.mk

.PHONY: install-tools
install-tools: install-prettier install-yamllint install-mbake install-checkmake install-shellcheck install-shfmt install-markdownlint

.PHONY: format
format: format-yaml format-makefile format-shell format-markdown

.PHONY: lint
lint: lint-yaml lint-makefile lint-shell lint-markdown
```

Required tools (`prettier`, `yamllint`, `mbake`, `checkmake`, `shellcheck`, `shfmt`, `markdownlint`)
are installed on demand — each
`format-*`/`lint-*` target depends on an `install-*` target that installs the tool if it isn't
already on `PATH`. Run `make install-tools` to install all of them upfront.

## Available files

| File          | Targets                                     | Tools                      | Description                                                           |
| ------------- | ------------------------------------------- | -------------------------- | --------------------------------------------------------------------- |
| `yaml.mk`     | `format-yaml`, `lint-yaml`                  | `prettier`, `yamllint`     | Format and lint YAML files                                            |
| `makefile.mk` | `format-makefile`, `lint-makefile`          | `mbake`, `checkmake`       | Format and lint Makefiles                                             |
| `shell.mk`    | `format-shell`, `lint-shell`                | `shfmt`, `shellcheck`      | Format and lint shell scripts                                         |
| `markdown.mk` | `format-markdown`, `lint-markdown`          | `prettier`, `markdownlint` | Format and lint Markdown files                                        |
| `uv.mk`       | `install-invoke`, `update-uv`, `setup-venv` | `uv`                       | Manage Python virtual environments with `uv`                          |
| `prettier.mk` | `install-prettier`                          | `prettier`                 | Shared `install-prettier` target, included by `yaml.mk`/`markdown.mk` |

### `yaml.mk`

Optional variables (set before `include`):

| Variable           | Default | Description                                                   |
| ------------------ | ------- | ------------------------------------------------------------- |
| `YAML_FORMAT_PATH` | `.`     | Root path globbed for `**/*.{yml,yaml}`, passed to `prettier` |
| `YAML_LINT_PATH`   | `.`     | Path passed to `yamllint`                                     |

```makefile
include yaml.mk

YAML_LINT_PATH = .github/workflows
```

### `makefile.mk`

Optional variables (set before `include`):

| Variable                | Default    | Description                    |
| ----------------------- | ---------- | ------------------------------ |
| `MAKEFILE_FORMAT_FILES` | `Makefile` | Files passed to `mbake format` |
| `MAKEFILE_LINT_FILES`   | `Makefile` | Files passed to `checkmake`    |

```makefile
include makefile.mk

MAKEFILE_LINT_FILES = Makefile makefile.mk yaml.mk
```

### `shell.mk`

Optional variables (set before `include`):

| Variable            | Default | Description                                           |
| ------------------- | ------- | ----------------------------------------------------- |
| `SHELL_FORMAT_PATH` | `.`     | Path passed to `shfmt` (walked recursively)           |
| `SHELL_LINT_PATH`   | `.`     | Path searched for `*.sh` files passed to `shellcheck` |

```makefile
include shell.mk

SHELL_LINT_PATH = scripts
```

### `markdown.mk`

Optional variables (set before `include`):

| Variable               | Default   | Description                                           |
| ---------------------- | --------- | ----------------------------------------------------- |
| `MARKDOWN_FORMAT_PATH` | `.`       | Root path globbed for `**/*.md`, passed to `prettier` |
| `MARKDOWN_LINT_GLOB`   | `**/*.md` | Glob passed to `markdownlint`                         |

```makefile
include markdown.mk

MARKDOWN_LINT_GLOB = docs/**/*.md
```

### `uv.mk`

Optional variables (set before `include`):

| Variable         | Default | Description                                 |
| ---------------- | ------- | ------------------------------------------- |
| `PYTHON_VERSION` | `3.14`  | Python version passed to `uv venv --python` |

```makefile
include uv.mk

PYTHON_VERSION = 3.12
```

`install-invoke` installs `uv` on demand and then `invoke>=3.0` via `uv pip install` into the
active virtual environment (create one first, e.g. with `uv venv`).
`update-uv` runs `uv self update`. `setup-venv` updates `uv`, creates a fresh `.venv`
(`uv venv --python $(PYTHON_VERSION) --clear`), installs `invoke` into it, and runs
`.venv/bin/inv create-venv` and `.venv/bin/inv install --docs-deps` — it assumes the
project's `tasks.py` (or equivalent) defines `create-venv` and `install` invoke tasks.

## Design

- **Include guards** — each file defines an `_MK_INCLUDED` variable so it's safe to `include`
  more than once (e.g. transitively from multiple project Makefiles).
- **On-demand install** — every lint/format target depends on an `install-<tool>` target that
  checks `command -v` before installing, so CI and local runs don't need the tool preinstalled.
- **Configurable paths** — variables default to sensible project-wide values but can be
  overridden per project or per target invocation.

## Testing

[`.github/workflows/ci-test.yml`](.github/workflows/ci-test.yml) exercises every file against the
Unix OS matrix (Ubuntu and macOS runners) resolved dynamically via
[`durandtibo/workflow-config-action`](https://github.com/durandtibo/workflow-config-action) on
every push/PR to `main`, running both the lint and format targets (including on-demand tool
installation) to make sure the rules stay portable across platforms.

## License

This repo is licensed under BSD 3-Clause "New" or "Revised" license available in [LICENSE](LICENSE)
file.
