include yaml.mk
include makefile.mk
include shell.mk
include markdown.mk
include actions.mk
include help.mk

.DEFAULT_GOAL := help

.PHONY: install-tools
install-tools: install-prettier install-yamllint install-mbake install-checkmake install-shellcheck install-shfmt install-markdownlint install-actionlint ## Install all formatting/linting tools

.PHONY: format
format: format-yaml format-makefile format-shell format-markdown ## Format all files

.PHONY: lint
lint: lint-yaml lint-makefile lint-shell lint-markdown lint-actions ## Lint all files