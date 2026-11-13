SHELL = /bin/bash

include .make/help.mk
include .make/actions.mk
include .make/git.mk
include .make/github.mk
include .make/makefile.mk
include .make/markdown.mk
include .make/self.mk
include .make/shell.mk
include .make/toml.mk
include .make/tools.mk
include .make/uv.mk
include .make/yaml.mk

.DEFAULT_GOAL := help

.PHONY: install-tools
install-tools: install-prettier install-yamllint install-mbake install-checkmake install-shellcheck install-shfmt install-bats install-taplo install-markdownlint install-actionlint install-gh ## Install all formatting/linting tools

.PHONY: format
format: format-yaml format-makefile format-shell format-toml format-markdown ## Format all files

.PHONY: lint
lint: lint-yaml lint-makefile lint-shell lint-markdown lint-actions ## Lint all files

.PHONY: clean
clean: ## Remove the generated coverage reports and caches
	rm -rf coverage.xml .coverage htmlcov .pytest_cache .ruff_cache .hypothesis
