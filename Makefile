SHELL=/bin/bash

include .make/help.mk
include .make/makefile.mk
include .make/markdown.mk
include .make/self.mk
include .make/shell.mk
include .make/uv.mk
include .make/yaml.mk

.DEFAULT_GOAL := help

.PHONY: format
format: format-yaml format-makefile format-shell format-markdown ## Format all files

.PHONY: lint
lint: lint-yaml lint-makefile lint-shell lint-markdown ## Lint all files
