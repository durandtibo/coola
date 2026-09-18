SHELL=/bin/bash

include .make/makefile.mk
include .make/markdown.mk
include .make/self.mk
include .make/shell.mk
include .make/uv.mk
include .make/yaml.mk

.PHONY: format
format: format-yaml format-makefile format-shell format-markdown
