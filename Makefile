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