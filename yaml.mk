ifndef YAML_MK_INCLUDED
YAML_MK_INCLUDED := 1

# Targets for formatting and linting YAML files.
#
# Optional variables (set before including this file):
#   YAML_FORMAT_PATH ?= .   # path passed to prettier
#   YAML_LINT_PATH   ?= .   # path passed to yamllint

YAML_FORMAT_PATH ?= .
YAML_LINT_PATH ?= .

include $(dir $(lastword $(MAKEFILE_LIST)))prettier.mk

.PHONY: format-yaml
format-yaml: install-prettier ## Format YAML files with prettier
	@echo "✨ Running prettier to format YAML files..."
	prettier --write '$(YAML_FORMAT_PATH)/**/*.{yml,yaml}'
	@echo "✅ Prettier formatting complete"

.PHONY: install-yamllint
install-yamllint:
	@if ! command -v yamllint >/dev/null 2>&1; then \
		echo "📦 yamllint not found, installing..."; \
		case "$$(uname -s)" in \
			Darwin) brew install yamllint ;; \
			*) pip3 install --user yamllint ;; \
		esac; \
	fi

.PHONY: lint-yaml
lint-yaml: install-yamllint ## Lint YAML files with yamllint
	@echo "🔍 Running yamllint on YAML files..."
	yamllint -f colored $(YAML_LINT_PATH)
	@echo "✅ Yamllint passed"

endif
