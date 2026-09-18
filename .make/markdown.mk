ifndef MARKDOWN_MK_INCLUDED
MARKDOWN_MK_INCLUDED := 1

# Targets for formatting and linting Markdown files.
#
# Optional variables (set before including this file):
#   MARKDOWN_FORMAT_PATH ?= .            # path passed to prettier
#   MARKDOWN_LINT_GLOB   ?= **/*.md      # glob passed to markdownlint

MARKDOWN_FORMAT_PATH ?= .
MARKDOWN_LINT_GLOB ?= **/*.md

include prettier.mk

.PHONY: format-markdown
format-markdown: install-prettier
	@echo "✨ Running prettier to format Markdown files..."
	prettier --write '$(MARKDOWN_FORMAT_PATH)/**/*.md'
	@echo "✅ Prettier formatting complete"

.PHONY: install-markdownlint
install-markdownlint:
	@if ! command -v markdownlint >/dev/null 2>&1; then \
		echo "📦 markdownlint not found, installing..."; \
		npm install -g markdownlint-cli; \
	fi

.PHONY: lint-markdown
lint-markdown: install-markdownlint
	@echo "🔍 Running markdownlint on Markdown files..."
	markdownlint $(MARKDOWN_LINT_GLOB)
	@echo "✅ Markdownlint passed"

endif
