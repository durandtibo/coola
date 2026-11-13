ifndef TOML_MK_INCLUDED
TOML_MK_INCLUDED := 1

# Targets for formatting TOML files.
#
# Optional variables (set before including this file):
#   TOML_FORMAT_PATH ?= .   # path searched for *.toml files passed to taplo

TOML_FORMAT_PATH ?= .

.PHONY: install-taplo
install-taplo:
	@if ! command -v taplo >/dev/null 2>&1; then \
		echo "📦 taplo not found, installing..."; \
		case "$$(uname -s)" in \
			Darwin) brew install taplo ;; \
			*) npm install -g @taplo/cli ;; \
		esac; \
	fi

.PHONY: format-toml
format-toml: install-taplo ## Format TOML files with taplo
	@echo "✨ Running taplo to format TOML files..."
	find $(TOML_FORMAT_PATH) -type d -name '.make' -prune -o -type f -name '*.toml' -print0 | xargs -0 -r taplo format
	@echo "✅ Taplo formatting complete"

endif
