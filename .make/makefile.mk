ifndef MAKEFILE_MK_INCLUDED
MAKEFILE_MK_INCLUDED := 1

# Targets for formatting and linting Makefiles and *.mk files.
#
# Optional variables (set before including this file):
#   MAKEFILE_FORMAT_FILES ?= Makefile *.mk   # files passed to mbake format
#   MAKEFILE_LINT_FILES   ?= Makefile *.mk   # files passed to checkmake
#   MBAKE_CONFIG          ?= <this dir>/bake.toml   # config passed to mbake format

MAKEFILE_FORMAT_FILES ?= Makefile *.mk
MAKEFILE_LINT_FILES ?= Makefile *.mk

MBAKE_CONFIG ?= $(dir $(lastword $(MAKEFILE_LIST)))bake.toml

CHECKMAKE_VERSION ?= v0.3.2

# mbake has no Homebrew formula, so macOS also goes through pipx/pip.
.PHONY: install-mbake
install-mbake:
	@if ! command -v mbake >/dev/null 2>&1; then \
		echo "📦 mbake not found, installing..."; \
		if command -v pipx >/dev/null 2>&1; then \
			pipx install mbake; \
		elif [ "$$(uname -s)" = "Darwin" ]; then \
			brew install pipx && pipx install mbake; \
		else \
			pip3 install --user --break-system-packages mbake; \
		fi; \
	fi

.PHONY: install-checkmake
install-checkmake:
	@if ! command -v checkmake >/dev/null 2>&1; then \
		echo "📦 checkmake not found, installing..."; \
		case "$$(uname -s)" in \
			Darwin) brew install checkmake ;; \
			*) \
			if command -v go >/dev/null 2>&1 && go install github.com/mrtazz/checkmake/cmd/checkmake@latest; then \
				: ; \
			else \
				arch="$$(uname -m)"; \
				case "$$arch" in \
					x86_64) arch=amd64 ;; \
					aarch64|arm64) arch=arm64 ;; \
				esac; \
					curl -fsSL -o /tmp/checkmake "https://github.com/checkmake/checkmake/releases/download/$(CHECKMAKE_VERSION)/checkmake-$(CHECKMAKE_VERSION).linux.$${arch}"; \
					chmod +x /tmp/checkmake; \
					sudo mv /tmp/checkmake /usr/local/bin/checkmake; \
				fi; \
				;; \
		esac; \
	fi

.PHONY: format-makefile
format-makefile: install-mbake ## Format Makefiles and .mk files with mbake
	@echo "✨ Running mbake to format Makefiles..."
	@files=""; \
	for pattern in $(MAKEFILE_FORMAT_FILES); do \
		for f in $$pattern; do \
			[ -e "$$f" ] && files="$$files $$f"; \
		done; \
	done; \
	if [ -n "$$files" ]; then \
		mbake format --config $(MBAKE_CONFIG) $$files; \
	else \
		echo "No files found matching: $(MAKEFILE_FORMAT_FILES)"; \
	fi
	@echo "✅ Makefile formatting complete"

.PHONY: lint-makefile
lint-makefile: install-checkmake ## Lint Makefiles and .mk files with checkmake
	@echo "🔍 Running checkmake on Makefiles..."
	checkmake $(MAKEFILE_LINT_FILES)
	@echo "✅ Checkmake passed"

endif
