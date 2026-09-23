ifndef SHELL_MK_INCLUDED
SHELL_MK_INCLUDED := 1

# Targets for formatting and linting shell scripts.
#
# Optional variables (set before including this file):
#   SHELL_FORMAT_PATH ?= .   # path passed to shfmt (walked recursively)
#   SHELL_LINT_PATH   ?= .   # path searched for *.sh files passed to shellcheck

SHELL_FORMAT_PATH ?= .
SHELL_LINT_PATH ?= .
SHFMT_VERSION ?= v3.10.0

.PHONY: install-shellcheck
install-shellcheck:
	@if ! command -v shellcheck >/dev/null 2>&1; then \
		echo "📦 shellcheck not found, installing..."; \
		case "$$(uname -s)" in \
			Darwin) brew install shellcheck ;; \
			*) sudo apt-get update && sudo apt-get install -y shellcheck ;; \
		esac; \
	fi

.PHONY: install-shfmt
install-shfmt:
	@if ! command -v shfmt >/dev/null 2>&1; then \
		echo "📦 shfmt not found, installing..."; \
		case "$$(uname -s)" in \
			Darwin) brew install shfmt ;; \
			*) \
			if command -v go >/dev/null 2>&1 && go install mvdan.cc/sh/v3/cmd/shfmt@latest; then \
				gobin="$$(go env GOBIN)"; \
				if [ -z "$$gobin" ]; then gobin="$$(go env GOPATH)/bin"; fi; \
					sudo cp "$$gobin/shfmt" /usr/local/bin/shfmt; \
				else \
					arch="$$(uname -m)"; \
					case "$$arch" in \
						x86_64) arch=amd64 ;; \
						aarch64|arm64) arch=arm64 ;; \
					esac; \
					curl -fsSL -o /tmp/shfmt "https://github.com/mvdan/sh/releases/download/$(SHFMT_VERSION)/shfmt_$(SHFMT_VERSION)_linux_$${arch}"; \
					chmod +x /tmp/shfmt; \
					sudo mv /tmp/shfmt /usr/local/bin/shfmt; \
				fi; \
				;; \
		esac; \
	fi

.PHONY: lint-shell
lint-shell: install-shellcheck ## Lint shell scripts with shellcheck
	@echo "🐚 Running shellcheck on shell scripts..."
	find $(SHELL_LINT_PATH) -type d -name '.make' -prune -o -type f -name '*.sh' -print0 | xargs -0 -r shellcheck
	@echo "✅ Shellcheck passed"

.PHONY: format-shell
format-shell: install-shfmt ## Format shell scripts with shfmt
	@echo "🔧 Running shfmt to format shell scripts..."
	find $(SHELL_FORMAT_PATH) -type d -name '.make' -prune -o -type f -name '*.sh' -print0 | xargs -0 -r shfmt -l -w
	@echo "✅ Shell formatting complete"

endif
