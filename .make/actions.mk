ifndef ACTIONS_MK_INCLUDED
ACTIONS_MK_INCLUDED := 1

# Targets for linting GitHub Actions workflow files.
#
# Optional variables (set before including this file):
#   ACTIONS_LINT_PATH ?= .   # path searched for workflow/action files passed to actionlint

ACTIONS_LINT_PATH ?= .

.PHONY: install-actionlint
install-actionlint:
	@if ! command -v actionlint >/dev/null 2>&1; then \
		echo "📦 actionlint not found, installing..."; \
		case "$$(uname -s)" in \
			Darwin) brew install actionlint ;; \
			*) \
			if command -v go >/dev/null 2>&1 && go install github.com/rhysd/actionlint/cmd/actionlint@latest; then \
				gobin="$$(go env GOBIN)"; \
				if [ -z "$$gobin" ]; then gobin="$$(go env GOPATH)/bin"; fi; \
					sudo cp "$$gobin/actionlint" /usr/local/bin/actionlint; \
				else \
					curl -fsSL https://raw.githubusercontent.com/rhysd/actionlint/main/scripts/download-actionlint.bash | bash -s -- latest /tmp; \
					sudo mv /tmp/actionlint /usr/local/bin/actionlint; \
				fi \
				;; \
		esac; \
	fi

.PHONY: lint-actions
lint-actions: install-actionlint ## Lint GitHub Actions workflow files with actionlint
	@echo "⚙️ Running actionlint on GitHub Actions workflow files..."
	find $(ACTIONS_LINT_PATH) -type d -name '.make' -prune -o -type f -path '*/.github/workflows/*' \( -name '*.yml' -o -name '*.yaml' \) -print0 \
		| xargs -0 -r actionlint -color
	@echo "✅ Actionlint passed"

endif