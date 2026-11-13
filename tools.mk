ifndef TOOLS_MK_INCLUDED
TOOLS_MK_INCLUDED := 1

# Target that reports which command-line tools are installed.
#
# Every tool of `TOOLS` is looked up on the `PATH`; one line per tool says where it was found or
# that it is missing. All tools are listed so one run shows everything that is missing. The target
# fails with exit code 127 if a tool is missing.
#
# Optional variables (set before including this file, or on the command line):
#   TOOLS ?= actionlint bats ...   # space-separated tools to check (default: tools used by this repo)

TOOLS ?= actionlint bats checkmake gh markdownlint mbake prettier shellcheck shfmt taplo yamllint

.PHONY: check-tools
check-tools: ## Show which tools are installed and which are missing
	@echo "🧰 Checking the tools..."
	@missing=""; total=0; \
	for tool in $(TOOLS); do \
		total=$$((total + 1)); \
		if path=$$(command -v "$$tool"); then \
			echo "✅ $$tool: $$path"; \
		else \
			echo "❌ $$tool: not found"; \
			missing="$$missing $$tool"; \
		fi; \
	done; \
	if [ -n "$$missing" ]; then \
		echo "⚠️ Missing:$$missing"; \
		exit 127; \
	fi; \
	echo "✅ All the $$total tools are available"

endif
