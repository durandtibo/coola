ifndef PRETTIER_MK_INCLUDED
PRETTIER_MK_INCLUDED := 1

include $(dir $(lastword $(MAKEFILE_LIST)))tool-info.mk

# Shared target for installing prettier, used by yaml.mk and markdown.mk.

.PHONY: install-prettier
install-prettier:
	@if ! command -v prettier >/dev/null 2>&1; then \
		echo "📦 prettier not found, installing..."; \
		npm install -g prettier; \
	fi
	$(call show-tool,prettier)

endif
