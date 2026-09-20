ifndef PRETTIER_MK_INCLUDED
PRETTIER_MK_INCLUDED := 1

# Shared target for installing prettier, used by yaml.mk and markdown.mk.

.PHONY: install-prettier
install-prettier:
	@if ! command -v prettier >/dev/null 2>&1; then \
		echo "📦 prettier not found, installing..."; \
		npm install -g prettier; \
	fi

endif