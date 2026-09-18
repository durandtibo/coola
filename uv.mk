ifndef UV_MK_INCLUDED
UV_MK_INCLUDED := 1

# Targets for installing Python tools via uv.
#
# Optional variables (set before including this file):
#   PYTHON_VERSION ?= 3.14   # python version passed to `uv venv --python`

PYTHON_VERSION ?= 3.14

.PHONY: install-uv
install-uv:
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "📦 uv not found, installing..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi

.PHONY: install-invoke
install-invoke: install-uv
	@echo "📦 Installing invoke..."
	uv pip install "invoke>=3.0"
	@echo "✅ invoke installed"

.PHONY: update-uv
update-uv: install-uv
	uv self update

.PHONY: setup-venv
setup-venv:
	$(MAKE) update-uv
	uv venv --python $(PYTHON_VERSION) --clear
	$(MAKE) install-invoke
	.venv/bin/inv create-venv
	.venv/bin/inv install --docs-deps

endif
