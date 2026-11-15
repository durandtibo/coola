SHELL = /bin/bash

include .make/all.mk

.PHONY: clean
clean: ## Remove the generated coverage reports and caches
	rm -rf coverage.xml .coverage htmlcov .pytest_cache .ruff_cache .hypothesis
