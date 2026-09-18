ifndef SELF_MK_INCLUDED
SELF_MK_INCLUDED := 1

# Target for syncing the shared-makefiles subtree in a consuming project.
#
# Optional variables (set before including this file):
#   SHARED_MAKEFILES_REMOTE_NAME ?= shared-makefiles                                   # git remote name
#   SHARED_MAKEFILES_REMOTE_URL  ?= https://github.com/durandtibo/shared-makefiles.git # git remote URL
#   SHARED_MAKEFILES_BRANCH      ?= main                                               # branch to pull
#   SHARED_MAKEFILES_PREFIX      ?= .make                                              # subtree prefix

SHARED_MAKEFILES_REMOTE_NAME ?= shared-makefiles
SHARED_MAKEFILES_REMOTE_URL ?= https://github.com/durandtibo/shared-makefiles.git
SHARED_MAKEFILES_BRANCH ?= main
SHARED_MAKEFILES_PREFIX ?= .make

.PHONY: update-subtree
update-subtree:
	@echo "🔄 Syncing $(SHARED_MAKEFILES_PREFIX) subtree from $(SHARED_MAKEFILES_REMOTE_URL)..."
	git remote add $(SHARED_MAKEFILES_REMOTE_NAME) $(SHARED_MAKEFILES_REMOTE_URL) || true
	git fetch $(SHARED_MAKEFILES_REMOTE_NAME) $(SHARED_MAKEFILES_BRANCH)
	git subtree pull --prefix=$(SHARED_MAKEFILES_PREFIX) $(SHARED_MAKEFILES_REMOTE_NAME) $(SHARED_MAKEFILES_BRANCH) --squash -m "chore: sync shared Makefile subtree" || true
	@if [ -d "$(SHARED_MAKEFILES_PREFIX)/.github" ] || [ -d "$(SHARED_MAKEFILES_PREFIX)/testdata" ]; then \
		echo "🧹 Removing .github/ and testdata/ from $(SHARED_MAKEFILES_PREFIX)..."; \
		git rm -rq --ignore-unmatch $(SHARED_MAKEFILES_PREFIX)/.github $(SHARED_MAKEFILES_PREFIX)/testdata; \
		rm -rf $(SHARED_MAKEFILES_PREFIX)/.github $(SHARED_MAKEFILES_PREFIX)/testdata; \
		git commit -m "chore: remove .github and testdata from synced subtree" || true; \
	fi
	@echo "✅ Subtree sync complete"

endif