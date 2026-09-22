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
	@if [ -d "$(SHARED_MAKEFILES_PREFIX)" ] && ! git log --grep="git-subtree-dir: $(SHARED_MAKEFILES_PREFIX)$$" --format=%H -1 | grep -q .; then \
		echo "⚠️  $(SHARED_MAKEFILES_PREFIX) exists but has no subtree history; re-adding it..."; \
		git rm -rq $(SHARED_MAKEFILES_PREFIX); \
		git commit -m "chore: remove $(SHARED_MAKEFILES_PREFIX) before subtree re-add"; \
		git subtree add --prefix=$(SHARED_MAKEFILES_PREFIX) $(SHARED_MAKEFILES_REMOTE_NAME) $(SHARED_MAKEFILES_BRANCH) --squash -m "chore: sync shared Makefile subtree"; \
	elif [ ! -d "$(SHARED_MAKEFILES_PREFIX)" ]; then \
		git subtree add --prefix=$(SHARED_MAKEFILES_PREFIX) $(SHARED_MAKEFILES_REMOTE_NAME) $(SHARED_MAKEFILES_BRANCH) --squash -m "chore: sync shared Makefile subtree"; \
	else \
		git subtree pull --prefix=$(SHARED_MAKEFILES_PREFIX) $(SHARED_MAKEFILES_REMOTE_NAME) $(SHARED_MAKEFILES_BRANCH) --squash -m "chore: sync shared Makefile subtree" || true; \
	fi
	@if [ -d "$(SHARED_MAKEFILES_PREFIX)/.github" ] || [ -d "$(SHARED_MAKEFILES_PREFIX)/testdata" ] || [ -f "$(SHARED_MAKEFILES_PREFIX)/.gitignore" ]; then \
		echo "🧹 Removing .github/, testdata/, and .gitignore from $(SHARED_MAKEFILES_PREFIX)..."; \
		git rm -rq --ignore-unmatch $(SHARED_MAKEFILES_PREFIX)/.github $(SHARED_MAKEFILES_PREFIX)/testdata $(SHARED_MAKEFILES_PREFIX)/.gitignore; \
		rm -rf $(SHARED_MAKEFILES_PREFIX)/.github $(SHARED_MAKEFILES_PREFIX)/testdata $(SHARED_MAKEFILES_PREFIX)/.gitignore; \
		git commit -m "chore: remove .github, testdata, and .gitignore from synced subtree" || true; \
	fi
	@echo "✅ Subtree sync complete"

endif
