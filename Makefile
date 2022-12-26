.PHONY : conda
conda :
	conda env create -f environment.yaml --force

.PHONY : config-poetry
config-poetry :
	poetry config experimental.system-git-client true
	poetry config --list

.PHONY : install
install :
	poetry install --no-interaction

.PHONY : update
update :
	-poetry self update
	poetry update
	-pre-commit autoupdate

.PHONY : lint
lint :
	flake8 .

.PHONY : format
format :
	black --check .

.PHONY : test
test :
	python -m pytest

.PHONY : unit-test
unit-test :
	python -m pytest --timeout 10 tests/unit

.PHONY : unit-test-cov
unit-test-cov :
	python -m pytest --timeout 10 --cov-report html --cov-report xml --cov-report term --cov=coola tests/unit

.PHONY : integration-test
integration-test :
	python -m pytest tests/integration

.PHONY : integration-test-cov
integration-test-cov :
	python -m pytest --cov-report html --cov-report xml --cov-report term --cov=coola --cov-append tests/integration
