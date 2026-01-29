.PHONY: clean-pyc clean-c clean-test clean-build clean-latex clean

## clean compiled python files
clean-pyc:
	@find . -path ./.venv -prune -o -name '*.pyc' -exec rm -f {} +
	@find . -path ./.venv -prune -o -name '*.pyo' -exec rm -f {} +
	@find . -path ./.venv -prune -o -name '*~' -exec rm -f {} +
	@find . -path ./.venv -prune -o -name '__pycache__' -exec rm -fr {} +

## clean transpiled and compiled C files
clean-c:
	@find . -path ./.venv -prune -o -name '*.so' -exec rm -fr {} +
	@find . -path ./.venv -prune -o -name '*.o' -exec rm -fr {} +

## clean cached test data
clean-test:
	@find . -path ./.venv -prune -o -name 'tox' -exec rm -fr {} +
	@find . -path ./.venv -prune -o -name 'htmlcov' -exec rm -fr {} +
	@find . -path ./.venv -prune -o -name '.mypy_cache' -exec rm -fr {} +
	@find . -path ./.venv -prune -o -name '.pytest_cache' -exec rm -fr {} +
	@find . -path ./.venv -prune -o -name '.ruff_cache' -exec rm -fr {} +

## clean build and distribution files
clean-build:
	@find . -path ./.venv -prune -o -name 'dist' -exec rm -fr {} +
	@find . -path ./.venv -prune -o -name '*.pex' -exec rm -fr {} +
	@find . -path ./.venv -prune -o -name 'pip-wheel-metadata' -exec rm -fr {} +
	@find . -path ./.venv -prune -o -name '*.egg-info' -exec rm -fr {} +
	@rm -rf build dist

## clean LaTeX build files
clean-latex:
	@find . -path ./.venv -prune -o -name '*.aux' -exec rm -fr {} +
	@find . -path ./.venv -prune -o -name '*.fdb_latexmk' -exec rm -fr {} +
	@find . -path ./.venv -prune -o -name '*.fls' -exec rm -fr {} +
	@find . -path ./.venv -prune -o -name '*.log' -exec rm -fr {} +
	@find . -path ./.venv -prune -o -name '*.bbl' -exec rm -fr {} +
	@find . -path ./.venv -prune -o -name '*.blg' -exec rm -fr {} +
	@find . -path ./.venv -prune -o -name '*.nav' -exec rm -fr {} +
	@find . -path ./.venv -prune -o -name '*.out' -exec rm -fr {} +
	@find . -path ./.venv -prune -o -name '*.snm' -exec rm -fr {} +
	@find . -path ./.venv -prune -o -name '*.toc' -exec rm -fr {} +
	@find . -path ./.venv -prune -o -name '*.synctex.gz' -exec rm -fr {} +
	@find . -path ./.venv -prune -o -name '*.bcf' -exec rm -fr {} +
	@find . -path ./.venv -prune -o -name '*.run.xml' -exec rm -fr {} +

## clean up all temporary files
clean: clean-pyc clean-c clean-test clean-build clean-latex
