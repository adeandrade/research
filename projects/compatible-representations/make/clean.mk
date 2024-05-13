.PHONY: clean-pyc clean-c clean-test clean-build clean

## clean compiled python files
clean-pyc:
	@find . -name '*.pyc' -exec rm -f {} +
	@find . -name '*.pyo' -exec rm -f {} +
	@find . -name '*~' -exec rm -f {} +
	@find . -name '__pycache__' -exec rm -fr {} +

## clean transpiled and compiled C files
clean-c:
	@find . -name '*.so' -exec rm -fr {} +
	@find . -name '*.o' -exec rm -fr {} +

## clean cached test data
clean-test:
	@find . -name 'tox' -exec rm -fr {} +
	@find . -name 'htmlcov' -exec rm -fr {} +
	@find . -name '.mypy_cache' -exec rm -fr {} +
	@find . -name '.pytest_cache' -exec rm -fr {} +
	@find . -name '.ruff_cache' -exec rm -fr {} +

## clean build and distribution files
clean-build:
	@find . -name 'dist' -exec rm -fr {} +
	@find . -name '*.pex' -exec rm -fr {} +
	@find . -name 'pip-wheel-metadata' -exec rm -fr {} +
	@find . -name '*.egg-info' -exec rm -fr {} +
	@rm -rf build dist

## clean LaTeX build files
clean-latex:
	@find . -name '*.aux' -exec rm -fr {} +
	@find . -name '*.fdb_latexmk' -exec rm -fr {} +
	@find . -name '*.fls' -exec rm -fr {} +
	@find . -name '*.log' -exec rm -fr {} +
	@find . -name '*.bbl' -exec rm -fr {} +
	@find . -name '*.blg' -exec rm -fr {} +
	@find . -name '*.nav' -exec rm -fr {} +
	@find . -name '*.out' -exec rm -fr {} +
	@find . -name '*.snm' -exec rm -fr {} +
	@find . -name '*.toc' -exec rm -fr {} +
	@find . -name '*.synctex.gz' -exec rm -fr {} +
	@find . -name '*.bcf' -exec rm -fr {} +
	@find . -name '*.run.xml' -exec rm -fr {} +

## clean up all temporary files
clean: clean-pyc clean-c clean-test clean-build clean-latex
