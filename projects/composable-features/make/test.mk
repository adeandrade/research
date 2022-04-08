.PHONY: flake8 mypy test lint

## run flake8 formatting
flake8: ${BUILDER_IMAGE_TASK}
	@echo "Linting format..."
	@${DOCKERIZED_CMD} poetry run flake8 .

## run mypy type checking
mypy: ${BUILDER_IMAGE_TASK}
	@echo "Checking types..."
	@${DOCKERIZED_CMD} poetry run mypy --install-types --non-interactive .

## run pytest
test: ${BUILDER_IMAGE_TASK} install
	@echo "Running tests..."
	@${DOCKERIZED_CMD} poetry run pytest tests/

## run all linting tools
lint: flake8 mypy
