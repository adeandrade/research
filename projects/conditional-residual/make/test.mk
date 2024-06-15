.PHONY: format type-check test lint

## run linter
format: ${BUILDER_IMAGE_TASK}
	@echo "Linting format..."
	@${DOCKERIZED_CMD} poetry run ruff check --fix .

## run type checker
type-check: ${BUILDER_IMAGE_TASK}
	@echo "Checking types..."
	@${DOCKERIZED_CMD} poetry run basedpyright

## run pytest
test: ${BUILDER_IMAGE_TASK} install
	@echo "Running tests..."
	@${DOCKERIZED_CMD} poetry run pytest tests/

## run all linting tools
lint: format type-check
