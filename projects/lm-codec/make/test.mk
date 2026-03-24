.PHONY: format type-check test lint

## run linter
format: ${BUILDER_IMAGE_TASK}
	@echo "Linting format..."
	@${DOCKERIZED_CMD} uv run ruff check
	@${DOCKERIZED_CMD} uv run ruff format

## run type checker
type-check: ${BUILDER_IMAGE_TASK}
	@echo "Checking types..."
	@${DOCKERIZED_CMD} uv run basedpyright

## run pytest
test: ${BUILDER_IMAGE_TASK} install
	@echo "Running tests..."
	@${DOCKERIZED_CMD} uv run pytest tests/

## run all linting tools
lint: format type-check
