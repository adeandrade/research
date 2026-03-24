.PHONY: info builder-image python install update init

## prints project info
info:
	@echo "Package Name: ${PACKAGE_NAME}"
	@echo "Path: ${PROJECT_DIR}"
	@echo "Builder Image: ${BUILDER_IMAGE}"
	@echo "Python: ${VERSION_PYTHON}"

## build project Docker image
builder-image: authenticate-docker
	@-docker pull ${BUILDER_IMAGE}
	@docker build \
		--rm \
		--cache-from ${BUILDER_IMAGE} \
		--build-arg VERSION_PYTHON=${VERSION_PYTHON} \
		--tag ${BUILDER_IMAGE} \
		- < docker/builder.Dockerfile
	@docker push ${BUILDER_IMAGE}

## setup Python
python: ${BUILDER_IMAGE_TASK}
	@echo "Setting up Python..."
	@${DOCKERIZED_CMD} uv python pin ${VERSION_PYTHON}

## install dependencies
install: ${BUILDER_IMAGE_TASK}
	@echo "Installing dependencies..."
	@${DOCKERIZED_CMD} uv sync --inexact

## update dependencies
update: ${BUILDER_IMAGE_TASK}
	@echo "Updating dependencies..."
	@${DOCKERIZED_CMD} uv lock

## initialize environment
init: ${BUILDER_IMAGE_TASK} python install
	@echo "Environment is set up and ready for use."
