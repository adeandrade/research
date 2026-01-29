.PHONY: info builder-image python poetry install update init

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
		--build-arg VERSION_POETRY=${VERSION_POETRY} \
		--build-arg VERSION_PIP=${VERSION_PIP} \
		--tag ${BUILDER_IMAGE} \
		- < docker/builder.Dockerfile
	@docker push ${BUILDER_IMAGE}

## install python via PyEnv
python: ${BUILDER_IMAGE_TASK}
	@echo "Setting up Python..."
	@${DOCKERIZED_CMD} pyenv install --skip-existing ${VERSION_PYTHON}
	@${DOCKERIZED_CMD} pyenv local ${VERSION_PYTHON}

## change global Poetry version
poetry: ${BUILDER_IMAGE_TASK}
	@echo "Setting up Poetry..."
	@${DOCKERIZED_CMD} python -m pip install --upgrade pip
	@${DOCKERIZED_CMD} python -m pip install --upgrade poetry

## install dependencies
install: ${BUILDER_IMAGE_TASK} authenticate-poetry
	@echo "Installing dependencies..."
	@${DOCKERIZED_CMD} poetry install --extras extras

## update dependencies
update: ${BUILDER_IMAGE_TASK} authenticate-poetry
	@echo "Updating dependencies..."
	@${DOCKERIZED_CMD} poetry update

## initialize environment
init: ${BUILDER_IMAGE_TASK} python poetry install
	@echo "Environment is set up and ready for use."
