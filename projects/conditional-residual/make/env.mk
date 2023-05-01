.PHONY: info builder-image python poetry pip install update init

## prints project info
info:
	@echo "Package Name: ${PACKAGE_NAME}"
	@echo "Path: ${PROJECT_DIR}"
	@echo "Builder Image: ${BUILDER_IMAGE}"
	@echo "Poetry: ${VERSION_POETRY}"
	@echo "Python: ${VERSION_PYTHON}"
	@echo "Pip: ${VERSION_PIP}"

## build project Docker image
builder-image: authenticate-docker
	@-docker pull ${BUILDER_IMAGE}
	@docker build \
		--platform linux/amd64 \
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
	@${DOCKERIZED_CMD} python -m pip install --upgrade poetry==${VERSION_POETRY}

## install Pip in virtual environment
pip: ${BUILDER_IMAGE_TASK} python
	@echo "Setting up PIP..."
	@${DOCKERIZED_CMD} poetry run pip install --upgrade pip==${VERSION_PIP}

## install dependencies
install: ${BUILDER_IMAGE_TASK} authenticate-poetry
	@echo "Installing dependencies..."
	@${DOCKERIZED_CMD} poetry install --extras extras

## update dependencies
update: ${BUILDER_IMAGE_TASK} authenticate-poetry
	@echo "Updating dependencies..."
	@${DOCKERIZED_CMD} poetry update

## initialize environment
init: ${BUILDER_IMAGE_TASK} python poetry pip install
	@echo "Environment is set up and ready for use."
