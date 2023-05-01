.PHONY: build deploy

VERSION_PACKAGE ?= 0.0.1rc0

## build python package
build: ${BUILDER_IMAGE_TASK} install lint test
	@${DOCKERIZED_CMD} poetry version ${VERSION_PACKAGE}
	@${DOCKERIZED_CMD} poetry build

## version and deploy python package to artifact repository
deploy: ${BUILDER_IMAGE_TASK} install lint test
	@${DOCKERIZED_CMD} poetry version ${VERSION_PACKAGE}
	@${DOCKERIZED_CMD} poetry publish \
		--build \
		--repository private-upload \
		--username "${PYPI_USERNAME}" \
		--password "${PYPI_PASSWORD}"
