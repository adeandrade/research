.PHONY: authenticate-poetry authenticate-docker authenticate

## set private repository credentials
authenticate-poetry: ${BUILDER_IMAGE_TASK}
	@${DOCKERIZED_CMD} poetry config http-basic.private "${PYPI_USERNAME}" "${PYPI_PASSWORD}"
	@echo "Authenticated poetry!"

## login to docker registry
authenticate-docker:
	@echo "${DOCKER_PASSWORD}" | docker login --username "${DOCKER_USERNAME}" --password-stdin "${DOCKER_URL}"
	@echo "Authenticated docker!"

## authenticate all tools
authenticate: authenticate-poetry authenticate-docker
