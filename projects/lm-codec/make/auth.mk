.PHONY: authenticate-docker authenticate

## login to docker registry
authenticate-docker:
	@echo "${DOCKER_PASSWORD}" | docker login --username "${DOCKER_USERNAME}" --password-stdin "${DOCKER_URL}"
	@echo "Authenticated docker!"

## authenticate all tools
authenticate: authenticate-docker
