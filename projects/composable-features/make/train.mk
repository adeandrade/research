.PHONY: training-image

## create training Docker image
training-image: authenticate-docker clean
	@echo "Building training image for ${PACKAGE_NAME}..."
	@docker build \
		--platform linux/amd64 \
		--build-arg BASE_IMAGE="${TRAINING_BASE_IMAGE}" \
		--build-arg VERSION_PYTHON="${VERSION_PYTHON}" \
		--build-arg VERSION_PIP="${VERSION_PIP}" \
		--build-arg VERSION_POETRY="${VERSION_POETRY}" \
		--build-arg PYPI_USERNAME="${PYPI_USERNAME}" \
		--build-arg PYPI_PASSWORD="${PYPI_PASSWORD}" \
		--tag ${TRAINING_IMAGE} \
		--file docker/training.Dockerfile \
		.
	@docker push ${TRAINING_IMAGE}
