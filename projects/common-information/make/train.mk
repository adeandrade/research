.PHONY: training-image

## create training Docker image
training-image: authenticate-docker clean
	@echo "Building training image for ${PACKAGE_NAME}..."
	@docker build \
		--platform linux/amd64 \
		--build-arg BASE_IMAGE="${TRAINING_BASE_IMAGE}" \
		--build-arg VERSION_PYTHON="${VERSION_PYTHON}" \
		--secret id=PYPI_USERNAME \
		--secret id=PYPI_PASSWORD \
		--tag ${TRAINING_IMAGE} \
		--file docker/training.Dockerfile \
		.
	@docker push ${TRAINING_IMAGE}
