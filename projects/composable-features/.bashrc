# Names
export PACKAGE_NAME="composable-features"
export SLACK_USER="U02MSJM2ANN"

# Versions
export VERSION_PYTHON="3.9.9"
export VERSION_POETRY="1.1.12"
export VERSION_PIP="22.0.4"
export VERSION_UBUNTU="20.04"

# Paths
export PATH="${PWD}/bin:${PATH}"
export PROJECT_DIR="${PWD}"
export POETRY_CACHE="${HOME}/.cache"

# URIs
export DOCKER_URL="registry.hub.docker.com/multimedialabsfu/research"
export PYPI_HOST="https://pypi.org/simple"

# Docker
if [[ -n "${USE_DOCKER}" ]]; then
    export DOCKERIZED_CMD="dockerized"
    export BUILDER_IMAGE_TASK="builder-image"
fi

export BUILDER_IMAGE="${DOCKER_URL}:${PACKAGE_NAME}_builder"
export TRAINING_IMAGE="${DOCKER_URL}:${PACKAGE_NAME}_training"
export TRAINING_BASE_IMAGE="ubuntu:${VERSION_UBUNTU}"

# Flags
export DEVELOPMENT_MODE=1
