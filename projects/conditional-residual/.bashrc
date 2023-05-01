# Names
export PACKAGE_NAME="conditional-residual"
export SLACK_USER="U02MSJM2ANN"

# Versions
export VERSION_PYTHON="3.10.6"
export VERSION_POETRY="1.4.2"
export VERSION_PIP="23.1.2"
export VERSION_UBUNTU="22.04"

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
