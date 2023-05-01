ARG         BASE_IMAGE
FROM        ${BASE_IMAGE}

ARG         DEBIAN_FRONTEND=noninteractive
RUN         apt-get update && \
            apt-get install --no-install-recommends -y \
                ca-certificates \
                git \
                curl \
                htop \
                build-essential \
                libreadline-dev \
                libncursesw5-dev \
                libssl-dev \
                libsqlite3-dev \
                libgdbm-dev \
                libc6-dev \
                libbz2-dev \
                libffi-dev \
                libpq-dev \
                liblzma-dev \
                tk-dev \
                && \
            apt-get clean && \
            rm -rf /var/lib/apt/lists/*

ARG         VERSION_PYTHON
RUN         git clone "https://github.com/pyenv/pyenv.git" ./pyenv && \
            (cd pyenv/plugins/python-build && ./install.sh) && \
            rm -rf pyenv

RUN         python-build --no-warn-script-location ${VERSION_PYTHON} /opt/python
ENV         PATH="/opt/python/bin:${PATH}"

ARG         VERSION_PIP
RUN	        python -m pip install --no-cache-dir --upgrade pip==${VERSION_PIP}

ARG         VERSION_POETRY
RUN         python -m pip install --upgrade poetry==${VERSION_POETRY}

ARG         PYPI_USERNAME
ARG         PYPI_PASSWORD
ARG         POETRY_HTTP_BASIC_PRIVATE_USERNAME=${PYPI_USERNAME}
ARG         POETRY_HTTP_BASIC_PRIVATE_PASSWORD=${PYPI_PASSWORD}
COPY        pyproject.toml /
COPY        poetry.lock /
RUN         poetry config virtualenvs.create false --local
RUN         poetry install --no-dev --no-root --no-interaction --extras extras && \
            rm -r poetry.toml pyproject.toml poetry.lock /root/.cache

ENV         PYTHONUNBUFFERED=1
ENTRYPOINT  ["mlflow", "run", "--env-manager", "local"]
