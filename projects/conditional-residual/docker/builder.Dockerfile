ARG     VERSION_PYTHON
FROM    python:${VERSION_PYTHON}-slim-buster

# Install system packages
RUN     apt-get update && \
        apt-get install -y --no-install-recommends \
            git \
            curl \
            build-essential \
            python3-dev \
            libpq-dev \
            && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Pip
ARG     VERSION_PIP
RUN     pip install --upgrade pip==${VERSION_PIP}

# Install Poetry
ARG     VERSION_POETRY
RUN     pip install --upgrade poetry==${VERSION_POETRY}

ENTRYPOINT ["/bin/bash", "-c"]
