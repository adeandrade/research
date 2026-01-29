ARG     VERSION_PYTHON
FROM    python:${VERSION_PYTHON}-slim

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

RUN     pip install --upgrade pip
RUN     pip install --upgrade poetry

ENTRYPOINT ["/bin/bash", "-c"]
