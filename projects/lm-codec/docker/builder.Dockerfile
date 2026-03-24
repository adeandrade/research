ARG     VERSION_PYTHON=latest
FROM    python:${VERSION_PYTHON}

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

RUN     curl -LsSf https://astral.sh/uv/install.sh | sh
ENV     PATH="/root/.local/bin/:${PATH}"

ENTRYPOINT ["/bin/bash", "-c"]
