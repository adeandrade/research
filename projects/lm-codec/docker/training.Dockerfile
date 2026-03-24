ARG         BASE_IMAGE=ubuntu:latest
FROM        ${BASE_IMAGE}

ARG         DEBIAN_FRONTEND=noninteractive

RUN         apt-get update && \
            apt-get install --no-install-recommends -y \
                ca-certificates \
                git \
                curl \
                wget \
                htop \
                build-essential \
                && \
            apt-get clean && \
            rm -rf /var/lib/apt/lists/*

RUN         curl -LsSf https://astral.sh/uv/install.sh | sh
ENV         PATH="/root/.local/bin/:${PATH}"

COPY        pyproject.toml uv.lock .python-version /
RUN         --mount=type=secret,id=PYPI_USERNAME,env=UV_INDEX_PRIVATE_USERNAME \
            --mount=type=secret,id=PYPI_PASSWORD,env=UV_INDEX_PRIVATE_PASSWORD \
            uv sync --no-dev --no-install-project && \
            rm -r pyproject.toml uv.lock .python-version /root/.cache
ENV         PATH="/.venv/bin:${PATH}"

COPY        torch_ans/dist/torch_ans-0.1.1.post1-cp313-cp313-linux_x86_64.whl /
RUN         uv pip install /torch_ans-0.1.1.post1-cp313-cp313-linux_x86_64.whl && rm /*.whl

ENV         PYTHONUNBUFFERED=1
ENTRYPOINT  ["mlflow", "run", "--env-manager", "local"]
