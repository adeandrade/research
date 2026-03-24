# Rate-Distortion Optimization for Transformer Inference

**Abstract:** Transformers achieve superior performance on many tasks, build
impose heavy compute and memory requirements during inference. This inference
can be made more efficient by partitioning the process across multiple devices,
which, in turn, requires compressing its intermediate representations. We
introduce a principled rate-distortion-based framework for lossy compression
that learns compact encodings that explicitly trade bitrate for accuracy.
Experiments on language benchmarks show that the simplest of the proposed codecs
achieves substantial rate savings, outperforming more complex methods. We
characterize and analyze the rate-distortion behaviour of transformers,
offering a unified lens for understanding performance in representation coding.
This formulation extends information-theoretic concepts to define the gap
between rate and entropy, and derive some of its bounds. We further develop
probably approximately correct (PAC)-style bounds for estimating this gap. For
different architectures and tasks, we empirically demonstrate that their rates
are driven by these bounds, adding to the explainability of the formulation.

This is the official code release of this publication. It is under the MIT
license. Please cite it as follows:

```bib
@article{RDOTI26,
  author       = {Anderson de Andrade and Alon Harell and Ivan Bajić},
  title        = {Rate-Distortion Optimization for Transformer Inference},
  journal      = {{ArXiv}},
  year         = {2026},
  volume       = {2601.22002},
}
```

## Using This Project

### Set Up Local Environment

Project settings are specified in `.envrc` and set using
[`direnv`](https://github.com/direnv/direnv). You need to allow them by running:

```bash
direnv allow
```

Dependencies and scripts are defined in `pyproject.toml`.
They can be installed using [`uv`](https://docs.astral.sh/uv).
Run the following command to build set it up:

```bash
make init
```

Script documentation can be obtained when running it with `--help`.

### Managing your Virtual Environments

We use `uv` to manage virtual environments. For details read the
[documentation here](https://docs.astral.sh/uv/getting-started/features).
Some of the most common commands are listed below:

```bash
# Add dependencies
uv add torch==2.10.0

# Install dependencies
make install

# Update dependencies
make update

# Activate local virtual environment
source .venv/bin/activate

# Package code
make build 
```

### Run Experiments

#### 1. Build and deploy your container

Make sure your dependencies are properly defined in your `uv.lock`
configuration file. If you have made changes to it, do not forget to update your
lock file by running:

```bash
make update
```

 To build your container image run:

```bash
make training-image
```

#### 2. Define entry point to launch experiments

Define entry points to your tasks in the `MLProject` file.

#### 3. Launch and Manage jobs

To launch and manage jobs using Slurm we created the following scripts:
`slurm-launch`, `slurm-log`, `slurm-status`, and `slurm-cancel`. The
documentation on how `slurm-launch` works is available by running:

```bash
slurm-launch --help
```

To launch jobs using the Kubernetes please use `k8s-launch`. Instructions for
this script is available by running:

```bash
k8s-launch --help
```

To manage Kubernetes jobs we recommend to use [k9s](https://github.com/derailed/k9s).
