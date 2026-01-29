# Lossy Common Information in a Learnable Gray-Wyner Network

**Abstract:** Many computer vision tasks share substantial overlapping
information, yet conventional codecs tend to ignore this, leading to redundant
and inefficient representations. The Gray-Wyner network, a classical concept
from information theory, offers a principled framework for separating common
and task-specific information. Inspired by this idea, we develop a Learnable
three-channel codec that disentangles shared information from task-specific
details across multiple vision tasks. We characterize the limits of this
approach through the notion of lossy common information, and propose an
optimization objective that balances inherent tradeoffs in learning such
representations. Through comparisons of three codec architectures on two-task
scenarios spanning six vision benchmarks, we demonstrate that our approach
substantially reduces redundancy and consistently outperforms independent
coding. These results highlight the practical value of revisiting Gray-Wyner
theory in modern machine learning contexts, bridging classic information theory
with task-driven representation learning.

This is the official code release of this publication. It is under the MIT
license. Please cite it as follows:

```bib
@inproceedings{LCILGWN26,
  author       = {Anderson de Andrade and Alon Harell and Ivan Bajić},
  title        = {Lossy Common Information in a Learnable Gray-Wyner Network},
  booktitle    = {{ICLR}},
  year         = {2026},
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
They can be installed using [`poetry`](https://github.com/python-poetry/poetry).
Run the following command to build set it up:

```bash
make init
```

Script documentation can be obtained when running it with `--help`.

### Managing your Virtual Environments

We use `poetry` to manage virtual environments. For details read the
[documentation here](https://python-poetry.org/docs/basic-usage). Some of the
most common commands are listed below:

```bash
# Add dependencies
poetry add torch==2.10.0

# Install dependencies
make install

# Update dependencies
make update

# Activate local virtual environment
poetry shell

# Package code
make build 
```

### Run Experiments

#### 1. Build and deploy your container

Make sure your dependencies are properly defined in your `poetry.lock`
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
