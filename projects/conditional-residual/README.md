# Conditional and Residual Methods in Scalable Coding for Humans and Machines

This is the official code release of this publication. It is under the MIT license. Please cite it as follows:

```bib
@inproceedings{CRMSCHUM23,
  author       = {Anderson de Andrade and
                  Alon Harell and
                  Yalda Foroutan and
                  Ivan Bajić},
  title        = {Conditional and Residual Methods in Scalable Coding for Humans and Machines},
  booktitle    = {{IEEE} International Conference on Multimedia and Expo Workshop on Coding for Machines, {ICME CfM} 2023},
  year         = {2023},
}
```

# Using This Project

## Set Up Local Environment
First, you also need to allow `direnv` by using:

```bash
direnv allow
```

Then, to get started, run the following command to build set up your project.
```bash
make init
```

## Managing your Virtual Environments
We use `poetry` to manage virtual environments. For details read the [documentation here](https://python-poetry.org/docs/basic-usage). Some of the most common commands are listed below:

```bash
# Add dependencies
poetry add scikit-learn==2.1.2

# Install dependencies
make install

# Update dependencies
make update

# Activate local virtual environment
poetry shell

# Package code
make build 
```

## Run Experiments

#### 1. Build and deploy your container

Make sure your dependencies are properly defined in your `poetry.toml` configuration file. If you have made changes to it, do not forget to update your lock file by running:
```
make update
```
 To build your container image run:
```bash
make training-image
```

#### 2. Define entry point to launch experiments
Define entrypoints to your tasks in the `MLProject` file. Two empty tasks are available as examples.

#### 3. Launch and Manage jobs
To launch and manage jobs using Compute Canada resources we created the following scripts: `slurm-launch`, `slurm-log`, `slurm-status`, and `slurm-cancel`. The documentation on how `slurm-launch` works is available by running:
```bash
slurm-launch --help
```

To launch jobs using the SFU Kubernetes cluster please use `k8s-launch`. Instructions for this script is available by running:
```bash
k8s-launch --help
```
To manage Kubernetes jobs we recommend to use [k9s](https://github.com/derailed/k9s).

Your MLFlow experiment ID could be the same as the package name.
