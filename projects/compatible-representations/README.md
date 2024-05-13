# Towards Task-Compatible Compressible Representations

**Abstract:** We identify an issue in multi-task learnable compression, in which a representation learned for one task does not positively contribute to the rate-distortion performance of a different task as much as expected, given the estimated amount of information available in it. We interpret this issue using the predictive $\mathcal{V}$-information framework. In learnable scalable coding, previous work increased the utilization of side-information for input reconstruction by also rewarding input reconstruction when learning this shared representation. We evaluate the impact of this idea in the context of input reconstruction more rigorously and extended it to other computer vision tasks. We perform experiments using representations trained for object detection on COCO 2017 and depth estimation on the Cityscapes dataset, and use them to assist in image reconstruction and semantic segmentation tasks. The results show considerable improvements in the rate-distortion performance of the assisted tasks. Moreover, using the proposed representations, the performance of the base tasks are also improved. Results suggest that the proposed method induces simpler representations that are more compatible with downstream processes.

This is the official code release of this publication. It is under the MIT license. Please cite it as follows:

```bib
@inproceedings{TTCCR24,
  author       = {Anderson de Andrade and Ivan Bajić},
  title        = {Towards Task-Compatible Compressible Representations},
  booktitle    = {{IEEE} International Conference on Multimedia and Expo Workshop on Coding for Machines, {ICME CfM} 2024},
  year         = {2024},
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

Make sure your dependencies are properly defined in your `poetry.lock` configuration file. If you have made changes to it, do not forget to update your lock file by running:
```
make update
```
 To build your container image run:
```bash
make training-image
```

#### 2. Define entry point to launch experiments
Define entry points to your tasks in the `MLProject` file.

#### 3. Launch and Manage jobs
To launch and manage jobs using Slurm we created the following scripts: `slurm-launch`, `slurm-log`, `slurm-status`, and `slurm-cancel`. The documentation on how `slurm-launch` works is available by running:
```bash
slurm-launch --help
```

To launch jobs using the Kubernetes please use `k8s-launch`. Instructions for this script is available by running:
```bash
k8s-launch --help
```
To manage Kubernetes jobs we recommend to use [k9s](https://github.com/derailed/k9s).

