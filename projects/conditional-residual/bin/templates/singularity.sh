#!/bin/bash

module load singularity

mkdir -p "/scratch/${{USER}}/data"
mkdir -p "${{TMPDIR}}/{job_name}/data"
mkdir -p "${{TMPDIR}}/{job_name}/singularity/"{{tmp,workdir,home}}

export HTTP_PROXY="{proxy_url}"
export HTTPS_PROXY="{proxy_url}"
export SINGULARITY_DOCKER_USERNAME="{docker_username}"
export SINGULARITY_DOCKER_PASSWORD="{docker_password}"
export SINGULARITY_CACHEDIR="/scratch/${{USER}}/singularity"
export SINGULARITY_TMPDIR="${{TMPDIR}}/singularity/tmp"
export CUDA_VISIBLE_DEVICES="$( \
	nvidia-smi --query-gpu=utilization.memory --format=csv,noheader | \
	awk {{'print $1, NR-1'}} | \
	sort -k1n | \
	awk {{'print $2'}} | \
	head -{num_gpus} | \
	paste -sd ',' \
)"

nohup bash -s > "{job_name}.out" 2>&1 <<- EOM &
	singularity \
		run \
		--contain \
		--cleanenv \
		--nv \
		--workdir "${{TMPDIR}}/{job_name}/singularity/workdir" \
		--home "\${{SLURM_TMPDIR}}/singularity/home" \
		--bind "/scratch/${{USER}}/data":/data-remote \
		--bind "${{TMPDIR}}/{job_name}/data":/data-local \
		--env MLFLOW_TRACKING_URI="https://mlflow.multimedialabsfu.xyz" \
		--env MLFLOW_TRACKING_USERNAME="{mlflow_username}" \
		--env MLFLOW_TRACKING_PASSWORD="{mlflow_password}" \
		--env MLFLOW_S3_ENDPOINT_URL="https://io.multimedialabsfu.xyz" \
		--env AWS_ACCESS_KEY_ID="{aws_access_key_id}" \
		--env AWS_SECRET_ACCESS_KEY="{aws_secret_access_key}" \
		--env S3_ENDPOINT_URL="https://io.multimedialabsfu.xyz" \
		--env DATA_PATH="/data-remote" \
		--env DATA_EPHEMERAL_PATH="/data-local" \
		--env SLACK_URL="{slack_url}" \
		--env SLACK_USER="{slack_user}" \
		--env HTTP_PROXY="{proxy_url}" \
		--env HTTPS_PROXY="{proxy_url}" \
		--env DEVELOPMENT_MODE="{development_mode}" \
		--env CUDA_VISIBLE_DEVICES="${{CUDA_VISIBLE_DEVICES}}" \
		"docker://{training_image}" \
		{job_arguments}

	rm -rf "${{TMPDIR}}/{job_name}"
EOM

echo "Launched {job_name}"
