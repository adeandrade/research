#!/bin/bash

cd {directory}

module load apptainer

mkdir -p "/scratch/${{USER}}/data"
mkdir -p "${{TMPDIR}}/{job_name}/data"
mkdir -p "${{TMPDIR}}/{job_name}/apptainer/"{{tmp,workdir,home}}

export APPTAINER_DOCKER_USERNAME="{docker_username}"
export APPTAINER_DOCKER_PASSWORD="{docker_password}"
export APPTAINER_CACHEDIR="/scratch/${{USER}}/apptainer"
export APPTAINER_TMPDIR="${{TMPDIR}}/{job_name}/apptainer/tmp"

nohup bash -s > "{job_name}.out" 2>&1 <<- EOM &
	source ~/.bash_profile
	genv activate --gpus {num_gpus}

	apptainer \
		run \
		--contain \
		--cleanenv \
		--nv \
		--workdir "\${{TMPDIR}}/{job_name}/apptainer/workdir" \
		--home "\${{TMPDIR}}/{job_name}/apptainer/home" \
		--bind "/scratch/\${{USER}}/data":/data-remote \
		--bind "\${{TMPDIR}}/{job_name}/data":/data-local \
		--env MLFLOW_TRACKING_URI="{mlflow_tracking_uri}" \
		--env MLFLOW_TRACKING_USERNAME="{mlflow_username}" \
		--env MLFLOW_TRACKING_PASSWORD="{mlflow_password}" \
		--env MLFLOW_S3_ENDPOINT_URL="{s3_endpoint_url}" \
		--env S3_ENDPOINT_URL="{s3_endpoint_url}" \
		--env AWS_ACCESS_KEY_ID="{aws_access_key_id}" \
		--env AWS_SECRET_ACCESS_KEY="{aws_secret_access_key}" \
		--env DATA_PATH="/data-remote" \
		--env DATA_EPHEMERAL_PATH="/data-local" \
		--env SLACK_URL="{slack_url}" \
		--env SLACK_USER="{slack_user}" \
		--env DEVELOPMENT_MODE="{development_mode}" \
		--env CUDA_VISIBLE_DEVICES="\${{CUDA_VISIBLE_DEVICES}}" \
		"docker://{training_image}" \
		{job_arguments}

	genv deactivate

	rm -rf "${{TMPDIR}}/{job_name}"
EOM

echo "Launched {job_name}"
