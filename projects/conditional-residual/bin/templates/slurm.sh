#!/bin/bash

launch_job(){{
	if [[ -n "${{PID}}" ]]; then
		kill -INT "$(pgrep -P "$(pgrep -P "${{PID}}" mlflow)")"
		wait "${{PID}}"
		echo "Task terminated gracefully."
	fi

	sbatch \
		--job-name {job_name} \
		--account {account} \
		--cpus-per-task {cpus} \
		--mem {memory} \
		--gpus-per-node {gpus} \
		--time {time} \
		--signal {signal} \
		<<- EOM
			#!/bin/bash

			$(type launch_job | sed '1d')
			trap launch_job USR1

			module load apptainer

			mkdir -p "/scratch/\${{USER}}/data"
			mkdir -p "\${{SLURM_TMPDIR}}/data"
			mkdir -p "\${{SLURM_TMPDIR}}/apptainer/"{{tmp,workdir,home}}

			export HTTP_PROXY="{proxy_url}"
			export HTTPS_PROXY="{proxy_url}"
			export APPTAINER_DOCKER_USERNAME="{docker_username}"
			export APPTAINER_DOCKER_PASSWORD="{docker_password}"
			export APPTAINER_CACHEDIR="/scratch/${{USER}}/apptainer"
			export APPTAINER_TMPDIR="\${{SLURM_TMPDIR}}/apptainer/tmp"

			apptainer \
				run \
				--contain \
				--cleanenv \
				--nv \
				--workdir "\${{SLURM_TMPDIR}}/apptainer/workdir" \
				--home "\${{SLURM_TMPDIR}}/apptainer/home" \
				--bind "/scratch/\${{USER}}/data":/data-remote \
				--bind "\${{SLURM_TMPDIR}}/data":/data-local \
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
				--env HTTP_PROXY="{proxy_url}" \
				--env HTTPS_PROXY="{proxy_url}" \
				--env DEVELOPMENT_MODE="{development_mode}" \
				--env JOB_ID="\${{SLURM_JOB_ID}}" \
				"docker://{training_image}" \
				{job_arguments} \
				&

			PID="\$!"

			wait
		EOM
}}

cd {directory} && launch_job
